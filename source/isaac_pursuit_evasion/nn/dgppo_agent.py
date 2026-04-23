"""
Standalone DGPPO agent with a skrl-compatible public surface.

Implements the public contract that :class:`skrl.trainers.torch.SequentialTrainer`
(and the monkey-patched hooks in ``scripts/skrl/train.py``) rely on
(``init`` / ``act`` / ``record_transition`` / ``pre_interaction`` /
``post_interaction`` / ``write_checkpoint`` / ``load`` / ``set_running_mode`` /
``track_data`` / ``experiment_dir`` / ``value``) without inheriting from the
skrl ``Agent`` base. Decoupling from skrl internals makes the code robust
across skrl point releases; the trainer only ever sees duck-typed methods.

The algorithm itself follows ``dgppo/algo/dgppo.py`` (the JAX reference)
step-by-step, using the torch helpers already verified in
``train_dgppo.py`` (``compute_dec_ocp_gae`` / ``compute_cbf_advantages`` /
``compute_policy_surrogate``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F

from .dgppo_memory import DGPPORolloutMemory
from .dgppo_models import DGPPOCritic, DGPPOPolicy
from .graph_builder import GraphLayout, build_graph_data
from .train_dgppo import compute_cbf_advantages, compute_dec_ocp_gae, compute_policy_surrogate


# ---------------------------------------------------------------------------
# Env accessor -- tiny protocol the agent uses to read side-channel info
# ---------------------------------------------------------------------------


class EnvAccessor:
    """Thin adapter around an IsaacLab env exposing DGPPO-specific data.

    Centralising access here keeps the agent agnostic to whether it is
    talking to a single-agent ``DirectRLEnv`` or a multi-agent
    ``DirectMARLEnv`` in the future: the accessor always returns stacked
    tensors of the shapes the graph builder expects.
    """

    def __init__(self, base_env: Any) -> None:
        self._env = base_env

    @property
    def num_envs(self) -> int:
        return int(getattr(self._env, "num_envs"))

    @property
    def n_agents(self) -> int:
        # ``PosTrackingEnv`` is single-agent; multi-agent envs override.
        return int(getattr(self._env, "n_agents", 1))

    @property
    def n_obs(self) -> int:
        return int(getattr(self._env, "n_obstacles", 0))

    @property
    def n_constraints(self) -> int:
        return int(getattr(self._env, "n_constraints"))

    @property
    def state_dim(self) -> int:
        return int(getattr(self._env, "graph_state_dim"))

    @property
    def dt(self) -> float:
        return float(getattr(self._env, "step_dt"))

    def get_agent_state(self) -> torch.Tensor:
        return self._env.get_graph_agent_state()

    def get_goal_state(self) -> torch.Tensor:
        return self._env.get_graph_goal_state()

    def get_obs_state(self) -> torch.Tensor:
        return self._env.get_graph_obstacle_state()

    def get_costs(self) -> torch.Tensor:
        return self._env.get_costs()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DGPPOAgent:
    """DGPPO implemented on top of IsaacLab + skrl trainer.

    Args:
        policy: :class:`DGPPOPolicy` instance (already on the target device).
        critic: :class:`DGPPOCritic` instance.
        env_accessor: :class:`EnvAccessor` wrapping the unwrapped IsaacLab env.
        graph_layout: dimensionality of the graph the env produces.
        cfg: DGPPO hyper-parameters and experiment config (see
            ``skrl_dgppo_cfg.yaml`` for the expected schema).
        observation_space / action_space: gym spaces (kept for skrl
            compatibility / checkpoint metadata).
        device: torch device.
    """

    # Fraction of the total timesteps after which we run one update.
    #
    # ``post_interaction`` is called once per env step; we trigger an update
    # every ``rollouts`` steps so the rollout buffer fills exactly once.

    def __init__(
        self,
        policy: DGPPOPolicy,
        critic: DGPPOCritic,
        env_accessor: EnvAccessor,
        det_env: Any,
        det_env_accessor: EnvAccessor,
        graph_layout: GraphLayout,
        cfg: Mapping[str, Any],
        observation_space,
        action_space,
        device: torch.device,
    ) -> None:
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.env = env_accessor
        self.det_env = det_env
        self.det_env_accessor = det_env_accessor
        self.layout = graph_layout
        self.cfg = dict(cfg)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

        # Hyper-parameters (mirror the YAML schema).
        self.gamma: float = float(cfg.get("discount_factor", 0.99))
        self.gae_lambda: float = float(cfg.get("lambda", 0.95))
        self.rollouts: int = int(cfg.get("rollouts", 32))
        self.learning_epochs: int = int(cfg.get("learning_epochs", 8))
        self.mini_batches: int = int(cfg.get("mini_batches", 8))
        self.clip_eps: float = float(cfg.get("ratio_clip", 0.2))
        self.alpha: float = float(cfg.get("alpha", 10.0))
        self.cbf_eps: float = float(cfg.get("cbf_eps", 1e-2))
        self.cbf_weight: float = float(cfg.get("cbf_weight", 1.0))
        self.cbf_schedule: bool = bool(cfg.get("cbf_schedule", True))
        self.grad_clip: float = float(cfg.get("grad_norm_clip", 2.0))
        self.entropy_scale: float = float(cfg.get("entropy_loss_scale", 0.0))
        self.vl_loss_scale: float = float(cfg.get("vl_loss_scale", 1.0))
        self.vh_loss_scale: float = float(cfg.get("vh_loss_scale", 1.0))
        self.obs_radius: float = float(cfg.get("obs_radius", 2.0))
        self.lr_policy: float = float(cfg.get("lr_policy", 3e-4))
        self.lr_vl: float = float(cfg.get("lr_vl", 1e-3))
        self.lr_vh: float = float(cfg.get("lr_vh", 1e-3))

        self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        vl_params = list(self.critic.vl_gnn.parameters()) + list(self.critic.vl_head.mlp.parameters()) + list(
            self.critic.vl_head.value_out.parameters()
        )
        if self.critic.vl_head.rnn is not None:
            vl_params += list(self.critic.vl_head.rnn.parameters())
        vh_params = list(self.critic.vh_gnn.parameters()) + list(self.critic.vh_head.mlp.parameters()) + list(
            self.critic.vh_head.value_out.parameters()
        )
        if self.critic.vh_head.rnn is not None:
            vh_params += list(self.critic.vh_head.rnn.parameters())
        self._vl_opt = torch.optim.Adam(vl_params, lr=self.lr_vl)
        self._vh_opt = torch.optim.Adam(vh_params, lr=self.lr_vh)
        self._vl_grad_params = vl_params
        self._vh_grad_params = vh_params

        # Rollout memory (allocated in :meth:`init`).
        self.memory: Optional[DGPPORolloutMemory] = None
        self.det_memory: Optional[DGPPORolloutMemory] = None

        # Running-mode flag (mirrors skrl).
        self.training: bool = True

        # skrl-compatible bookkeeping.
        self.experiment_dir: str = ""
        self.tracking_data: dict[str, list[float]] = {}
        self._writer = None
        self._checkpoint_interval: int = 0
        self._write_interval: int = 0
        self._total_timesteps: int = 0

        # skrl trainers inspect ``agent.value`` when running eval helpers.
        # The helper in ``train.py`` expects ``value.act(inputs, role="value")``
        # returning ``(values, _, _)`` for a flat state tensor; we leave it as
        # ``None`` and the helper degrades gracefully.
        self.value = None

        # State/value preprocessors (skrl compat). Not used internally by
        # DGPPO since normalization is applied inside ``compute_cbf_advantages``.
        self._state_preprocessor: Callable = lambda x: x
        self._value_preprocessor: Callable = lambda x, inverse=False: x

        # Cached pre-step graph / costs so ``record_transition`` does not
        # have to re-read them (they were computed in ``act``).
        self._last_graph_snapshot: dict[str, torch.Tensor] = {}
        self._last_log_prob: Optional[torch.Tensor] = None
        self._last_value_l: Optional[torch.Tensor] = None
        self._last_value_h: Optional[torch.Tensor] = None
        self._last_rnn_state: Optional[torch.Tensor] = None
        self._last_mean_action: Optional[torch.Tensor] = None

        # Rollout-local RNN carry (one per agent across the env batch).
        self._rnn_state: Optional[torch.Tensor] = None
        self._stoch_env_ids: Optional[torch.Tensor] = None
        self._det_env_ids: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # skrl-compatible lifecycle
    # ------------------------------------------------------------------

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Called once by the trainer before the first interaction.

        Allocates the rollout memory, creates the experiment directory and
        the TensorBoard writer (so skrl's ``sync_tensorboard`` picks up our
        metrics in wandb), and resolves write/checkpoint intervals.

        Idempotent: skrl's trainer may call ``init`` after the runner has
        already set things up.
        """
        if self.memory is not None:
            return
        exp_cfg = self.cfg.get("experiment", {})
        base_dir = exp_cfg.get("directory", "logs/dgppo")
        name = exp_cfg.get("experiment_name", "dgppo")
        exp_dir = os.path.join(base_dir, name) if name else base_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.experiment_dir = exp_dir
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

        try:
            from torch.utils.tensorboard.writer import SummaryWriter

            self._writer = SummaryWriter(log_dir=exp_dir)
        except Exception:
            self._writer = None

        total_timesteps = 0
        if trainer_cfg is not None:
            total_timesteps = int(trainer_cfg.get("timesteps", 0))
        self._total_timesteps = total_timesteps

        write_interval = exp_cfg.get("write_interval", "auto")
        checkpoint_interval = exp_cfg.get("checkpoint_interval", "auto")
        self._write_interval = self._resolve_interval(write_interval, total_timesteps, default_fraction=250)
        self._checkpoint_interval = self._resolve_interval(
            checkpoint_interval, total_timesteps, default_fraction=10
        )

        # Allocate the rollout memory now that we know the env shape.
        if self.env.num_envs < 2 or (self.env.num_envs % 2) != 0:
            raise ValueError(
                f"DGPPO split rollout requires an even num_envs >= 2, got {self.env.num_envs}"
            )
        split = self.env.num_envs // 2
        self._det_env_ids = torch.arange(0, split, device=self.device, dtype=torch.long)
        self._stoch_env_ids = torch.arange(split, self.env.num_envs, device=self.device, dtype=torch.long)

        self.memory = DGPPORolloutMemory(
            rollout_length=self.rollouts,
            num_envs=int(self._stoch_env_ids.numel()),
            n_agents=self.layout.n_agents,
            n_obs=self.layout.n_obs,
            state_dim=self.layout.state_dim,
            action_dim=int(self.action_space.shape[-1]),
            n_constraints=self.layout.n_obs if False else self.env.n_constraints,
            device=self.device,
            use_rnn=self.policy.backbone.use_rnn,
        )
        # Deterministic rollout stream used for Vh targets (mirrors the
        # reference update that trains Vh on a deterministic-policy rollout).
        self.det_memory = DGPPORolloutMemory(
            rollout_length=self.rollouts,
            num_envs=int(self._det_env_ids.numel()),
            n_agents=self.layout.n_agents,
            n_obs=self.layout.n_obs,
            state_dim=self.layout.state_dim,
            action_dim=int(self.action_space.shape[-1]),
            n_constraints=self.layout.n_obs if False else self.env.n_constraints,
            device=self.device,
            use_rnn=self.policy.backbone.use_rnn,
        )

        # Initial RNN carry for the rollout (B * A agents, regardless of env).
        if self.policy.backbone.use_rnn:
            self._rnn_state = self.policy.initialize_carry(
                n_agents_total=self.env.num_envs * self.layout.n_agents, device=self.device
            )

    @staticmethod
    def _resolve_interval(value: Any, total_timesteps: int, *, default_fraction: int) -> int:
        if value == "auto":
            if total_timesteps <= 0:
                return 0
            return max(1, total_timesteps // default_fraction)
        if value is None:
            return 0
        return int(value)

    def _cbf_scale(self, timestep: int) -> float:
        """Mirror the reference piecewise CBF schedule when enabled."""
        if not self.cbf_schedule:
            return self.cbf_weight
        if self._total_timesteps <= 0:
            return self.cbf_weight
        scale = self.cbf_weight
        half = int(self._total_timesteps * 0.5)
        three_quarter = int(self._total_timesteps * 0.75)
        if timestep >= half:
            scale *= 2.0
        if timestep >= three_quarter:
            scale *= 2.0
        return scale

    def set_running_mode(self, mode: str) -> None:
        assert mode in ("train", "eval")
        self.training = mode == "train"
        self.policy.train(self.training)
        self.critic.train(self.training)

    def track_data(self, tag: str, value: float) -> None:
        self.tracking_data.setdefault(tag, []).append(float(value))

    def _flush_tracking(self, timestep: int) -> None:
        if self._writer is None:
            return
        for tag, values in self.tracking_data.items():
            if not values:
                continue
            mean_val = sum(values) / len(values)
            try:
                self._writer.add_scalar(tag, mean_val, timestep)
            except Exception:
                pass
        self.tracking_data.clear()

    # ------------------------------------------------------------------
    # Rollout: act / record_transition / pre / post
    # ------------------------------------------------------------------

    def _build_graph(self) -> tuple[Any, dict[str, torch.Tensor]]:
        """Pull per-env state from the env and assemble a batched graph."""
        agent_s = self.env.get_agent_state()
        goal_s = self.env.get_goal_state()
        obs_s = self.env.get_obs_state()
        graph = build_graph_data(agent_s, goal_s, obs_s, obs_radius=self.obs_radius)
        snapshot = {"agent": agent_s, "goal": goal_s, "obs": obs_s}
        return graph, snapshot

    def _compute_values(self, graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward both critic heads, producing ``(Vl[B], Vh[B, A, NH])``."""
        vl, _ = self.critic.get_vl(graph, rnn_state=None, n_agents=self.layout.n_agents)
        vh, _ = self.critic.get_vh(graph, rnn_state=None, n_agents=self.layout.n_agents)
        vl = vl.squeeze(-1).squeeze(-1)  # (E, 1, 1) -> (E,)
        return vl, vh

    def act(
        self,
        states: torch.Tensor,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Sample an action from the policy. ``states`` is ignored (flat obs
        tensor from skrl's wrapper; we read the graph directly from the env).
        """
        del states, timestep, timesteps
        with torch.no_grad():
            graph, snapshot = self._build_graph()
            n_total = self.env.num_envs * self.layout.n_agents
            action, log_prob, mean_action, new_rnn = self.policy.act(
                graph,
                rnn_state=self._rnn_state,
                n_agents_total=n_total,
                deterministic=not self.training,
            )
            vl, vh = self._compute_values(graph)

        # Cache everything needed by ``record_transition``.
        self._last_graph_snapshot = snapshot
        self._last_log_prob = log_prob.detach()
        self._last_value_l = vl.detach()
        self._last_value_h = vh.detach()
        self._last_mean_action = mean_action.detach()
        if self.policy.backbone.use_rnn:
            self._last_rnn_state = (
                self._rnn_state.detach() if self._rnn_state is not None else None
            )
            self._rnn_state = new_rnn.detach() if new_rnn is not None else None

        # skrl expects (action, log_prob, outputs_dict). We also stash the
        # deterministic mean for eval-time use (see ``train.py`` line 926).
        assert self._det_env_ids is not None and self._stoch_env_ids is not None
        action_mixed = action.clone()
        action_mixed[self._det_env_ids] = mean_action[self._det_env_ids]
        log_prob_mixed = torch.zeros_like(log_prob)
        log_prob_mixed[self._stoch_env_ids] = log_prob[self._stoch_env_ids]

        action_flat = action_mixed.reshape(self.env.num_envs, -1)
        mean_flat = mean_action.reshape(self.env.num_envs, -1)
        return action_flat, log_prob_mixed.reshape(self.env.num_envs, -1), {"mean_actions": mean_flat}

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        # Nothing to do; included for skrl lifecycle compatibility.
        return

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        del states, next_states, infos, timestep, timesteps
        assert self.memory is not None and self.det_memory is not None
        assert self._det_env_ids is not None and self._stoch_env_ids is not None
        # Actions arrive flat ``(B, Da)``; reshape to ``(B, A, Da)``.
        B = self.env.num_envs
        A = self.layout.n_agents
        action = actions.reshape(B, A, -1)

        cost = self.env.get_costs()  # (B, A, NH)

        # ``rewards`` may be ``(B,)`` or ``(B, 1)`` depending on wrapper.
        reward = rewards.reshape(B).to(self.device)

        stoch_ids = self._stoch_env_ids
        det_ids = self._det_env_ids
        self.memory.add(
            agent_state=self._last_graph_snapshot["agent"][stoch_ids],
            goal_state=self._last_graph_snapshot["goal"][stoch_ids],
            obs_state=self._last_graph_snapshot["obs"][stoch_ids],
            action=action[stoch_ids].to(self.device),
            log_prob=(
                self._last_log_prob[stoch_ids].to(self.device)
                if self._last_log_prob is not None
                else torch.zeros(stoch_ids.numel(), A, device=self.device)
            ),
            reward=reward[stoch_ids],
            cost=cost[stoch_ids].to(self.device),
            value_l=(
                self._last_value_l[stoch_ids]
                if self._last_value_l is not None
                else torch.zeros(stoch_ids.numel(), device=self.device)
            ),
            value_h=(
                self._last_value_h[stoch_ids]
                if self._last_value_h is not None
                else torch.zeros(stoch_ids.numel(), A, self.env.n_constraints, device=self.device)
            ),
            rnn_state=None,
        )
        self.det_memory.add(
            agent_state=self._last_graph_snapshot["agent"][det_ids],
            goal_state=self._last_graph_snapshot["goal"][det_ids],
            obs_state=self._last_graph_snapshot["obs"][det_ids],
            action=action[det_ids].to(self.device),
            log_prob=torch.zeros(det_ids.numel(), A, device=self.device),
            reward=reward[det_ids],
            cost=cost[det_ids].to(self.device),
            value_l=(
                self._last_value_l[det_ids]
                if self._last_value_l is not None
                else torch.zeros(det_ids.numel(), device=self.device)
            ),
            value_h=(
                self._last_value_h[det_ids]
                if self._last_value_h is not None
                else torch.zeros(det_ids.numel(), A, self.env.n_constraints, device=self.device)
            ),
            rnn_state=None,
        )

        # Reset RNN carry for envs that just ended (mirrors the common PPO
        # recurrent convention).
        if self.policy.backbone.use_rnn and self._rnn_state is not None:
            done = (terminated | truncated).reshape(-1).to(self.device)
            if done.any():
                # ``_rnn_state`` shape: (L, B*A, C, H). With A=1 the indexing
                # is trivial; with A>1 each done-env resets all its agents.
                done_idx = torch.nonzero(done, as_tuple=False).flatten()
                # Expand env ids to the agent-flat axis.
                agent_ids = (
                    done_idx[:, None] * A + torch.arange(A, device=self.device)[None, :]
                ).reshape(-1)
                self._rnn_state[:, agent_ids] = 0.0

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Drive training, logging, and periodic checkpointing."""
        assert self.memory is not None and self.det_memory is not None
        assert self._det_env_ids is not None and self._stoch_env_ids is not None

        if self.memory.is_full:
            # Bootstrap values at the terminal step.
            with torch.no_grad():
                graph, _ = self._build_graph()
                vl_boot, vh_boot = self._compute_values(graph)
            self.memory.set_final_values(vl_boot[self._stoch_env_ids], vh_boot[self._stoch_env_ids])
            self.det_memory.set_final_values(vl_boot[self._det_env_ids], vh_boot[self._det_env_ids])
            self._update(timestep, timesteps)

            self.memory.reset()
            self.det_memory.reset()
            # Reset RNN carry between rollouts to avoid stale gradients.
            if self.policy.backbone.use_rnn:
                self._rnn_state = self.policy.initialize_carry(
                    n_agents_total=self.env.num_envs * self.layout.n_agents, device=self.device
                )

        if self._write_interval > 0 and (timestep + 1) % self._write_interval == 0:
            self._flush_tracking(timestep + 1)

        if self._checkpoint_interval > 0 and (timestep + 1) % self._checkpoint_interval == 0:
            self.write_checkpoint(timestep + 1, timesteps)

    # ------------------------------------------------------------------
    # DGPPO update (mirrors dgppo/algo/dgppo.py::update_inner)
    # ------------------------------------------------------------------

    def _update(self, timestep: int, timesteps: int) -> None:
        assert self.memory is not None
        mem = self.memory
        det_mem = self.det_memory
        assert det_mem is not None
        view = mem.as_bTah_view()
        det_view = det_mem.as_bTah_view()

        # (1) GAE targets for Qh (per-agent, per-head) and Ql (scalar reward).
        Qh, Ql = compute_dec_ocp_gae(
            Tah_hs=view["bTah_hs"],
            T_l=view["bT_l"],
            Tp1ah_Vh=view["bTp1ah_Vh"],
            Tp1_Vl=view["bTp1_Vl"],
            disc_gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        # Deterministic rollout target stream for Vh (reference DGPPO logic).
        Qh_det, _ = compute_dec_ocp_gae(
            Tah_hs=det_view["bTah_hs"],
            T_l=det_view["bT_l"],
            Tp1ah_Vh=det_view["bTp1ah_Vh"],
            Tp1_Vl=view["bTp1_Vl"],
            disc_gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # (2) CBF-based advantage (already flipped for PPO use).
        adv_info = compute_cbf_advantages(
            bT_Ql=Ql,
            bT_Vl=view["bT_Vl"],
            bTah_Vh=view["bTah_Vh"],
            bTp1ah_Vh=view["bTp1ah_Vh"],
            alpha=self.alpha,
            cbf_eps=self.cbf_eps,
            cbf_weight=self.cbf_weight,
            dt=self.env.dt,
            cbf_scale=self._cbf_scale(timestep),
        )
        bTa_A = adv_info["bTa_A"].detach()
        self.track_data("DGPPO/safe_rate", float(adv_info["bTa_is_safe"].float().mean().item()))
        self.track_data("DGPPO/adv_raw_mean", float(adv_info["bT_Al_raw"].mean().item()))

        # (3) PPO epochs with minibatching over the env axis.
        loss_p_acc = loss_vl_acc = loss_vh_acc = clipfrac_acc = 0.0
        n_minibatches = 0

        for _ in range(self.learning_epochs):
            for idx in mem.minibatch_iter(self.mini_batches):
                info = self._update_minibatch(
                    idx=idx,
                    Qh_det=Qh_det,
                    Ql=Ql,
                    bTa_A=bTa_A,
                    view=view,
                    det_view=det_view,
                )
                loss_p_acc += info["loss_p"]
                loss_vl_acc += info["loss_vl"]
                loss_vh_acc += info["loss_vh"]
                clipfrac_acc += info["clip_frac"]
                n_minibatches += 1

        if n_minibatches > 0:
            self.track_data("DGPPO/loss_policy", loss_p_acc / n_minibatches)
            self.track_data("DGPPO/loss_value_l", loss_vl_acc / n_minibatches)
            self.track_data("DGPPO/loss_value_h", loss_vh_acc / n_minibatches)
            self.track_data("DGPPO/clip_frac", clipfrac_acc / n_minibatches)
        self.track_data("DGPPO/lr_policy", self.lr_policy)

    def _update_minibatch(
        self,
        *,
        idx: torch.Tensor,
        Qh_det: torch.Tensor,
        Ql: torch.Tensor,
        bTa_A: torch.Tensor,
        view: dict[str, torch.Tensor],
        det_view: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Single PPO minibatch step over one chunk of the ``B`` axis."""
        # Gather the minibatch (axes: B first after ``as_bTah_view``).
        agent_s = view["bTa_agent_state"][idx]       # (b, T, A, S)
        goal_s = view["bTa_goal_state"][idx]
        obs_s = view["bTo_obs_state"][idx]
        actions = view["bTa_actions"][idx]           # (b, T, A, Da)
        old_logp = view["bTa_logp"][idx]             # (b, T, A)
        Qh_det_mb = Qh_det[idx]                      # (b, T, A, NH)
        Ql_mb = Ql[idx]                              # (b, T)
        adv_mb = bTa_A[idx]                          # (b, T, A)

        b, T, A, _ = actions.shape
        BT = b * T

        # Flatten (B, T) -> (BT) sub-graphs and rebuild a single batched graph.
        graph = build_graph_data(
            agent_state=agent_s.reshape(BT, A, -1),
            goal_state=goal_s.reshape(BT, A, -1),
            obs_state=obs_s.reshape(BT, obs_s.shape[2], -1),
            obs_radius=self.obs_radius,
        )
        n_agents_total = BT * A

        # ---- Policy ----
        log_prob, entropy, _ = self.policy.evaluate(
            graph,
            action=actions.reshape(n_agents_total, -1),
            rnn_state=None,
            n_agents_total=n_agents_total,
        )
        log_prob = log_prob.reshape(b, T, A)
        ratio = torch.exp(log_prob - old_logp.detach())
        surrogate = compute_policy_surrogate(ratio, adv_mb, self.clip_eps)
        loss_policy = surrogate["loss_policy"]
        entropy_bonus = -self.entropy_scale * entropy.mean() if self.entropy_scale > 0 else 0.0
        loss_p_total = loss_policy + entropy_bonus

        self._policy_opt.zero_grad()
        loss_p_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self._policy_opt.step()

        # ---- Critic Vl ----
        vl, _ = self.critic.get_vl(graph, rnn_state=None, n_agents=A)
        vl = vl.reshape(b, T)
        loss_vl = self.vl_loss_scale * F.mse_loss(vl, Ql_mb)

        self._vl_opt.zero_grad()
        loss_vl.backward()
        torch.nn.utils.clip_grad_norm_(
            self._vl_grad_params,
            self.grad_clip,
        )
        self._vl_opt.step()

        # ---- Critic Vh ----
        det_agent_s = det_view["bTa_agent_state"][idx]
        det_goal_s = det_view["bTa_goal_state"][idx]
        det_obs_s = det_view["bTo_obs_state"][idx]
        det_graph = build_graph_data(
            agent_state=det_agent_s.reshape(BT, A, -1),
            goal_state=det_goal_s.reshape(BT, A, -1),
            obs_state=det_obs_s.reshape(BT, det_obs_s.shape[2], -1),
            obs_radius=self.obs_radius,
        )
        vh, _ = self.critic.get_vh(det_graph, rnn_state=None, n_agents=A)
        vh = vh.reshape(b, T, A, -1)
        loss_vh = self.vh_loss_scale * F.mse_loss(vh, Qh_det_mb)

        self._vh_opt.zero_grad()
        loss_vh.backward()
        torch.nn.utils.clip_grad_norm_(
            self._vh_grad_params,
            self.grad_clip,
        )
        self._vh_opt.step()

        return {
            "loss_p": float(loss_policy.item()),
            "loss_vl": float(loss_vl.item()),
            "loss_vh": float(loss_vh.item()),
            "clip_frac": float(surrogate["clip_frac"].item()),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_path(self, timestep: int) -> Path:
        return Path(self.experiment_dir) / "checkpoints" / f"agent_{timestep}.pt"

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        del timesteps
        path = self._checkpoint_path(timestep)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(path))

    def save(self, path: str) -> None:
        state = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "policy_opt": self._policy_opt.state_dict(),
            "vl_opt": self._vl_opt.state_dict(),
            "vh_opt": self._vh_opt.state_dict(),
            "cfg": self.cfg,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])
        self.critic.load_state_dict(state["critic"])
        if "policy_opt" in state:
            self._policy_opt.load_state_dict(state["policy_opt"])
        if "vl_opt" in state:
            self._vl_opt.load_state_dict(state["vl_opt"])
        if "vh_opt" in state:
            self._vh_opt.load_state_dict(state["vh_opt"])
