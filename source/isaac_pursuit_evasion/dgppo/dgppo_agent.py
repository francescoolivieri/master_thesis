import dataclasses
import time
from collections.abc import Mapping
from typing import Any

import torch
from skrl.agents.torch import Agent

from .dgppo_config import DGPPOAgentCfg
from .dgppo_memory import DGPPORolloutMemory
from .dgppo_models import DGPPOPolicy, DGPPOValueNet
from .update_helpers import (
    apply_policy_update,
    apply_value_update,
    build_rollout_graph,
    build_update_graph_batch,
    compute_policy_loss,
    compute_rollout_policy_loss,
    compute_value_losses,
    evaluate_rollout_vh_values,
    evaluate_vh_batch_from_states,
    evaluate_vl_batch_from_states,
    rollout_graph_chunks,
    scan_rollout_vl_values,
)
from .utils import (
    GraphData,
    align_safety_cost_heads,
    build_graph_data,
    compute_cbf_advantages,
    compute_dec_ocp_gae,
    extract_graph_states_from_flat_obs,
    zero_env_rnn_states_for_done,
    zero_policy_rnn_states_for_done,
)


@dataclasses.dataclass(frozen=True)
class EnvSplit:
    """Environment ids used for DG-PPO's paired deterministic/stochastic rollout."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor


@dataclasses.dataclass(frozen=True)
class RolloutStep:
    """One collected environment step, before it is written to rollout memory."""

    agent_state: torch.Tensor
    goal_state: torch.Tensor
    obs_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    log_prob: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    policy_rnn_state: torch.Tensor | None
    value_l_rnn_state: torch.Tensor | None

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


@dataclasses.dataclass(frozen=True)
class ActCache:
    """Policy/value outputs from act(), consumed by the following transition record."""

    agent_state: torch.Tensor
    goal_state: torch.Tensor
    obs_state: torch.Tensor
    log_prob: torch.Tensor
    policy_rnn_state: torch.Tensor | None
    value_l_rnn_state: torch.Tensor | None


@dataclasses.dataclass(frozen=True)
class UpdateTargets:
    """Returns and advantages consumed by the PPO minibatch loop."""

    qh_det: torch.Tensor
    ql: torch.Tensor
    ql_value_targets: torch.Tensor
    bT_vl: torch.Tensor
    advantages: torch.Tensor
    adv_info: dict[str, torch.Tensor]
    vl_error: torch.Tensor


@dataclasses.dataclass(frozen=True)
class UpdateStats:
    """Accumulated scalar tensors from all minibatches in one update."""

    loss_policy: torch.Tensor
    loss_value_l: torch.Tensor
    loss_value_h: torch.Tensor
    clip_frac: torch.Tensor
    n_minibatches: int


class DGPPOAgent(Agent):
    def __init__(
        self,
        policy: DGPPOPolicy,
        Vl: DGPPOValueNet,
        Vh: DGPPOValueNet,
        env: Any,
        cfg: DGPPOAgentCfg,
        observation_space,
        state_space,
        action_space,
        device: torch.device,
    ) -> None:

        super().__init__(
            models={"policy": policy, "Vl": Vl, "Vh": Vh},
            memory=None,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

        self.policy = policy.to(device)
        self.Vl = Vl.to(device)
        self.Vh = Vh.to(device)
        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

        self._load_hyperparameters_from_cfg()
        self._setup_optimizers()
        self._register_checkpoint_modules()
        self._reset_runtime_state()

    def _setup_optimizers(self) -> None:
        self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        vl_params = self._value_optimizer_params(self.Vl)
        vh_params = self._value_optimizer_params(self.Vh)
        self._vl_opt = torch.optim.Adam(vl_params, lr=self.lr_vl)
        self._vh_opt = torch.optim.Adam(vh_params, lr=self.lr_vh)
        self._vl_grad_params = vl_params
        self._vh_grad_params = vh_params

    @staticmethod
    def _value_optimizer_params(value_net: DGPPOValueNet) -> list[torch.nn.Parameter]:
        params = (
            list(value_net.gnn.parameters())
            + list(value_net.head.parameters())
            + list(value_net.net.value_out.parameters())
        )
        if value_net.rnn is not None:
            params += list(value_net.rnn.parameters())
        return params

    def _register_checkpoint_modules(self) -> None:
        self.checkpoint_modules = {
            "policy": self.policy,
            "Vl": self.Vl,
            "Vh": self.Vh,
            "policy_optimizer": self._policy_opt,
            "Vl_optimizer": self._vl_opt,
            "Vh_optimizer": self._vh_opt,
        }

    def _reset_runtime_state(self) -> None:
        self.memory: DGPPORolloutMemory | None = None
        self.num_envs = self.cfg.get("num_envs", 1)
        self._rollout = 0
        self._policy_rnn_state = None
        self._vl_rnn_state = None
        self._env_split = self._make_env_split()
        self._act_cache: ActCache | None = None

    def _make_env_split(self) -> EnvSplit:
        if self.env.num_envs < 2 or (self.env.num_envs % 2) != 0:
            raise ValueError(f"DGPPO split rollout requires an even num_envs >= 2, got {self.env.num_envs}")
        split = self.env.num_envs // 2
        return EnvSplit(
            deterministic=torch.arange(0, split, device=self.device, dtype=torch.long),
            stochastic=torch.arange(split, self.env.num_envs, device=self.device, dtype=torch.long),
        )

    def init(self, *, trainer_cfg: Any | None = None) -> None:
        """
        Called once by the trainer before the first interaction.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        if self.memory is not None:
            return

        # Rollout memory dimensions.
        rollout_length = int(self.rollouts)
        n_agents = self.env.num_agents
        layout = self.env.unwrapped.graph_obs_layout
        n_obs = int(layout.get("n_obstacles", 0))
        state_dim = int(layout["state_dim"])
        action_dim = self.env.action_space.shape[0]
        n_constraints = int(getattr(self.env, "n_constraints", getattr(self.env.unwrapped, "n_constraints", 1)))
        use_rnn = self.policy.use_rnn

        self.memory = DGPPORolloutMemory(
            rollout_length=rollout_length,
            num_det_envs=int(self._env_split.deterministic.numel()),
            num_stc_envs=int(self._env_split.stochastic.numel()),
            n_agents=n_agents,
            n_obs=n_obs,
            state_dim=state_dim,
            action_dim=action_dim,
            n_constraints=n_constraints,
            device=self.device,
            use_rnn=use_rnn,
            use_vl_rnn=self.Vl.rnn is not None,
            rnn_layers=int(self.cfg.rnn.get("layers", 1)),
            rnn_hidden=int(self.cfg.rnn.get("hidden", 64)),
            rnn_cell=str(self.cfg.rnn.get("cell", "gru")),
        )

        # Policy/Vh carry is per env-agent sequence; Vl carry is per env sequence.
        if self.policy.use_rnn:
            self._policy_rnn_state = self.policy.initialize_carry(
                num_sequences=self.env.num_envs * n_agents, device=self.device
            )
        if self.Vl.rnn is not None:
            self._vl_rnn_state = self.Vl.rnn.initialize_carry(self.env.num_envs, device=self.device)

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample actions (and compute values if training) for the current environment step.

        The graph topology, safety adapter, policy, and critics use raw physical
        units decoded from IsaacLab's flat observation.

        :param observations: Per-agent observations (E*A, obs_dim) — unused directly.
        :param states:       Global env state (E, state_dim)       — unused directly.
        :param timestep:     Current timestep.
        :param timesteps:    Total timesteps.
        :return: (actions (E*A, action_dim), extras dict with log_prob and mean_action).
        """
        env_split = self._env_split

        with torch.no_grad():  # Saves memory
            agent_state, goal_state, obs_state = self._extract_graph_states(observations)
            raw_graph = self._build_graph_from_states(
                agent_state=agent_state,
                goal_state=goal_state,
                obs_state=obs_state,
            )
            policy_rnn_state = None if self._policy_rnn_state is None else self._policy_rnn_state.detach()
            value_l_rnn_state = None

            action, log_prob, mean_action, new_rnn = self.policy.act(
                raw_graph,
                rnn_state=self._policy_rnn_state,
                deterministic=not self.training,
            )

            if self.training:
                value_l_rnn_state = None if self._vl_rnn_state is None else self._vl_rnn_state.detach()
                _, self._vl_rnn_state = self.Vl(raw_graph, self._vl_rnn_state)

        # Carry the RNN state forward to the next timestep
        if new_rnn is not None:
            self._policy_rnn_state = new_rnn

        action_dim = action.shape[-1]
        action_by_env = action.reshape(self.env.num_envs, self.env.num_agents, action_dim)
        mean_by_env = mean_action.reshape(self.env.num_envs, self.env.num_agents, action_dim)
        log_prob_by_env = log_prob.reshape(self.env.num_envs, self.env.num_agents)

        action_by_env[env_split.deterministic] = mean_by_env[env_split.deterministic]
        log_prob_by_env[env_split.deterministic] = 0.0  # not used for deterministic envs

        action = action_by_env.reshape(-1, action_dim)
        log_prob = log_prob_by_env.reshape(-1)
        if self.training:
            self._act_cache = ActCache(
                agent_state=agent_state,
                goal_state=goal_state,
                obs_state=obs_state,
                log_prob=log_prob,
                policy_rnn_state=policy_rnn_state,
                value_l_rnn_state=value_l_rnn_state,
            )
        else:
            self._act_cache = None

        action_flat = action.reshape(self.env.num_envs, -1)
        mean_flat = mean_action.reshape(self.env.num_envs, -1)

        return action_flat, {"log_prob": log_prob, "mean_action": mean_flat}

    def set_running_mode(self, mode: str) -> None:
        # Needed for compatibility, since our models are not skrl "Model" subclasses
        self.training = mode == "train"
        self.policy.train(self.training)  # nn.Module.train()
        self.Vl.train(self.training)
        self.Vh.train(self.training)

    # Transition recording and update hooks.

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """

        record_actions = actions
        record_rewards = rewards

        if self.training:
            step = self._collect_rollout_step(
                observations=observations,
                actions=record_actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
            )
            record_rewards = step.rewards
            self._store_rollout_step(step)
            self._reset_rnn_states_for_done(step.done)
            self._store_final_rollout_state_if_full(next_observations=next_observations)

        # To handle skrl bookkeeping after DG-PPO has selected its aligned reward signal.
        super().record_transition(
            observations=observations,
            states=states,
            actions=record_actions,
            rewards=record_rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

    def _collect_rollout_step(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> RolloutStep:
        """Decode the latest env interaction into tensors with rollout-memory shapes."""
        cache = self._act_cache
        if cache is None:
            raise RuntimeError("DG-PPO record_transition expected cached policy/value outputs from act().")

        n_envs = self.env.num_envs
        n_agents = self.env.num_agents
        action_dim = self.env.action_space.shape[0]
        n_constraints = int(getattr(self.env, "n_constraints", getattr(self.env.unwrapped, "n_constraints", 1)))
        shaped_actions = actions.reshape(n_envs, n_agents, action_dim)
        shaped_rewards = self._rewards_from_observation_or_env(
            observations=observations,
            actions=actions,
            rewards=rewards,
            n_envs=n_envs,
        )
        if self.rewards_shaper_scale != 1.0:
            shaped_rewards = shaped_rewards * self.rewards_shaper_scale

        return RolloutStep(
            agent_state=cache.agent_state,
            goal_state=cache.goal_state,
            obs_state=cache.obs_state,
            actions=shaped_actions,
            rewards=shaped_rewards,
            costs=self._costs_from_observation(
                observations=observations,
                n_envs=n_envs,
                n_agents=n_agents,
                n_constraints=n_constraints,
            ),
            log_prob=cache.log_prob.reshape(n_envs, n_agents),
            terminated=self._env_done_mask(terminated, n_envs),
            truncated=self._env_done_mask(truncated, n_envs),
            policy_rnn_state=cache.policy_rnn_state,
            value_l_rnn_state=cache.value_l_rnn_state,
        )

    def _store_rollout_step(self, step: RolloutStep) -> None:
        """Append one collected step to stochastic and deterministic memory splits."""
        memory = self.memory
        assert memory is not None
        env_split = self._env_split
        stc = env_split.stochastic
        det = env_split.deterministic

        stc_rnn_state = self._select_policy_rnn_envs(step.policy_rnn_state, stc)
        det_rnn_state = self._select_policy_rnn_envs(step.policy_rnn_state, det)
        if memory.cursor == 0:
            memory.set_initial_vl_state("stc", self._select_env_rnn_envs(step.value_l_rnn_state, stc))
            memory.set_initial_vl_state("det", self._select_env_rnn_envs(step.value_l_rnn_state, det))

        memory.add(
            stc_agent_state=step.agent_state[stc],
            stc_goal_state=step.goal_state[stc],
            stc_obs_state=step.obs_state[stc],
            stc_action=step.actions[stc],
            stc_log_prob=step.log_prob[stc],
            stc_reward=step.rewards[stc],
            stc_cost=step.costs[stc],
            stc_terminated=step.terminated[stc],
            stc_truncated=step.truncated[stc],
            det_agent_state=step.agent_state[det],
            det_goal_state=step.goal_state[det],
            det_obs_state=step.obs_state[det],
            det_action=step.actions[det],
            det_log_prob=step.log_prob[det],
            det_reward=step.rewards[det],
            det_cost=step.costs[det],
            det_terminated=step.terminated[det],
            det_truncated=step.truncated[det],
            stc_rnn_state=stc_rnn_state,
            det_rnn_state=det_rnn_state,
        )

    def _store_final_rollout_state_if_full(
        self,
        *,
        next_observations: torch.Tensor,
    ) -> None:
        """Store final rollout state once the rollout memory is full."""
        memory = self.memory
        assert memory is not None
        if not memory.is_full:
            return

        env_split = self._env_split
        with torch.no_grad():
            next_agent_state, next_goal_state, next_obs_state = self._extract_graph_states(next_observations)
        memory.set_final_state(
            "stc",
            agent_state=next_agent_state[env_split.stochastic],
            goal_state=next_goal_state[env_split.stochastic],
            obs_state=next_obs_state[env_split.stochastic],
        )
        memory.set_final_state(
            "det",
            agent_state=next_agent_state[env_split.deterministic],
            goal_state=next_goal_state[env_split.deterministic],
            obs_state=next_obs_state[env_split.deterministic],
        )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """

        if self.training:
            self._rollout += 1
            if (self._rollout % self.rollouts) == 0 and timestep >= self.learning_starts:
                t0 = time.perf_counter()
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", (time.perf_counter() - t0) * 1000.0)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        memory = self.memory
        if memory is None or memory.cursor < self.rollouts:
            return

        view = memory.as_bTah_view("stc")
        det_view = memory.as_bTah_view("det")

        graph = build_rollout_graph(view=view, obs_radius=self.obs_radius)
        det_graph = build_rollout_graph(view=det_view, obs_radius=self.obs_radius)
        final_graph = self._build_final_rollout_graph(view)
        det_final_graph = self._build_final_rollout_graph(det_view)
        chunk_ids = self._rnn_chunk_ids(T=view["bTa_actions"].shape[1], device=torch.device("cpu"))
        targets = self._compute_update_targets(
            view=view,
            det_view=det_view,
            graph=graph,
            det_graph=det_graph,
            final_graph=final_graph,
            det_final_graph=det_final_graph,
            timestep=timestep,
            timesteps=timesteps,
        )
        self._track_update_targets(view=view, targets=targets)

        stats = self._run_update_epochs(
            memory=memory,
            initial_targets=targets,
            view=view,
            det_view=det_view,
            graph=graph,
            det_graph=det_graph,
            final_graph=final_graph,
            det_final_graph=det_final_graph,
            chunk_ids=chunk_ids,
            timestep=timestep,
            timesteps=timesteps,
        )
        if stats.n_minibatches == 0:
            return

        update_summary = self._summarize_update(stats)
        self._track_update_summary(update_summary)

        memory.reset()

    def _compute_update_targets(
        self,
        *,
        view: dict[str, torch.Tensor],
        det_view: dict[str, torch.Tensor],
        graph: GraphData,
        det_graph: GraphData,
        final_graph: GraphData,
        det_final_graph: GraphData,
        timestep: int,
        timesteps: int,
    ) -> UpdateTargets:
        """Compute rollout returns and CBF-shaped PPO advantages."""
        with torch.no_grad():
            B, T = view["bT_l"].shape
            det_B, det_T = det_view["bT_l"].shape

            vl_model, _, final_vl_state = scan_rollout_vl_values(
                Vl=self.Vl,
                graph=graph,
                B=B,
                T=T,
                initial_rnn_state=view.get("b_initial_vl_rnn_state"),
                done_mask=view.get("bT_done"),
            )
            final_vl = evaluate_vl_batch_from_states(
                Vl=self.Vl,
                graph=final_graph,
                rnn_states=final_vl_state,
            )
            vl = vl_model
            vh = evaluate_rollout_vh_values(
                Vh=self.Vh,
                graph=graph,
                B=B,
                T=T,
                rnn_states=view.get("bTa_rnn_states"),
            )
            final_rnn_state = self._advance_final_policy_state(final_graph, view)
            final_vh = evaluate_vh_batch_from_states(
                Vh=self.Vh,
                graph=final_graph,
                rnn_states=final_rnn_state,
            )
            vh_det = evaluate_rollout_vh_values(
                Vh=self.Vh,
                graph=det_graph,
                B=det_B,
                T=det_T,
                rnn_states=det_view.get("bTa_rnn_states"),
            )
            det_final_rnn_state = self._advance_final_policy_state(det_final_graph, det_view)
            final_vh_det = evaluate_vh_batch_from_states(
                Vh=self.Vh,
                graph=det_final_graph,
                rnn_states=det_final_rnn_state,
            )

            vl_tp1 = torch.cat([vl, final_vl[:, None]], dim=1)
            vh_tp1 = torch.cat([vh, final_vh[:, None]], dim=1)
            vh_det_tp1 = torch.cat([vh_det, final_vh_det[:, None]], dim=1)

            _qh_stc, ql = compute_dec_ocp_gae(
                Tah_hs=view["bTah_hs"],
                T_l=view["bT_l"],
                Tp1ah_Vh=vh_tp1,
                Tp1_Vl=vl_tp1,
                disc_gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                T_terminated=view["bT_terminated"],
                T_truncated=view["bT_truncated"],
                bootstrap_on_truncated=self.bootstrap_on_truncated,
            )
            qh_det, _ = compute_dec_ocp_gae(
                Tah_hs=det_view["bTah_hs"],
                T_l=det_view["bT_l"],
                Tp1ah_Vh=vh_det_tp1,
                Tp1_Vl=vl_tp1,
                disc_gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                T_terminated=det_view["bT_terminated"],
                T_truncated=det_view["bT_truncated"],
                bootstrap_on_truncated=self.bootstrap_on_truncated,
            )
            ql_value_targets = ql

            cbf_scale = self._cbf_scale(timestep=timestep, timesteps=timesteps)
            adv_info = compute_cbf_advantages(
                bT_Ql=ql,
                bT_Vl=vl,
                bTah_Vh=vh,
                bTp1ah_Vh=vh_tp1,
                alpha=self.alpha,
                cbf_eps=self.cbf_eps,
                cbf_weight=self.cbf_weight,
                dt=self.env._step_dt,
                cbf_scale=cbf_scale,
                bT_done=view["bT_done"],
            )
        return UpdateTargets(
            qh_det=qh_det.detach(),
            ql=ql.detach(),
            ql_value_targets=ql_value_targets.detach(),
            bT_vl=vl.detach(),
            advantages=adv_info["bTa_A"].detach(),
            adv_info=adv_info,
            vl_error=(ql - vl).detach(),
        )

    def _build_final_rollout_graph(self, view: dict[str, torch.Tensor]) -> GraphData:
        return build_graph_data(
            agent_state=view["b_final_agent_state"],
            goal_state=view["b_final_goal_state"],
            obs_state=view["b_final_obs_state"],
            obs_radius=self.obs_radius,
        )

    def _advance_final_policy_state(
        self,
        graph: GraphData,
        view: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Advance the last stored policy carry on the final next-observation graph."""
        if self.policy.rnn is None:
            return None
        rnn_states = view.get("bTa_rnn_states")
        if rnn_states is None:
            raise ValueError("Recurrent final Vh bootstrap requires stored policy RNN states")
        start_state = self._flatten_policy_batch_rnn_state(rnn_states[:, -1])
        _action, _log_prob, _mean_action, final_state = self.policy.act(
            graph,
            rnn_state=start_state,
            deterministic=True,
        )
        return self._unflatten_policy_batch_rnn_state(final_state, batch_size=rnn_states.shape[0])

    def _track_update_targets(self, *, view: dict[str, torch.Tensor], targets: UpdateTargets) -> None:
        """Track rollout-level metrics before minibatch optimization starts."""
        self._track_scalars(
            {
                "DGPPO/safe_rate": targets.adv_info["bTa_is_safe"].float().mean(),
                "DGPPO/adv_raw_mean": targets.adv_info["bT_Al_raw"].mean(),
                "DGPPO/low_level_cost_mean": view["bT_l"].mean(),
                "DGPPO/ql_mean": targets.ql.mean(),
                "DGPPO/ql_abs_max": targets.ql.abs().max(),
                "DGPPO/vl_rollout_mean": targets.bT_vl.mean(),
                "DGPPO/vl_target_error_mean": targets.vl_error.mean(),
                "DGPPO/vl_target_error_abs_mean": targets.vl_error.abs().mean(),
                "DGPPO/rollout_terminated_rate": view["bT_terminated"].float().mean(),
                "DGPPO/rollout_truncated_rate": view["bT_truncated"].float().mean(),
                "DGPPO/bootstrap_on_truncated": float(self.bootstrap_on_truncated),
            }
        )
        self._track_safety_cost_metrics(view["bTah_hs"])

    def _run_update_epochs(
        self,
        *,
        memory: DGPPORolloutMemory,
        initial_targets: UpdateTargets,
        view: dict[str, torch.Tensor],
        det_view: dict[str, torch.Tensor],
        graph: GraphData,
        det_graph: GraphData,
        final_graph: GraphData,
        det_final_graph: GraphData,
        chunk_ids: torch.Tensor,
        timestep: int,
        timesteps: int,
    ) -> UpdateStats:
        """Run all PPO epochs and accumulate minibatch summaries."""
        loss_policy = initial_targets.advantages.new_zeros(())
        loss_value_l = initial_targets.advantages.new_zeros(())
        loss_value_h = initial_targets.advantages.new_zeros(())
        clip_frac = initial_targets.advantages.new_zeros(())
        n_minibatches = 0

        for epoch in range(self.learning_epochs):
            targets = initial_targets
            if epoch > 0:
                targets = self._compute_update_targets(
                    view=view,
                    det_view=det_view,
                    graph=graph,
                    det_graph=det_graph,
                    final_graph=final_graph,
                    det_final_graph=det_final_graph,
                    timestep=timestep,
                    timesteps=timesteps,
                )
            sampled_batches = memory.sample_minibatches(self.mini_batches)
            for idx in sampled_batches:
                info = self._update_minibatch(
                    idx=idx,
                    Qh_det=targets.qh_det,
                    Ql=targets.ql_value_targets,
                    bTa_A=targets.advantages,
                    view=view,
                    det_view=det_view,
                    graph=graph,
                    det_graph=det_graph,
                    chunk_ids=chunk_ids,
                )
                loss_policy += info["loss_p"]
                loss_value_l += info["loss_vl"]
                loss_value_h += info["loss_vh"]
                clip_frac += info["clip_frac"]
                n_minibatches += 1

        return UpdateStats(
            loss_policy=loss_policy,
            loss_value_l=loss_value_l,
            loss_value_h=loss_value_h,
            clip_frac=clip_frac,
            n_minibatches=n_minibatches,
        )

    def _summarize_update(self, stats: UpdateStats) -> dict[str, float]:
        inv_n = 1.0 / float(stats.n_minibatches)
        return {
            "loss_policy": float((stats.loss_policy * inv_n).item()),
            "loss_value_l": float((stats.loss_value_l * inv_n).item()),
            "loss_value_l_rmse": float(torch.sqrt((2.0 * stats.loss_value_l * inv_n).clamp_min(0.0)).item()),
            "loss_value_h": float((stats.loss_value_h * inv_n).item()),
            "clip_frac": float((stats.clip_frac * inv_n).item()),
            "lr_policy": float(self._policy_opt.param_groups[0]["lr"]),
            "lr_vl": float(self._vl_opt.param_groups[0]["lr"]),
            "lr_vh": float(self._vh_opt.param_groups[0]["lr"]),
        }

    def _track_update_summary(self, summary: Mapping[str, float]) -> None:
        self._track_scalars(
            {
                "DGPPO/loss_policy": summary["loss_policy"],
                "DGPPO/loss_value_l": summary["loss_value_l"],
                "DGPPO/loss_value_l_rmse": summary["loss_value_l_rmse"],
                "DGPPO/loss_value_h": summary["loss_value_h"],
                "DGPPO/clip_frac": summary["clip_frac"],
                "DGPPO/lr_policy": summary["lr_policy"],
                "DGPPO/lr_vl": summary["lr_vl"],
                "DGPPO/lr_vh": summary["lr_vh"],
            }
        )

    def _update_minibatch(
        self,
        *,
        idx: torch.Tensor,
        Qh_det: torch.Tensor,
        Ql: torch.Tensor,
        bTa_A: torch.Tensor,
        view: dict[str, torch.Tensor],
        det_view: dict[str, torch.Tensor],
        graph: GraphData,
        det_graph: GraphData,
        chunk_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single PPO minibatch step"""

        batch = build_update_graph_batch(
            idx=idx,
            view=view,
            det_view=det_view,
            qh_det=Qh_det,
            ql=Ql,
            advantages=bTa_A,
            obs_radius=self.obs_radius,
            graph=graph,
            det_graph=det_graph,
        )
        chunk_graph = None
        det_chunk_graph = None
        if self.policy.rnn is not None or self.Vl.rnn is not None or self.Vh.rnn is not None:
            chunk_graph = rollout_graph_chunks(batch.graph, chunk_ids=chunk_ids, T=batch.T, B=batch.b)
            det_chunk_graph = rollout_graph_chunks(batch.det_graph, chunk_ids=chunk_ids, T=batch.T, B=batch.b)

        # ---- Policy ----
        if self.policy.rnn is not None:
            policy_info = compute_rollout_policy_loss(
                policy=self.policy,
                graph=batch.graph,
                actions=batch.actions,
                old_logp=batch.old_logp,
                advantages=batch.advantages,
                chunk_ids=chunk_ids,
                clip_eps=self.clip_eps,
                entropy_scale=self.entropy_scale,
                n_agents=batch.A,
                chunk_graph=chunk_graph,
                done_mask=batch.done_mask,
            )
        else:
            policy_info = compute_policy_loss(
                policy=self.policy,
                graph=batch.graph,
                actions=batch.actions,
                old_logp=batch.old_logp,
                advantages=batch.advantages,
                clip_eps=self.clip_eps,
                entropy_scale=self.entropy_scale,
                n_agents=batch.A,
                rnn_state=None,
            )
        apply_policy_update(
            optimizer=self._policy_opt,
            loss=policy_info["loss_policy_total"],
            parameters=self.policy.parameters(),
            grad_clip=self.grad_clip,
        )

        # ---- Critics ----
        value_info = compute_value_losses(
            Vl=self.Vl,
            Vh=self.Vh,
            graph=batch.graph,
            det_graph=batch.det_graph,
            ql_targets=batch.ql_targets,
            qh_det_targets=batch.qh_det_targets,
            A=batch.A,
            vl_loss_scale=self.vl_loss_scale,
            vh_loss_scale=self.vh_loss_scale,
            det_rnn_states=batch.det_rnn_states,
            done_mask=batch.done_mask,
            chunk_ids=chunk_ids,
            chunk_graph=chunk_graph,
            det_chunk_graph=det_chunk_graph,
        )
        apply_value_update(
            optimizer=self._vl_opt,
            loss=value_info["loss_vl"],
            parameters=self._vl_grad_params,
            grad_clip=self.grad_clip,
        )
        apply_value_update(
            optimizer=self._vh_opt,
            loss=value_info["loss_vh"],
            parameters=self._vh_grad_params,
            grad_clip=self.grad_clip,
        )

        return {
            "loss_p": policy_info["loss_policy"].detach(),
            "loss_vl": value_info["loss_vl"].detach(),
            "loss_vh": value_info["loss_vh"].detach(),
            "clip_frac": policy_info["clip_frac"].detach(),
        }

    def _track_scalars(self, scalars: Mapping[str, float | torch.Tensor]) -> None:
        for name, value in scalars.items():
            if isinstance(value, torch.Tensor):
                value = float(value.item())
            self.track_data(name, float(value))

    def _load_hyperparameters_from_cfg(self) -> None:
        self.gamma: float = float(self.cfg.discount_factor)
        self.gae_lambda: float = float(self.cfg.gae_lambda)
        self.bootstrap_on_truncated: bool = bool(self.cfg.bootstrap_on_truncated)
        self.learning_starts: int = int(self.cfg.learning_starts)
        self.rollouts: int = int(self.cfg.rollouts)
        self.rnn_step: int = int(self.cfg.rnn_step)
        self.learning_epochs: int = int(self.cfg.learning_epochs)
        self.mini_batches: int = int(self.cfg.mini_batches)
        self.clip_eps: float = float(self.cfg.ratio_clip)
        self.alpha: float = float(self.cfg.alpha)
        self.cbf_eps: float = float(self.cfg.cbf_eps)
        self.cbf_weight: float = float(self.cfg.cbf_weight)
        self.cbf_schedule: bool = bool(self.cfg.cbf_schedule)
        self.grad_clip: float = float(self.cfg.grad_norm_clip)
        self.entropy_scale: float = float(self.cfg.entropy_loss_scale)
        self.vl_loss_scale: float = float(self.cfg.vl_loss_scale)
        self.vh_loss_scale: float = float(self.cfg.vh_loss_scale)
        self.obs_radius: float = float(self.cfg.obs_radius)
        self.lr_policy: float = float(self.cfg.lr_policy)
        self.lr_vl: float = float(self.cfg.lr_vl)
        self.lr_vh: float = float(self.cfg.lr_vh)
        self.rewards_shaper_scale: float = float(self.cfg.rewards_shaper_scale)

    def _cbf_scale(self, *, timestep: int, timesteps: int) -> float:
        """Piecewise-constant CBF weight schedule."""
        if not self.cbf_schedule:
            return self.cbf_weight
        progress = float(timestep) / max(float(timesteps), 1.0)
        scale = self.cbf_weight
        if progress >= 0.5:
            scale *= 2.0
        if progress >= 0.75:
            scale *= 2.0
        return scale

    def _rnn_chunk_ids(self, *, T: int, device: torch.device) -> torch.Tensor:
        """Reference-style truncated BPTT chunks over the rollout time axis."""
        if not self.policy.use_rnn:
            return torch.arange(T, device=device, dtype=torch.long).reshape(1, T)
        rnn_step = max(1, min(int(self.rnn_step), T))
        if T % rnn_step != 0:
            rnn_step = T
        return torch.arange(T, device=device, dtype=torch.long).reshape(-1, rnn_step)

    def _track_safety_cost_metrics(self, safety_costs: torch.Tensor) -> None:
        """Track signed safety costs without hiding opposing heads in a mean."""
        if safety_costs.numel() == 0 or safety_costs.shape[-1] == 0:
            return
        max_per_agent = safety_costs.max(dim=-1).values
        self.track_data("DGPPO/safety_cost_max", float(safety_costs.max().item()))
        self.track_data("DGPPO/safety_cost_violation_rate", float((max_per_agent >= 0.0).float().mean().item()))

        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        components = tuple(getattr(base_env, "cost_components", ()))
        for idx, label in enumerate(components[: safety_costs.shape[-1]]):
            head = safety_costs[..., idx]
            metric_label = str(label).replace(" ", "_").replace("/", "_")
            self.track_data(f"DGPPO/safety_cost/{metric_label}_max", float(head.max().item()))
            self.track_data(
                f"DGPPO/safety_cost/{metric_label}_violation_rate",
                float((head >= 0.0).float().mean().item()),
            )

    def _extract_graph_states(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode flat policy observations into graph node state tensors."""
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        cfg = getattr(base_env, "cfg", None)
        if (
            int(self.env.num_agents) > 1
            and getattr(cfg, "obstacle_observation_mode", None) == "ray_caster"
        ):
            raise RuntimeError(
                "DG-PPO ray-caster graph observations currently support one agent only. "
                "Multi-agent ray-caster graphs need per-agent hit groups/obstacle nodes."
            )
        return extract_graph_states_from_flat_obs(
            observations,
            base_env.graph_obs_layout,
            n_agents=self.env.num_agents,
        )

    def _build_graph(self, observations: torch.Tensor, states: torch.Tensor | None) -> GraphData:
        """Parse the flat policy-obs tensor into structured node states and build the graph."""
        agent_state, goal_state, obs_state = self._extract_graph_states(observations)
        return self._build_graph_from_states(agent_state=agent_state, goal_state=goal_state, obs_state=obs_state)

    def _build_graph_from_states(
        self,
        *,
        agent_state: torch.Tensor,
        goal_state: torch.Tensor,
        obs_state: torch.Tensor,
    ) -> GraphData:
        return build_graph_data(
            agent_state=agent_state,
            goal_state=goal_state,
            obs_state=obs_state,
            obs_radius=self.obs_radius,
        )

    def _select_policy_rnn_envs(
        self, rnn_state: torch.Tensor | None, env_ids: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Select env slots from a policy carry that stores one sequence per env-agent pair."""
        if rnn_state is None or env_ids is None:
            return None
        num_layers, _num_sequences, num_carries, hidden_size = rnn_state.shape
        n_agents = self.env.num_agents
        state = rnn_state.reshape(num_layers, self.env.num_envs, n_agents, num_carries, hidden_size)
        return state[:, env_ids].reshape(num_layers, int(env_ids.numel()) * n_agents, num_carries, hidden_size)

    def _flatten_policy_batch_rnn_state(self, rnn_state: torch.Tensor) -> torch.Tensor:
        batch_size, num_layers, n_agents, num_carries, hidden_size = rnn_state.shape
        return rnn_state.permute(1, 0, 2, 3, 4).reshape(
            num_layers,
            batch_size * n_agents,
            num_carries,
            hidden_size,
        )

    def _unflatten_policy_batch_rnn_state(self, rnn_state: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        num_layers, total_agents, num_carries, hidden_size = rnn_state.shape
        n_agents = int(self.env.num_agents)
        if total_agents != batch_size * n_agents:
            raise ValueError(f"policy RNN state has {total_agents} agents, expected {batch_size * n_agents}")
        return rnn_state.reshape(num_layers, batch_size, n_agents, num_carries, hidden_size).permute(1, 0, 2, 3, 4)

    def _select_env_rnn_envs(self, rnn_state: torch.Tensor | None, env_ids: torch.Tensor | None) -> torch.Tensor | None:
        if rnn_state is None or env_ids is None:
            return None
        return rnn_state[:, env_ids]

    def _env_done_mask(self, mask: torch.Tensor, n_envs: int) -> torch.Tensor:
        """Return a flat boolean mask with one entry per IsaacLab env."""
        return torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(n_envs)

    def _rewards_from_observation_or_env(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        n_envs: int,
    ) -> torch.Tensor:
        """Return a DG-PPO reward aligned with the stored rollout graph when available."""
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        reward_fn = getattr(base_env, "compute_dgppo_reward_from_observation_action", None)
        if callable(reward_fn):
            rewards = reward_fn(observations=observations, actions=actions)
        return torch.as_tensor(rewards, device=self.device, dtype=torch.float32).reshape(n_envs)

    def _reset_rnn_states_for_done(self, done: torch.Tensor) -> None:
        """Reset recurrent state on ``terminated | truncated`` like skrl PPO_RNN."""
        self._policy_rnn_state = zero_policy_rnn_states_for_done(
            self._policy_rnn_state,
            done,
            n_agents=self.env.num_agents,
        )
        self._vl_rnn_state = zero_env_rnn_states_for_done(self._vl_rnn_state, done)

    def _costs_from_observation(
        self,
        *,
        observations: torch.Tensor,
        n_envs: int,
        n_agents: int,
        n_constraints: int,
    ) -> torch.Tensor:
        """Return DG-PPO safety costs from the same observation graph stored in memory."""
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        cost_fn = getattr(base_env, "compute_dgppo_costs_from_observation", None)
        if not callable(cost_fn):
            raise RuntimeError(
                "DG-PPO requires the environment to provide compute_dgppo_costs_from_observation(observations)."
            )
        costs_all = cost_fn(observations=observations)
        costs_all = torch.as_tensor(costs_all, device=self.device, dtype=torch.float32)
        costs_all = costs_all.reshape(n_envs, n_agents, -1)
        return align_safety_cost_heads(costs_all, n_constraints)
