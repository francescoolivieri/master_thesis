import dataclasses
import time
from collections.abc import Mapping
from pathlib import Path
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
    ql_value_targets: torch.Tensor
    advantages: torch.Tensor
    adv_info: dict[str, torch.Tensor]


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
        self._debug_rollout_plot_count = 0
        self._debug_rollout_plot_bucket = -1
        self._debug_warnings: set[str] = set()

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
        record_rewards = torch.as_tensor(record_rewards, device=self.device, dtype=torch.float32).reshape_as(rewards)
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

        view = memory.as_update_view("stc")
        det_view = memory.as_update_view("det")

        graph = build_rollout_graph(view=view, obs_radius=self.obs_radius)
        det_graph = build_rollout_graph(view=det_view, obs_radius=self.obs_radius)
        final_graph = self._build_final_rollout_graph(view)
        det_final_graph = self._build_final_rollout_graph(det_view)
        chunk_ids = self._rnn_chunk_ids(T=view["bTa_actions"].shape[1], device=torch.device("cpu"))

        update_info: dict[str, torch.Tensor] | None = None
        for _epoch in range(self.learning_epochs):
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
            for env_ids in memory.sample_minibatches(self.mini_batches):
                update_info = self._update_minibatch(
                    env_ids=env_ids,
                    targets=targets,
                    view=view,
                    det_view=det_view,
                    graph=graph,
                    det_graph=det_graph,
                    chunk_ids=chunk_ids,
                )
            if update_info is not None:
                update_info["eval/safe_data"] = targets.adv_info["bTa_is_safe"].float().mean().detach()

        if update_info is None:
            return

        self._track_scalars(update_info)
        self._maybe_log_debug_rollout(view=view, timestep=timestep)

        memory.reset()

    def write_checkpoint(self, *, timestep: int, timesteps: int) -> None:
        super().write_checkpoint(timestep=timestep, timesteps=timesteps)
        self._maybe_save_critic_debug_snapshot(timestep=timestep)

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
            batch_size, rollout_length = view["bT_l"].shape
            det_batch_size, det_rollout_length = det_view["bT_l"].shape

            bT_vl, _, final_vl_state = scan_rollout_vl_values(
                Vl=self.Vl,
                graph=graph,
                B=batch_size,
                T=rollout_length,
                initial_rnn_state=view.get("b_initial_vl_rnn_state"),
                done_mask=view.get("bT_done"),
            )
            final_vl = evaluate_vl_batch_from_states(
                Vl=self.Vl,
                graph=final_graph,
                rnn_states=final_vl_state,
            )
            bTah_vh = evaluate_rollout_vh_values(
                Vh=self.Vh,
                graph=graph,
                B=batch_size,
                T=rollout_length,
                rnn_states=view.get("bTa_rnn_states"),
            )
            final_policy_state = self._advance_final_policy_state(final_graph, view)
            final_vh = evaluate_vh_batch_from_states(
                Vh=self.Vh,
                graph=final_graph,
                rnn_states=final_policy_state,
            )
            bTah_vh_det = evaluate_rollout_vh_values(
                Vh=self.Vh,
                graph=det_graph,
                B=det_batch_size,
                T=det_rollout_length,
                rnn_states=det_view.get("bTa_rnn_states"),
            )
            det_final_policy_state = self._advance_final_policy_state(det_final_graph, det_view)
            final_vh_det = evaluate_vh_batch_from_states(
                Vh=self.Vh,
                graph=det_final_graph,
                rnn_states=det_final_policy_state,
            )

            bTp1_vl = torch.cat([bT_vl, final_vl[:, None]], dim=1)
            bTp1ah_vh = torch.cat([bTah_vh, final_vh[:, None]], dim=1)
            bTp1ah_vh_det = torch.cat([bTah_vh_det, final_vh_det[:, None]], dim=1)

            _bTah_qh, bT_ql = compute_dec_ocp_gae(
                Tah_hs=view["bTah_hs"],
                T_l=view["bT_l"],
                Tp1ah_Vh=bTp1ah_vh,
                Tp1_Vl=bTp1_vl,
                disc_gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                T_terminated=view["bT_terminated"],
                T_truncated=view["bT_truncated"],
                bootstrap_on_truncated=self.bootstrap_on_truncated,
            )
            bTah_qh_det, _ = compute_dec_ocp_gae(
                Tah_hs=det_view["bTah_hs"],
                T_l=det_view["bT_l"],
                Tp1ah_Vh=bTp1ah_vh_det,
                Tp1_Vl=bTp1_vl,
                disc_gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                T_terminated=det_view["bT_terminated"],
                T_truncated=det_view["bT_truncated"],
                bootstrap_on_truncated=self.bootstrap_on_truncated,
            )
            adv_info = compute_cbf_advantages(
                bT_Ql=bT_ql,
                bT_Vl=bT_vl,
                bTah_Vh=bTah_vh,
                bTp1ah_Vh=bTp1ah_vh,
                alpha=self.alpha,
                cbf_eps=self.cbf_eps,
                cbf_weight=self.cbf_weight,
                dt=self.env._step_dt,
                cbf_scale=self._cbf_scale(timestep=timestep, timesteps=timesteps),
                bT_done=view["bT_done"],
            )

        return UpdateTargets(
            qh_det=bTah_qh_det.detach(),
            ql_value_targets=bT_ql.detach(),
            advantages=adv_info["bTa_A"].detach(),
            adv_info=adv_info,
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

    def _update_minibatch(
        self,
        *,
        env_ids: torch.Tensor,
        targets: UpdateTargets,
        view: dict[str, torch.Tensor],
        det_view: dict[str, torch.Tensor],
        graph: GraphData,
        det_graph: GraphData,
        chunk_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Single PPO minibatch step"""

        batch = build_update_graph_batch(
            idx=env_ids,
            view=view,
            det_view=det_view,
            qh_det=targets.qh_det,
            ql=targets.ql_value_targets,
            advantages=targets.advantages,
            obs_radius=self.obs_radius,
            graph=graph,
            det_graph=det_graph,
        )
        chunk_graph = None
        det_chunk_graph = None
        if self.policy.rnn is not None or self.Vl.rnn is not None or self.Vh.rnn is not None:
            chunk_graph = rollout_graph_chunks(
                batch.graph,
                chunk_ids=chunk_ids,
                T=batch.rollout_length,
                B=batch.batch_size,
            )
            det_chunk_graph = rollout_graph_chunks(
                batch.det_graph,
                chunk_ids=chunk_ids,
                T=batch.rollout_length,
                B=batch.batch_size,
            )

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
                n_agents=batch.n_agents,
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
                n_agents=batch.n_agents,
                rnn_state=None,
            )
        with torch.no_grad():
            total_variation_dist = 0.5 * torch.abs(policy_info["ratio"] - 1.0).mean()
        policy_grad_norm = apply_policy_update(
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
            n_agents=batch.n_agents,
            vl_loss_scale=self.vl_loss_scale,
            vh_loss_scale=self.vh_loss_scale,
            det_rnn_states=batch.det_rnn_states,
            done_mask=batch.done_mask,
            chunk_ids=chunk_ids,
            chunk_graph=chunk_graph,
            det_chunk_graph=det_chunk_graph,
        )
        vl_grad_norm = apply_value_update(
            optimizer=self._vl_opt,
            loss=value_info["loss_vl"],
            parameters=self._vl_grad_params,
            grad_clip=self.grad_clip,
        )
        vh_grad_norm = apply_value_update(
            optimizer=self._vh_opt,
            loss=value_info["loss_vh"],
            parameters=self._vh_grad_params,
            grad_clip=self.grad_clip,
        )

        return {
            "Vl/loss": value_info["loss_vl"].detach(),
            "Vl/grad_norm": vl_grad_norm.detach(),
            "Vl/has_nan": (~torch.isfinite(vl_grad_norm)).to(dtype=value_info["loss_vl"].dtype).detach(),
            "Vl/max_target": batch.ql_targets.max().detach(),
            "Vl/min_target": batch.ql_targets.min().detach(),
            "Vh/loss_Vh": value_info["loss_vh"].detach(),
            "Vh/grad_Vh_norm": vh_grad_norm.detach(),
            "Vh/grad_Vh_has_nan": (~torch.isfinite(vh_grad_norm)).to(dtype=value_info["loss_vh"].dtype).detach(),
            "policy/loss": policy_info["loss_policy_total"].detach(),
            "policy/clip_frac": policy_info["clip_frac"].detach(),
            "policy/entropy": policy_info["entropy_mean"].detach(),
            "policy/total_variation_dist": total_variation_dist.detach(),
            "policy/grad_norm": policy_grad_norm.detach(),
            "policy/has_nan": (~torch.isfinite(policy_grad_norm)).to(dtype=policy_info["loss_policy"].dtype).detach(),
            "policy/log_pi_min": batch.old_logp.min().detach(),
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
        self.debug_rollout_plot_interval: int = int(self.cfg.get("debug_rollout_plot_interval", 0) or 0)

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

    def _rollout_safety_masks(self, view: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return boolean safety masks in ``[B, T, A]`` layout."""
        agent_state = view["bTa_agent_state"]
        pos = agent_state[..., :3]
        masks: dict[str, torch.Tensor] = {}

        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        safe_min = getattr(base_env, "_arena_min_safe", None)
        safe_max = getattr(base_env, "_arena_max_safe", None)
        if safe_min is not None and safe_max is not None:
            safe_min_t = torch.as_tensor(safe_min, device=pos.device, dtype=pos.dtype)
            safe_max_t = torch.as_tensor(safe_max, device=pos.device, dtype=pos.dtype)
            masks["vertical_bounds"] = (pos[..., 2] < safe_min_t[2]) | (pos[..., 2] > safe_max_t[2])
            masks["xy_boundary"] = torch.any(
                (pos[..., :2] < safe_min_t[:2]) | (pos[..., :2] > safe_max_t[:2]),
                dim=-1,
            )

        if view["bTah_hs"].shape[-1] > 1:
            masks["ray_obstacle"] = view["bTah_hs"][..., 1] >= 0.0

        pillar_mask = self._pillar_collision_mask_from_positions(pos)
        if pillar_mask is not None:
            masks["pillar_collision"] = pillar_mask
        return masks

    def _pillar_collision_mask_from_positions(self, pos: torch.Tensor) -> torch.Tensor | None:
        """Approximate env pillar collision from rollout positions in ``[B, T, A, 3]`` layout."""
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        pillar_xy = getattr(base_env, "_pillar_positions_xy", None)
        if pillar_xy is None:
            return None
        pillar_xy = torch.as_tensor(pillar_xy, device=pos.device, dtype=pos.dtype)
        if pillar_xy.numel() == 0:
            return torch.zeros(pos.shape[:-1], dtype=torch.bool, device=pos.device)

        radius = float(getattr(base_env, "_pillar_collision_radius", 0.0))
        top_z = float(getattr(base_env, "_pillar_top_z", float("inf")))
        cfg = getattr(base_env, "cfg", None)
        arena_min = getattr(cfg, "arena_min", (-float("inf"), -float("inf"), -float("inf")))
        min_z = float(arena_min[2])

        dxy = torch.linalg.vector_norm(pos[..., None, :2] - pillar_xy.view(1, 1, 1, -1, 2), dim=-1)
        inside_radius = torch.any(dxy <= radius, dim=-1)
        inside_height = (pos[..., 2] >= min_z) & (pos[..., 2] <= top_z)
        return inside_radius & inside_height

    def _maybe_log_debug_rollout(self, *, view: dict[str, torch.Tensor], timestep: int) -> None:
        """Save a compact visual summary of one stochastic rollout."""
        interval = int(getattr(self, "debug_rollout_plot_interval", 0))
        if interval <= 0:
            return
        step = int(timestep) + 1
        bucket = step // interval
        if self._debug_rollout_plot_count > 0 and bucket <= self._debug_rollout_plot_bucket:
            return
        try:
            path = self._save_rollout_summary_plot(view=view, step=step)
            self._debug_rollout_plot_bucket = bucket
            self._debug_rollout_plot_count += 1
            self._log_wandb_image("DGPPO/debug/rollout_summary", path=path, step=step)
        except Exception as exc:
            self._warn_debug_once("rollout_plot", f"Failed to save DG-PPO rollout plot: {exc}")

    def _save_rollout_summary_plot(self, *, view: dict[str, torch.Tensor], step: int) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pos = view["bTa_agent_state"][0, :, 0, :3].detach().to("cpu")
        goal = view["bTa_goal_state"][0, :, 0, :3].detach().to("cpu")
        safety_masks = self._rollout_safety_masks({key: value[:1] for key, value in view.items()})
        masks = {name: mask[0, :, 0].detach().to("cpu").bool() for name, mask in safety_masks.items()}
        unsafe = torch.stack(tuple(masks.values()), dim=0).any(dim=0) if masks else None
        t = torch.arange(pos.shape[0])
        goal_distance = torch.linalg.vector_norm(goal[:, :3] - pos[:, :3], dim=-1)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            dpi=140,
            gridspec_kw={"height_ratios": [1.35, 1.0]},
        )
        ax_xy, ax_z, ax_dist, ax_safety = axes.flatten()
        fig.suptitle(f"DG-PPO stochastic rollout summary at step {step}", fontsize=13)

        self._draw_debug_xy_context(ax_xy)
        ax_xy.plot(pos[:, 0], pos[:, 1], color="black", linewidth=1.8, label="trajectory")
        ax_xy.scatter(pos[:1, 0], pos[:1, 1], marker="o", facecolor="white", edgecolor="black", s=55, label="start")
        ax_xy.scatter(pos[-1:, 0], pos[-1:, 1], marker="s", color="black", s=42, label="last")
        ax_xy.scatter(goal[-1:, 0], goal[-1:, 1], marker="*", color="#cc7a00", s=115, label="goal")
        if unsafe is not None and bool(unsafe.any().item()):
            bad = pos[unsafe]
            ax_xy.scatter(bad[:, 0], bad[:, 1], marker="x", color="black", s=42, linewidths=1.3, label="violation")
        self._set_debug_xy_limits(ax_xy, pos=pos, goal=goal)
        ax_xy.set_title("XY top-down")
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(True, linewidth=0.4, alpha=0.35)
        ax_xy.legend(loc="best", fontsize=8)

        ax_z.plot(t, pos[:, 2], color="black", linewidth=1.8, label="z trajectory")
        bounds = self._debug_arena_bounds()
        if bounds is not None:
            mn, mx = bounds
            ax_z.axhspan(mn[2], mx[2], color="0.90", alpha=0.6, label="safe altitude band")
            ax_z.axhline(mn[2], color="black", linestyle="--", linewidth=1.0, label="min safe z")
            ax_z.axhline(mx[2], color="black", linestyle=":", linewidth=1.4, label="max safe z")
        vertical = masks.get("vertical_bounds")
        if vertical is not None and bool(vertical.any().item()):
            ax_z.scatter(t[vertical], pos[vertical, 2], marker="x", color="black", s=35, label="vertical violation")
        ax_z.set_title("Altitude over rollout")
        ax_z.set_xlabel("rollout step")
        ax_z.set_ylabel("z")
        ax_z.grid(True, linewidth=0.4, alpha=0.35)
        ax_z.legend(loc="best", fontsize=8)

        ax_dist.plot(t, goal_distance, color="black", linewidth=1.8)
        ax_dist.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        ax_dist.set_title("Distance to goal")
        ax_dist.set_xlabel("rollout step")
        ax_dist.set_ylabel("meters")
        ax_dist.grid(True, linewidth=0.4, alpha=0.35)

        names = list(masks)
        marker_cycle = ["x", "o", "s", "^"]
        for row, name in enumerate(names):
            mask = masks[name]
            ax_safety.hlines(row, 0, max(int(t[-1].item()), 1), color="0.82", linewidth=1.0)
            if bool(mask.any().item()):
                ax_safety.scatter(
                    t[mask],
                    torch.full_like(t[mask], row),
                    marker=marker_cycle[row % len(marker_cycle)],
                    color="black",
                    s=30,
                    linewidths=1.0,
                )
        ax_safety.set_title("Safety violations by constraint")
        ax_safety.set_xlabel("rollout step")
        ax_safety.set_yticks(range(len(names)))
        ax_safety.set_yticklabels(names)
        ax_safety.set_ylim(-0.6, max(len(names) - 0.4, 0.6))
        ax_safety.grid(True, axis="x", linewidth=0.4, alpha=0.35)

        fig.tight_layout()

        out_dir = Path(self.experiment_dir) / "debug" / "rollouts"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"rollout_summary_step_{step}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def _draw_debug_xy_context(self, ax: Any) -> None:
        from matplotlib.patches import Circle, Rectangle

        bounds = self._debug_arena_bounds()
        if bounds is not None:
            mn, mx = bounds
            ax.add_patch(
                Rectangle(
                    (mn[0], mn[1]),
                    mx[0] - mn[0],
                    mx[1] - mn[1],
                    fill=False,
                    edgecolor="black",
                    linewidth=1.2,
                    linestyle="--",
                    label="safe XY bounds",
                )
            )
        pillars = self._debug_pillars_cpu()
        if pillars is None:
            return
        pillar_xy, radius = pillars
        for center in pillar_xy:
            ax.add_patch(
                Circle(
                    (float(center[0]), float(center[1])),
                    radius,
                    facecolor="0.70",
                    edgecolor="black",
                    alpha=0.45,
                    linewidth=0.8,
                )
            )

    def _set_debug_xy_limits(self, ax: Any, *, pos: torch.Tensor, goal: torch.Tensor) -> None:
        bounds = self._debug_arena_bounds()
        if bounds is not None:
            mn, mx = bounds
            ax.set_xlim(mn[0], mx[0])
            ax.set_ylim(mn[1], mx[1])
            return
        xy = torch.cat([pos[:, :2], goal[:, :2]], dim=0)
        lo = xy.min(dim=0).values - 0.5
        hi = xy.max(dim=0).values + 0.5
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))

    def _debug_pillars_cpu(self) -> tuple[torch.Tensor, float] | None:
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        pillar_xy = getattr(base_env, "_pillar_positions_xy", None)
        if pillar_xy is None:
            return None
        if isinstance(pillar_xy, torch.Tensor):
            pillar_xy = pillar_xy.detach().to(device="cpu", dtype=torch.float32)
        else:
            pillar_xy = torch.as_tensor(pillar_xy, device="cpu", dtype=torch.float32)
        if pillar_xy.numel() == 0:
            return None
        radius = float(getattr(base_env, "_pillar_collision_radius", 0.0))
        return pillar_xy, radius

    def _debug_arena_bounds(self) -> tuple[list[float], list[float]] | None:
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        safe_min = getattr(base_env, "_arena_min_safe", None)
        safe_max = getattr(base_env, "_arena_max_safe", None)
        if safe_min is None or safe_max is None:
            return None
        mn = self._tensor_to_cpu_float(safe_min).tolist()
        mx = self._tensor_to_cpu_float(safe_max).tolist()
        return mn, mx

    def _maybe_save_critic_debug_snapshot(self, *, timestep: int) -> None:
        try:
            out_dir = Path(self.experiment_dir) / "debug" / "critics"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"critics_step_{timestep}.pt"
            torch.save(self._critic_debug_payload(timestep=timestep), path)
            self._log_wandb_artifact(path=path, name="dgppo-critic-debug-snapshots", kind="critic_snapshot")
        except Exception as exc:
            self._warn_debug_once("critic_snapshot", f"Failed to save DG-PPO critic debug snapshot: {exc}")

    def _critic_debug_payload(self, *, timestep: int) -> dict[str, Any]:
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        return {
            "timestep": int(timestep),
            "Vl": self._state_dict_to_cpu(self.Vl.state_dict()),
            "Vh": self._state_dict_to_cpu(self.Vh.state_dict()),
            "metadata": {
                "obs_radius": float(self.obs_radius),
                "num_agents": int(self.env.num_agents),
                "n_constraints": int(getattr(base_env, "n_constraints", 1)),
                "graph_obs_layout": dict(getattr(base_env, "graph_obs_layout", {})),
                "arena_min_safe": self._tensor_to_list(getattr(base_env, "_arena_min_safe", None)),
                "arena_max_safe": self._tensor_to_list(getattr(base_env, "_arena_max_safe", None)),
                "pillar_positions_xy": self._tensor_to_list(getattr(base_env, "_pillar_positions_xy", None)),
                "pillar_collision_radius": float(getattr(base_env, "_pillar_collision_radius", 0.0)),
                "pillar_top_z": float(getattr(base_env, "_pillar_top_z", 0.0)),
                "rnn": dict(self.cfg.rnn),
                "gnn": dict(self.cfg.gnn),
                "model": dict(self.cfg.model),
            },
        }

    @staticmethod
    def _state_dict_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.detach().to("cpu") for key, value in state_dict.items()}

    @staticmethod
    def _tensor_to_list(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu").tolist()
        return value

    @staticmethod
    def _tensor_to_cpu_float(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.float32)
        return torch.as_tensor(value, device="cpu", dtype=torch.float32)

    def _log_wandb_image(self, key: str, *, path: Path, step: int) -> None:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({key: wandb.Image(str(path), caption=path.stem)}, step=step)
        except Exception:
            pass

    def _log_wandb_artifact(self, *, path: Path, name: str, kind: str) -> None:
        try:
            import wandb

            if wandb.run is not None:
                artifact = wandb.Artifact(name=name, type=kind)
                artifact.add_file(str(path))
                wandb.log_artifact(artifact)
        except Exception:
            pass

    def _warn_debug_once(self, key: str, message: str) -> None:
        if key in self._debug_warnings:
            return
        self._debug_warnings.add(key)
        print(f"[WARN] {message}")

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
            reward_aux_fn = getattr(base_env, "get_dgppo_reward_auxiliary_data", None)
            if callable(reward_aux_fn):
                rewards = reward_fn(observations=observations, actions=actions, reward_aux=reward_aux_fn())
            else:
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
