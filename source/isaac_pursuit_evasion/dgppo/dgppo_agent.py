import dataclasses
import time
from collections.abc import Mapping
from typing import Any

import torch
from skrl.agents.torch import Agent, AgentCfg
from skrl.agents.torch.base import ExperimentCfg

from .dgppo_debug import DGPPODebugConfig, DGPPOTrainingDiagnostics
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
    rollout_graph_chunks,
)
from .utils import (
    GraphData,
    align_safety_cost_heads,
    build_graph_data,
    compute_pos_tracking_safety_costs,
    compute_cbf_advantages,
    compute_dec_ocp_gae,
    extract_graph_states_from_flat_obs,
    zero_env_rnn_states_for_done,
    zero_policy_rnn_states_for_done,
)


@dataclasses.dataclass(kw_only=True)
class DGPPOAgentCfg(AgentCfg):
    """skrl-compatible config wrapper for the custom DG-PPO agent."""

    alpha: float = 10.0
    cbf_eps: float = 1e-2
    cbf_weight: float = 1.0
    cbf_schedule: bool = True

    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    bootstrap_on_truncated: bool = False
    learning_starts: int = 0
    rollouts: int = 32
    rnn_step: int = 16
    learning_epochs: int = 8
    mini_batches: int = 8
    ratio_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    vl_loss_scale: float = 1.0
    vh_loss_scale: float = 1.0
    grad_norm_clip: float = 2.0

    lr_policy: float = 3e-4
    lr_vl: float = 1e-3
    lr_vh: float = 1e-3

    obs_radius: float = 2.0
    use_rnn: bool = True
    rnn: dict[str, Any] = dataclasses.field(default_factory=lambda: {"cell": "gru", "hidden": 64, "layers": 1})
    gnn: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "policy_layers": 1,
            "vl_layers": 1,
            "vh_layers": 1,
            "policy_out_dim": 64,
            "critic_out_dim": 64,
            "msg_dim": 32,
            "n_heads": 3,
        }
    )
    model: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "policy_mlp_hid": [128, 64],
            "critic_mlp_hid": [128, 64],
            "scale_hid": 64,
            "scale_final": 0.01,
            "std_dev_init": 0.5,
            "std_dev_min": 1e-5,
        }
    )
    debug: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "enabled": True,
            "step_interval": 10,
            "update_interval": 1,
            "minibatch_interval": 1,
            "sample_env_count": 8,
            "log_minibatches": True,
            "log_jsonl": True,
            "log_tensorboard_scalars": True,
            "log_on_done": True,
            "rnn_done_norm_epsilon": 1e-6,
            "anomaly_abs_threshold": 1e6,
        }
    )
    seed: int | None = None
    num_envs: int | None = None
    _raw: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DGPPOAgentCfg":
        raw = dict(data)
        default_rnn = {"cell": "gru", "hidden": 64, "layers": 1}
        default_gnn = {
            "policy_layers": 1,
            "critic_layers": 1,
            "policy_out_dim": 64,
            "critic_out_dim": 64,
            "msg_dim": 32,
            "n_heads": 3,
        }
        default_model = {
            "policy_mlp_hid": [128, 64],
            "critic_mlp_hid": [128, 64],
            "scale_hid": 64,
            "scale_final": 0.01,
            "std_dev_init": 0.5,
            "std_dev_min": 1e-5,
        }
        default_debug = {
            "enabled": True,
            "step_interval": 10,
            "update_interval": 1,
            "minibatch_interval": 1,
            "sample_env_count": 8,
            "log_minibatches": True,
            "log_jsonl": True,
            "log_tensorboard_scalars": True,
            "log_on_done": True,
            "rnn_done_norm_epsilon": 1e-6,
            "anomaly_abs_threshold": 1e6,
        }
        experiment_data = raw.get("experiment", {})
        experiment = (
            experiment_data if isinstance(experiment_data, ExperimentCfg) else ExperimentCfg(**dict(experiment_data))
        )

        return cls(
            experiment=experiment,
            alpha=float(raw.get("alpha", 10.0)),
            cbf_eps=float(raw.get("cbf_eps", 1e-2)),
            cbf_weight=float(raw.get("cbf_weight", 1.0)),
            cbf_schedule=bool(raw.get("cbf_schedule", True)),
            discount_factor=float(raw.get("discount_factor", 0.99)),
            gae_lambda=float(raw.get("gae_lambda", raw.get("lambda", 0.95))),
            bootstrap_on_truncated=bool(raw.get("bootstrap_on_truncated", False)),
            learning_starts=int(raw.get("learning_starts", 0)),
            rollouts=int(raw.get("rollouts", 32)),
            rnn_step=int(raw.get("rnn_step", 16)),
            learning_epochs=int(raw.get("learning_epochs", 8)),
            mini_batches=int(raw.get("mini_batches", 8)),
            ratio_clip=float(raw.get("ratio_clip", 0.2)),
            entropy_loss_scale=float(raw.get("entropy_loss_scale", 0.0)),
            vl_loss_scale=float(raw.get("vl_loss_scale", 1.0)),
            vh_loss_scale=float(raw.get("vh_loss_scale", 1.0)),
            grad_norm_clip=float(raw.get("grad_norm_clip", 2.0)),
            lr_policy=float(raw.get("lr_policy", 3e-4)),
            lr_vl=float(raw.get("lr_vl", 1e-3)),
            lr_vh=float(raw.get("lr_vh", 1e-3)),
            obs_radius=float(raw.get("obs_radius", 2.0)),
            use_rnn=bool(raw.get("use_rnn", True)),
            rnn={**default_rnn, **dict(raw.get("rnn", {}))},
            gnn={**default_gnn, **dict(raw.get("gnn", {}))},
            model={**default_model, **dict(raw.get("model", {}))},
            debug={**default_debug, **dict(raw.get("debug", {}))},
            seed=None if raw.get("seed") is None else int(raw["seed"]),
            num_envs=None if raw.get("num_envs") is None else int(raw["num_envs"]),
            _raw=raw,
        )

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._raw:
            return self._raw[key]
        return getattr(self, key, default)


class DGPPOAgent(Agent):
    def __init__(
        self,
        policy: DGPPOPolicy,
        Vl: DGPPOValueNet,
        Vh: DGPPOValueNet,
        env: Any,  # remove
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

        # self.training = False

        self.policy = policy.to(device)
        self.Vl = Vl.to(device)
        self.Vh = Vh.to(device)
        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

        # Load hyperparameters
        self.load_dgppo_hyperparameters()

        # Setup optimizers
        self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        vl_params = (
            list(self.Vl.gnn.parameters()) + list(self.Vl.head.parameters()) + list(self.Vl.net.value_out.parameters())
        )
        if self.Vl.rnn is not None:
            vl_params += list(self.Vl.rnn.parameters())
        vh_params = (
            list(self.Vh.gnn.parameters()) + list(self.Vh.head.parameters()) + list(self.Vh.net.value_out.parameters())
        )
        if self.Vh.rnn is not None:
            vh_params += list(self.Vh.rnn.parameters())
        self._vl_opt = torch.optim.Adam(vl_params, lr=self.lr_vl)
        self._vh_opt = torch.optim.Adam(vh_params, lr=self.lr_vh)
        self._vl_grad_params = vl_params
        self._vh_grad_params = vh_params
        self.checkpoint_modules = {
            "policy": self.policy,
            "Vl": self.Vl,
            "Vh": self.Vh,
            "policy_optimizer": self._policy_opt,
            "Vl_optimizer": self._vl_opt,
            "Vh_optimizer": self._vh_opt,
        }

        # Setup memory (Note: personalised memory for cleaner code)
        self.memory: DGPPORolloutMemory | None = None
        self.num_envs = self.cfg.get("num_envs", 1)
        self._rollout = 0
        self._policy_rnn_state = None
        self._vl_rnn_state = None
        self._stoch_env_ids = None
        self._det_env_ids = None
        self._last_graph = None
        self._last_log_prob = None
        self._last_policy_rnn_state = None
        self._last_vl_rnn_state = None
        self._current_Vl = None
        self._current_Vh = None
        self._current_next_observations = None
        self._update_id = 0
        self._debug = DGPPOTrainingDiagnostics(
            cfg=DGPPODebugConfig.from_mapping(self.cfg.debug),
            experiment_dir=self.experiment_dir,
            track_data=self.track_data,
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
        rollout_length = int(self.cfg.get("rollouts", 32))  # from AgentCfg
        n_agents = self.env.num_agents
        layout = self.env.unwrapped.graph_obs_layout
        n_obs = int(layout.get("n_obstacles", 0))
        state_dim = int(layout["state_dim"])
        action_dim = self.env.action_space.shape[0]
        n_constraints = int(getattr(self.env, "n_constraints", getattr(self.env.unwrapped, "n_constraints", 1)))
        use_rnn = self.policy.use_rnn

        # Allocate the DGPPO rollout memory now that we know the env shape
        if self.env.num_envs < 2 or (self.env.num_envs % 2) != 0:
            raise ValueError(f"DGPPO split rollout requires an even num_envs >= 2, got {self.env.num_envs}")
        split = abs(self.env.num_envs // 2)
        self._det_env_ids = torch.arange(0, split, device=self.device, dtype=torch.long)
        self._stoch_env_ids = torch.arange(split, self.env.num_envs, device=self.device, dtype=torch.long)

        # Check these
        self.memory = DGPPORolloutMemory(
            rollout_length=rollout_length,
            num_det_envs=int(self._det_env_ids.numel()),
            num_stc_envs=int(self._stoch_env_ids.numel()),
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

        # Initial RNN carry for the rollout (B * A agents, regardless of env).
        if self.policy.use_rnn:
            self._policy_rnn_state = self.policy.initialize_carry(
                n_agents_total=self.env.num_envs * n_agents, device=self.device
            )
        if self.Vl.rnn is not None:
            self._vl_rnn_state = self.Vl.rnn.initialize_carry(self.env.num_envs, device=self.device)
        self._debug.log_setup(agent=self, trainer_cfg=trainer_cfg)

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample actions (and compute values if training) for the current environment step.

        Note: the graph is built by querying the env for structured states (agent / goal /
        obstacle positions), not from the flat `observations` blob passed by skrl.
        This mirrors the reference DGPPO design: `build_graph_data` needs three separate
        (E, A/O, S) tensors; the flat observation vector does not provide that structure.

        :param observations: Per-agent observations (E*A, obs_dim) — unused directly.
        :param states:       Global env state (E, state_dim)       — unused directly.
        :param timestep:     Current timestep.
        :param timesteps:    Total timesteps.
        :return: (actions (E*A, action_dim), extras dict with log_prob and mean_action).
        """
        n_agents = self.env.num_agents

        with torch.no_grad():  # Saves memory
            graph = self._build_graph(observations, states)
            self._last_graph = graph
            self._last_policy_rnn_state = (
                None if self._policy_rnn_state is None else self._policy_rnn_state.detach().clone()
            )

            action, log_prob, mean_action, new_rnn = self.policy.act(
                graph,
                rnn_state=self._policy_rnn_state,
                n_agents_total=n_agents,
                deterministic=not self.training,
            )

            if self.training:
                self._last_vl_rnn_state = None if self._vl_rnn_state is None else self._vl_rnn_state.detach().clone()
                vl, self._vl_rnn_state = self.Vl(graph, self._vl_rnn_state, n_agents)
                vh, _ = self.Vh(graph, self._last_policy_rnn_state, n_agents)
                self._current_Vl = vl
                self._current_Vh = vh

        # Carry the RNN state forward to the next timestep
        if new_rnn is not None:
            self._policy_rnn_state = new_rnn

        self._last_log_prob = log_prob

        # Deterministic actions
        # action_mixed = action.clone()
        # action_mixed[self._det_env_ids] = mean_action[self._det_env_ids]
        # log_prob_mixed = torch.zeros_like(log_prob)
        # log_prob_mixed[self._stoch_env_ids] = log_prob[self._stoch_env_ids]

        # action_flat = action_mixed.reshape(self.env.num_envs, -1)
        # mean_flat = mean_action.reshape(self.env.num_envs, -1)

        action[self._det_env_ids] = mean_action[self._det_env_ids]
        log_prob[self._det_env_ids] = 0.0  # Note: won't be used for deterministic envs

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

        # To handle skrl bookkeeping
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.training:

            self._current_next_observations = next_observations

            """ OPTIONAL FUTURE STUFF
            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap:
                rewards += self.cfg.discount_factor * self._current_values * truncated
            """

            n_envs = self.env.num_envs
            n_agents = self.env.num_agents
            action_dim = self.env.action_space.shape[0]
            n_constraints = int(getattr(self.env, "n_constraints", getattr(self.env.unwrapped, "n_constraints", 1)))
            agent_state, goal_state, obs_state = self._extract_graph_states(observations)
            actions = actions.reshape(n_envs, n_agents, action_dim)
            rewards = rewards.reshape(n_envs)
            # DGPPO DEBUG FIX START: canonical live episode-boundary masks.
            terminated_1d = self._env_done_mask(terminated, n_envs)
            truncated_1d = self._env_done_mask(truncated, n_envs)
            done_1d = terminated_1d | truncated_1d
            # DGPPO DEBUG FIX END: canonical live episode-boundary masks.
            log_prob = self._last_log_prob.reshape(n_envs, n_agents)
            value_l_all = self._current_Vl.reshape(n_envs, -1).squeeze(-1)
            value_h_all = self._current_Vh.reshape(n_envs, n_agents, n_constraints)
            # DGPPO DEBUG FIX START: real/env or adapter safety costs.
            costs_all = self._costs_from_infos_or_adapter(
                infos=infos,
                agent_state=agent_state,
                obs_state=obs_state,
                n_envs=n_envs,
                n_agents=n_agents,
                n_constraints=n_constraints,
            )
            # DGPPO DEBUG FIX END: real/env or adapter safety costs.

            stc_rnn_state = self._select_policy_rnn_envs(self._last_policy_rnn_state, self._stoch_env_ids)
            det_rnn_state = self._select_policy_rnn_envs(self._last_policy_rnn_state, self._det_env_ids)
            stc_vl_rnn_state = self._select_env_rnn_envs(self._last_vl_rnn_state, self._stoch_env_ids)
            det_vl_rnn_state = self._select_env_rnn_envs(self._last_vl_rnn_state, self._det_env_ids)

            self.memory.add(
                stc_agent_state=agent_state[self._stoch_env_ids],
                stc_goal_state=goal_state[self._stoch_env_ids],
                stc_obs_state=obs_state[self._stoch_env_ids],
                stc_action=actions[self._stoch_env_ids],
                stc_log_prob=log_prob[self._stoch_env_ids],
                stc_reward=rewards[self._stoch_env_ids],
                stc_cost=costs_all[self._stoch_env_ids],
                stc_value_l=value_l_all[self._stoch_env_ids],
                stc_value_h=value_h_all[self._stoch_env_ids],
                stc_terminated=terminated_1d[self._stoch_env_ids],
                stc_truncated=truncated_1d[self._stoch_env_ids],
                det_agent_state=agent_state[self._det_env_ids],
                det_goal_state=goal_state[self._det_env_ids],
                det_obs_state=obs_state[self._det_env_ids],
                det_action=actions[self._det_env_ids],
                det_log_prob=log_prob[self._det_env_ids],
                det_reward=rewards[self._det_env_ids],
                det_cost=costs_all[self._det_env_ids],
                det_value_l=value_l_all[self._det_env_ids],
                det_value_h=value_h_all[self._det_env_ids],
                det_terminated=terminated_1d[self._det_env_ids],
                det_truncated=truncated_1d[self._det_env_ids],
                stc_rnn_state=stc_rnn_state,
                det_rnn_state=det_rnn_state,
                stc_vl_rnn_state=stc_vl_rnn_state,
                det_vl_rnn_state=det_vl_rnn_state,
            )

            # DGPPO DEBUG FIX START: skrl PPO_RNN-style recurrent reset.
            self._reset_rnn_states_for_done(done_1d)
            # DGPPO DEBUG FIX END: skrl PPO_RNN-style recurrent reset.

            self._debug.record_transition(
                timestep=timestep,
                timesteps=timesteps,
                env=self.env,
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                agent_state=agent_state,
                goal_state=goal_state,
                obs_state=obs_state,
                log_prob=log_prob,
                value_l=value_l_all,
                value_h=value_h_all,
                costs=costs_all,
                old_policy_rnn_state=self._last_policy_rnn_state,
                new_policy_rnn_state=self._policy_rnn_state,
                vl_rnn_state=self._vl_rnn_state,
                stoch_env_ids=self._stoch_env_ids,
                det_env_ids=self._det_env_ids,
                memory_cursor=self.memory.cursor,
                rollout_length=self.memory.rollout_length,
            )

            if self.memory.is_full:
                with torch.no_grad():
                    next_graph = self._build_graph(next_observations, next_states)
                    vl_boot, _ = self.Vl(next_graph, self._vl_rnn_state, self.env.num_agents)
                    vh_boot, _ = self.Vh(next_graph, self._policy_rnn_state, self.env.num_agents)
                    policy_reference_state = None
                    vh_boot_reference = None
                    if self.policy.rnn is not None and self._last_policy_rnn_state is not None:
                        _action, _log_prob, _mode, policy_reference_state = self.policy.act(
                            next_graph,
                            rnn_state=self._last_policy_rnn_state,
                            n_agents_total=self.env.num_agents,
                            deterministic=True,
                        )
                        vh_boot_reference, _ = self.Vh(next_graph, policy_reference_state, self.env.num_agents)
                    elif self.policy.rnn is None:
                        vh_boot_reference = vh_boot
                self._debug.log_bootstrap(
                    timestep=timestep,
                    vl_boot=vl_boot,
                    vh_boot=vh_boot,
                    vh_boot_reference=vh_boot_reference,
                    policy_bootstrap_rnn_state=self._policy_rnn_state,
                    policy_reference_rnn_state=policy_reference_state,
                )
                self.memory.set_final_values("stc", vl_boot[self._stoch_env_ids], vh_boot[self._stoch_env_ids])
                self.memory.set_final_values("det", vl_boot[self._det_env_ids], vh_boot[self._det_env_ids])

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass  # or super() — in any case does nothing for on-policy

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
        if self.memory is None or self.memory.cursor < self.rollouts:
            return

        self._update_id += 1
        update_id = self._update_id
        view = self.memory.as_bTah_view("stc")
        det_view = self.memory.as_bTah_view("det")

        # Compute returns and advantages.
        _Qh_stc, Ql = compute_dec_ocp_gae(
            Tah_hs=view["bTah_hs"],
            T_l=view["bT_l"],
            Tp1ah_Vh=view["bTp1ah_Vh"],
            Tp1_Vl=view["bTp1_Vl"],
            disc_gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            # DGPPO DEBUG FIX START: stochastic rollout-boundary masks.
            T_terminated=view["bT_terminated"],
            T_truncated=view["bT_truncated"],
            bootstrap_on_truncated=self.bootstrap_on_truncated,
            # DGPPO DEBUG FIX END: stochastic rollout-boundary masks.
        )

        Qh_det, _ = compute_dec_ocp_gae(
            Tah_hs=det_view["bTah_hs"],
            T_l=det_view["bT_l"],
            Tp1ah_Vh=det_view["bTp1ah_Vh"],
            Tp1_Vl=view["bTp1_Vl"],
            disc_gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            # DGPPO DEBUG FIX START: deterministic rollout-boundary masks.
            T_terminated=det_view["bT_terminated"],
            T_truncated=det_view["bT_truncated"],
            bootstrap_on_truncated=self.bootstrap_on_truncated,
            # DGPPO DEBUG FIX END: deterministic rollout-boundary masks.
        )

        # flipped for PPO use
        cbf_scale = self._cbf_scale(timestep=timestep, timesteps=timesteps)
        adv_info = compute_cbf_advantages(
            bT_Ql=Ql,
            bT_Vl=view["bT_Vl"],
            bTah_Vh=view["bTah_Vh"],
            bTp1ah_Vh=view["bTp1ah_Vh"],
            alpha=self.alpha,
            cbf_eps=self.cbf_eps,
            cbf_weight=self.cbf_weight,
            dt=self.env._step_dt,
            cbf_scale=cbf_scale,
            bT_done=view["bT_done"],
        )
        bTa_A = adv_info["bTa_A"].detach()
        self.track_data("DGPPO/safe_rate", float(adv_info["bTa_is_safe"].float().mean().item()))
        self.track_data("DGPPO/adv_raw_mean", float(adv_info["bT_Al_raw"].mean().item()))

        graph = build_rollout_graph(view=view, obs_radius=self.obs_radius)
        det_graph = build_rollout_graph(view=det_view, obs_radius=self.obs_radius)
        chunk_ids = self._rnn_chunk_ids(T=view["bTa_actions"].shape[1], device=torch.device("cpu"))
        self._debug.log_update_start(
            timestep=timestep,
            update_id=update_id,
            view=view,
            det_view=det_view,
            ql=Ql,
            qh_det=Qh_det,
            adv_info=adv_info,
            cbf_scale=cbf_scale,
            chunk_ids=chunk_ids,
        )

        loss_p_acc = bTa_A.new_zeros(())
        loss_vl_acc = bTa_A.new_zeros(())
        loss_vh_acc = bTa_A.new_zeros(())
        clipfrac_acc = bTa_A.new_zeros(())
        n_minibatches = 0

        # Learning epochs.
        for epoch in range(self.learning_epochs):
            sampled_batches = self.memory.sample_minibatches(self.mini_batches)
            for minibatch, idx in enumerate(sampled_batches):
                info = self._update_minibatch(
                    idx=idx,
                    Qh_det=Qh_det,
                    Ql=Ql,
                    bTa_A=bTa_A,
                    view=view,
                    det_view=det_view,
                    graph=graph,
                    det_graph=det_graph,
                    chunk_ids=chunk_ids,
                )
                self._debug.log_minibatch(
                    timestep=timestep,
                    update_id=update_id,
                    epoch=epoch,
                    minibatch=minibatch,
                    idx=idx,
                    info=info,
                )
                loss_p_acc += info["loss_p"]
                loss_vl_acc += info["loss_vl"]
                loss_vh_acc += info["loss_vh"]
                clipfrac_acc += info["clip_frac"]
                n_minibatches += 1

        if n_minibatches == 0:
            self._debug.reset_rollout()
            return

        inv_n = 1.0 / float(n_minibatches)
        update_summary = {
            "loss_policy": float((loss_p_acc * inv_n).item()),
            "loss_value_l": float((loss_vl_acc * inv_n).item()),
            "loss_value_h": float((loss_vh_acc * inv_n).item()),
            "clip_frac": float((clipfrac_acc * inv_n).item()),
            "lr_policy": float(self._policy_opt.param_groups[0]["lr"]),
            "lr_vl": float(self._vl_opt.param_groups[0]["lr"]),
            "lr_vh": float(self._vh_opt.param_groups[0]["lr"]),
        }
        self.track_data("DGPPO/loss_policy", update_summary["loss_policy"])
        self.track_data("DGPPO/loss_value_l", update_summary["loss_value_l"])
        self.track_data("DGPPO/loss_value_h", update_summary["loss_value_h"])
        self.track_data("DGPPO/clip_frac", update_summary["clip_frac"])
        self.track_data("DGPPO/lr_policy", float(self._policy_opt.param_groups[0]["lr"]))
        self.track_data("DGPPO/lr_vl", float(self._vl_opt.param_groups[0]["lr"]))
        self.track_data("DGPPO/lr_vh", float(self._vh_opt.param_groups[0]["lr"]))
        self._debug.log_update_end(timestep=timestep, update_id=update_id, summary=update_summary)

        self.memory.reset()
        self._debug.reset_rollout()

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
        """Single PPO minibatch step over one chunk of the ``B`` axis."""
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
                rnn_states=batch.rnn_states,
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
                n_agents_total=batch.A,
                rnn_state=None,
            )
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
            A=batch.A,
            vl_loss_scale=self.vl_loss_scale,
            vh_loss_scale=self.vh_loss_scale,
            rnn_states=batch.rnn_states,
            det_rnn_states=batch.det_rnn_states,
            vl_rnn_states=batch.vl_rnn_states,
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
            "loss_p": policy_info["loss_policy"].detach(),
            "loss_vl": value_info["loss_vl"].detach(),
            "loss_vh": value_info["loss_vh"].detach(),
            "clip_frac": policy_info["clip_frac"].detach(),
            "ratio": policy_info["ratio"].detach(),
            "log_prob": policy_info["log_prob"].detach(),
            "old_logp": policy_info["old_logp"].detach(),
            "log_prob_delta": policy_info["log_prob_delta"].detach(),
            "advantages": policy_info["advantages"].detach(),
            "entropy": policy_info["entropy"].detach(),
            "policy_grad_norm": policy_grad_norm.detach(),
            "vl_grad_norm": vl_grad_norm.detach(),
            "vh_grad_norm": vh_grad_norm.detach(),
            "vl": value_info["vl"].detach(),
            "vh": value_info["vh"].detach(),
        }

    def load_dgppo_hyperparameters(self) -> None:
        self.gamma: float = float(self.cfg.get("discount_factor", 0.99))
        self.gae_lambda: float = float(self.cfg.get("gae_lambda", self.cfg.get("lambda", 0.95)))
        self.bootstrap_on_truncated: bool = bool(self.cfg.get("bootstrap_on_truncated", False))
        self.learning_starts: int = int(self.cfg.get("learning_starts", 0))
        self.rollouts: int = int(self.cfg.get("rollouts", 32))
        self.rnn_step: int = int(self.cfg.get("rnn_step", min(16, self.rollouts)))
        self.learning_epochs: int = int(self.cfg.get("learning_epochs", 8))
        self.mini_batches: int = int(self.cfg.get("mini_batches", 8))
        self.clip_eps: float = float(self.cfg.get("ratio_clip", 0.2))
        self.alpha: float = float(self.cfg.get("alpha", 10.0))
        self.cbf_eps: float = float(self.cfg.get("cbf_eps", 1e-2))
        self.cbf_weight: float = float(self.cfg.get("cbf_weight", 1.0))
        self.cbf_schedule: bool = bool(self.cfg.get("cbf_schedule", True))
        self.grad_clip: float = float(self.cfg.get("grad_norm_clip", 2.0))
        self.entropy_scale: float = float(self.cfg.get("entropy_loss_scale", 0.0))
        self.vl_loss_scale: float = float(self.cfg.get("vl_loss_scale", 1.0))
        self.vh_loss_scale: float = float(self.cfg.get("vh_loss_scale", 1.0))
        self.obs_radius: float = float(self.cfg.get("obs_radius", 2.0))
        self.lr_policy: float = float(self.cfg.get("lr_policy", 3e-4))
        self.lr_vl: float = float(self.cfg.get("lr_vl", 1e-3))
        self.lr_vh: float = float(self.cfg.get("lr_vh", 1e-3))

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

    def _extract_graph_states(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode flat policy observations into graph node state tensors."""
        return extract_graph_states_from_flat_obs(
            observations,
            self.env.unwrapped.graph_obs_layout,
            n_agents=self.env.num_agents,
        )

    def _build_graph(self, observations: torch.Tensor, states: torch.Tensor | None) -> GraphData:
        """Parse the flat policy-obs tensor into structured node states and build the graph."""
        agent_state, goal_state, obs_state = self._extract_graph_states(observations)

        return build_graph_data(
            agent_state=agent_state,
            goal_state=goal_state,
            obs_state=obs_state,
            obs_radius=self.obs_radius,
        )

    def _select_policy_rnn_envs(
        self, rnn_state: torch.Tensor | None, env_ids: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Select env-major policy carry as ``[L, B*A, C, H]`` for memory."""
        if rnn_state is None or env_ids is None:
            return None
        L, _N, C, H = rnn_state.shape
        A = self.env.num_agents
        state = rnn_state.reshape(L, self.env.num_envs, A, C, H)
        return state[:, env_ids].reshape(L, int(env_ids.numel()) * A, C, H)

    def _select_env_rnn_envs(self, rnn_state: torch.Tensor | None, env_ids: torch.Tensor | None) -> torch.Tensor | None:
        if rnn_state is None or env_ids is None:
            return None
        return rnn_state[:, env_ids]

    # DGPPO DEBUG FIX START: live rollout mask/cost/RNN helpers.
    def _env_done_mask(self, mask: torch.Tensor, n_envs: int) -> torch.Tensor:
        """Return a flat boolean mask with one entry per IsaacLab env."""
        return torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(n_envs)

    def _reset_rnn_states_for_done(self, done: torch.Tensor) -> None:
        """Reset recurrent state on ``terminated | truncated`` like skrl PPO_RNN."""
        self._policy_rnn_state = zero_policy_rnn_states_for_done(
            self._policy_rnn_state,
            done,
            n_agents=self.env.num_agents,
        )
        self._vl_rnn_state = zero_env_rnn_states_for_done(self._vl_rnn_state, done)

    def _costs_from_infos_or_adapter(
        self,
        *,
        infos: Any,
        agent_state: torch.Tensor,
        obs_state: torch.Tensor,
        n_envs: int,
        n_agents: int,
        n_constraints: int,
    ) -> torch.Tensor:
        """Prefer env-provided costs, otherwise derive signed safety costs from graph states."""
        costs_all = None
        if isinstance(infos, Mapping):
            costs_all = infos.get("costs", infos.get("cost"))
        if costs_all is None:
            costs_all = self._adapter_safety_costs(
                agent_state=agent_state,
                obs_state=obs_state,
                n_constraints=n_constraints,
            )
        else:
            costs_all = torch.as_tensor(costs_all, device=self.device, dtype=torch.float32)
            if costs_all.ndim == 1:
                costs_all = costs_all[:, None, None]
            elif costs_all.ndim == 2:
                costs_all = costs_all[:, :, None]
            costs_all = costs_all.reshape(n_envs, n_agents, -1)
        return align_safety_cost_heads(costs_all.to(device=self.device, dtype=torch.float32), n_constraints)

    def _adapter_safety_costs(
        self,
        *,
        agent_state: torch.Tensor,
        obs_state: torch.Tensor,
        n_constraints: int,
    ) -> torch.Tensor:
        """Build signed arena/pillar costs when the IsaacLab info dict has no cost key."""
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        cfg = getattr(base_env, "cfg", None)
        if cfg is None:
            fallback = agent_state.new_full((*agent_state.shape[:2], max(1, int(n_constraints))), -1.0)
            return align_safety_cost_heads(fallback, n_constraints)

        default_pillar_top_z = float(getattr(cfg, "arena_min")[2]) + float(getattr(cfg, "pillar_height", 0.0))
        pillar_xy = getattr(base_env, "_pillar_positions_xy", None)
        if pillar_xy is not None:
            physical_obs_state = agent_state.new_zeros(agent_state.shape[0], pillar_xy.shape[0], agent_state.shape[-1])
            physical_obs_state[:, :, :2] = pillar_xy.unsqueeze(0).expand(agent_state.shape[0], -1, -1)
        else:
            physical_obs_state = obs_state

        costs = compute_pos_tracking_safety_costs(
            agent_state=agent_state,
            obs_state=physical_obs_state,
            arena_min=getattr(base_env, "_arena_min_safe", getattr(base_env, "_arena_min", getattr(cfg, "arena_min"))),
            arena_max=getattr(base_env, "_arena_max_safe", getattr(base_env, "_arena_max", getattr(cfg, "arena_max"))),
            collision_altitude=float(getattr(cfg, "collision_altitude")),
            pillar_collision_radius=float(
                getattr(base_env, "_pillar_collision_radius", getattr(cfg, "pillar_radius", 0.0))
            ),
            pillar_top_z=float(getattr(base_env, "_pillar_top_z", default_pillar_top_z)),
        )
        return align_safety_cost_heads(costs, n_constraints)
    # DGPPO DEBUG FIX END: live rollout mask/cost/RNN helpers.
