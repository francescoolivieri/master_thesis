import torch
import time
from typing import Any, Mapping, Optional
from skrl.agents.torch import Agent, AgentCfg
from .dgppo_memory import DGPPORolloutMemory
from .dgppo_models import DGPPOValueNet, DGPPOPolicy
from .update_helpers import (
    apply_policy_update,
    apply_value_update,
    build_update_graph_batch,
    compute_policy_loss,
    compute_value_losses,
)
from .utils import (
    GraphData,
    build_graph_data,
    compute_cbf_advantages,
    compute_dec_ocp_gae,
    extract_graph_states_from_flat_obs,
)


class DGPPOAgent(Agent):
    def __init__(self,
        policy: DGPPOPolicy,
        Vl: DGPPOValueNet,
        Vh: DGPPOValueNet,
        env: Any, #remove
        cfg: AgentCfg, 
        observation_space,
        action_space,
        device: torch.device,
    ) -> None:        
        
        super().__init__(
            models={"policy": policy, "Vl": Vl, "Vh": Vh},
            memory=None,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        
        #self.training = False
        
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
            list(self.Vl.gnn.parameters())
            + list(self.Vl.head.parameters())
            + list(self.Vl.net.value_out.parameters())
        )
        if self.Vl.rnn is not None:
            vl_params += list(self.Vl.rnn.parameters())
        vh_params = (
            list(self.Vh.gnn.parameters())
            + list(self.Vh.head.parameters())
            + list(self.Vh.net.value_out.parameters())
        )
        if self.Vh.rnn is not None:
            vh_params += list(self.Vh.rnn.parameters())
        self._vl_opt = torch.optim.Adam(vl_params, lr=self.lr_vl)
        self._vh_opt = torch.optim.Adam(vh_params, lr=self.lr_vh)
        self._vl_grad_params = vl_params
        self._vh_grad_params = vh_params
        
        # Setup memory (Note: personalised memory for cleaner code)
        self.memory: Optional[DGPPORolloutMemory] = None
        self.num_envs = self.cfg.get("num_envs", 1)
        self._rnn_state = None
        self._stoch_env_ids = None
        self._det_env_ids = None
        self._last_graph = None
        self._last_log_prob = None
        self._current_Vl = None
        self._current_Vh = None
        self._current_next_observations = None
        
    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """
        Called once by the trainer before the first interaction.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)
        
        if self.memory is not None:
            return
        
        ###
        rollout_length = int(self.cfg.get("rollouts", 32))  # from AgentCfg
        n_agents = self.env.num_agents
        layout = self.env.unwrapped.graph_obs_layout
        n_obs = int(layout.get("n_obstacles", 0))
        state_dim = self.env.state_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        n_constraints = int(getattr(self.env, "n_constraints", getattr(self.env.unwrapped, "n_constraints", 1)))
        use_rnn = self.policy.use_rnn

        # Allocate the DGPPO rollout memory now that we know the env shape
        if self.env.num_envs < 2 or (self.env.num_envs % 2) != 0:
            raise ValueError(
                f"DGPPO split rollout requires an even num_envs >= 2, got {self.env.num_envs}"
            )
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
        )

        # Initial RNN carry for the rollout (B * A agents, regardless of env).
        if self.policy.use_rnn:
            self._rnn_state = self.policy.initialize_carry(
                n_agents_total=self.env.num_envs * n_agents, device=self.device
            )
        
     
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
        n_agents_total = self.env.num_envs * self.env.num_agents

        
        with torch.no_grad():  #  Saves memory
            graph = self._build_graph(observations, states)
            self._last_graph = graph

            action, log_prob, mean_action, new_rnn = self.policy.act(
                graph,
                rnn_state=self._rnn_state,
                n_agents_total=n_agents_total,
                deterministic=not self.training,  
            )

            if self.training:
                # ?? Shouldn t I store also the rrn states?
                vl, _ = self.Vl(graph, self._rnn_state, self.env.num_agents)
                vh, _ = self.Vh(graph, self._rnn_state, self.env.num_agents)
                self._current_Vl = vl
                self._current_Vh = vh


        # Carry the RNN state forward to the next timestep
        if new_rnn is not None:
            self._rnn_state = new_rnn

        self._last_log_prob = log_prob

        # Determinstic actions
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
        self.training = (mode == "train")
        self.policy.train(self.training)   # nn.Module.train()
        self.Vl.train(self.training)
        self.Vh.train(self.training)
        

    ### CHECK FROM HERE

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
        super().record_transition(observations=observations, states=states, actions=actions, rewards=rewards, next_observations=next_observations, next_states=next_states, terminated=terminated, truncated=truncated, infos=infos, timestep=timestep, timesteps=timesteps)
            
            
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
            log_prob = self._last_log_prob.reshape(n_envs, n_agents)
            value_l_all = self._current_Vl.reshape(n_envs, -1).squeeze(-1)
            value_h_all = self._current_Vh.reshape(n_envs, n_agents, n_constraints)
            costs_all = None
            if isinstance(infos, Mapping):
                costs_all = infos.get("costs", infos.get("cost", None))
            if costs_all is None:
                costs_all = torch.zeros(
                    n_envs, n_agents, n_constraints, device=self.device, dtype=torch.float32
                )
            else:
                costs_all = torch.as_tensor(costs_all, device=self.device, dtype=torch.float32)
                if costs_all.ndim == 1:
                    costs_all = costs_all[:, None, None]
                elif costs_all.ndim == 2:
                    costs_all = costs_all[:, :, None]
                costs_all = costs_all.reshape(n_envs, n_agents, n_constraints)

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
                det_agent_state=agent_state[self._det_env_ids],
                det_goal_state=goal_state[self._det_env_ids],
                det_obs_state=obs_state[self._det_env_ids],
                det_action=actions[self._det_env_ids],
                det_log_prob=log_prob[self._det_env_ids],
                det_reward=rewards[self._det_env_ids],
                det_cost=costs_all[self._det_env_ids],
                det_value_l=value_l_all[self._det_env_ids],
                det_value_h=value_h_all[self._det_env_ids],
            )

            if self.memory.is_full:
                with torch.no_grad():
                    next_graph = self._build_graph(next_observations, next_states)
                    vl_boot, _ = self.Vl(next_graph, self._rnn_state, self.env.num_agents)
                    vh_boot, _ = self.Vh(next_graph, self._rnn_state, self.env.num_agents)
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
        )
        
        Qh_det, _ = compute_dec_ocp_gae(
            Tah_hs=det_view["bTah_hs"],
            T_l=det_view["bT_l"],
            Tp1ah_Vh=det_view["bTp1ah_Vh"],
            Tp1_Vl=view["bTp1_Vl"],
            disc_gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # flipped for PPO use
        adv_info = compute_cbf_advantages(
            bT_Ql=Ql,
            bT_Vl=view["bT_Vl"],
            bTah_Vh=view["bTah_Vh"],
            bTp1ah_Vh=view["bTp1ah_Vh"],
            alpha=self.alpha,
            cbf_eps=self.cbf_eps,
            cbf_weight=self.cbf_weight,
            dt=self.env.dt,
            cbf_scale=self._cbf_scale(timestep=timestep, timesteps=timesteps),
        )
        bTa_A = adv_info["bTa_A"].detach()
        self.track_data("DGPPO/safe_rate", float(adv_info["bTa_is_safe"].float().mean().item()))
        self.track_data("DGPPO/adv_raw_mean", float(adv_info["bT_Al_raw"].mean().item()))

        loss_p_acc = loss_vl_acc = loss_vh_acc = clipfrac_acc = 0.0
        n_minibatches = 0

        # Learning epochs.
        for _ in range(self.learning_epochs):
            sampled_batches = self.memory.sample_minibatches(self.mini_batches)
            for idx in sampled_batches:
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

        if n_minibatches == 0:
            return

        inv_n = 1.0 / float(n_minibatches)
        self.track_data("DGPPO/loss_policy", loss_p_acc * inv_n)
        self.track_data("DGPPO/loss_value_l", loss_vl_acc * inv_n)
        self.track_data("DGPPO/loss_value_h", loss_vh_acc * inv_n)
        self.track_data("DGPPO/clip_frac", clipfrac_acc * inv_n)
        self.track_data("DGPPO/lr_policy", float(self._policy_opt.param_groups[0]["lr"]))
        self.track_data("DGPPO/lr_vl", float(self._vl_opt.param_groups[0]["lr"]))
        self.track_data("DGPPO/lr_vh", float(self._vh_opt.param_groups[0]["lr"]))

        self.memory.reset()
    
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
        batch = build_update_graph_batch(
            idx=idx,
            view=view,
            det_view=det_view,
            qh_det=Qh_det,
            ql=Ql,
            advantages=bTa_A,
            obs_radius=self.obs_radius,
        )

        # ---- Policy ----
        policy_info = compute_policy_loss(
            policy=self.policy,
            graph=batch.graph,
            actions=batch.actions,
            old_logp=batch.old_logp,
            advantages=batch.advantages,
            clip_eps=self.clip_eps,
            entropy_scale=self.entropy_scale,
            n_agents_total=batch.n_agents_total,
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
            "loss_p": float(policy_info["loss_policy"].item()),
            "loss_vl": float(value_info["loss_vl"].item()),
            "loss_vh": float(value_info["loss_vh"].item()),
            "clip_frac": float(policy_info["clip_frac"].item()),
        }
    
    def load_dgppo_hyperparameters(self) -> None:
        self.gamma: float = float(self.cfg.get("discount_factor", 0.99))
        self.gae_lambda: float = float(self.cfg.get("gae_lambda", self.cfg.get("lambda", 0.95)))
        self.learning_starts: int = int(self.cfg.get("learning_starts", 0))
        self.rollouts: int = int(self.cfg.get("rollouts", 32))
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


    def _extract_graph_states(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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