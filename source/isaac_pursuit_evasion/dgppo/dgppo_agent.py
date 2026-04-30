import torch
import time
import torch.nn.functional as F
from typing import Any, Mapping, Optional
from skrl.agents.torch import Agent, AgentCfg
from .dgppo_memory import DGPPORolloutMemory
from .dgppo_models import DGPPOValueNet, DGPPOPolicy
from .utils import compute_cbf_advantages, compute_dec_ocp_gae, compute_policy_surrogate, GraphData


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
        # Gather the minibatch (axes: B first after ``as_bTah_view``).
        agent_s = view["bTa_agent_state"][idx]       # (b, T, A, S)
        goal_s = view["bTa_goal_state"][idx]
        obs_s = view["bTo_obs_state"][idx]
        actions = view["bTa_actions"][idx]           # (b, T, A, Da)
        old_logp = view["bTa_logp"][idx]             # (b, T, A)
        Qh_det_mb = Qh_det[idx]                      # (b, T, A, NH)
        Ql_mb = Ql[idx]                              # (b, T)
        adv_mb = bTa_A[idx]               
    
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

        self._policy_opt.zero_grad(set_to_none=True)
        loss_p_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self._policy_opt.step()

        # ---- Critic Vl ----
        vl, _ = self.Vl(graph, None, A)
        vl = vl.reshape(b, T)
        loss_vl = self.vl_loss_scale * F.mse_loss(vl, Ql_mb)

        self._vl_opt.zero_grad(set_to_none=True)
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
        vh, _ = self.Vh(det_graph, None, A)
        vh = vh.reshape(b, T, A, -1)
        loss_vh = self.vh_loss_scale * F.mse_loss(vh, Qh_det_mb)

        self._vh_opt.zero_grad(set_to_none=True)
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
        layout: dict = self.env.unwrapped.graph_obs_layout
        E = observations.shape[0]
        S = int(layout["state_dim"])
        A = int(layout.get("n_agents", self.env.num_agents))
        O = int(layout["n_obstacles"])

        agent_flat = observations[:, : layout["agent_end"]]
        agent_state = agent_flat.reshape(E, A, S)

        goal_pos_flat = observations[:, layout["agent_end"] : layout["goal_end"]]
        goal_pos = goal_pos_flat.reshape(E, A, 3)
        goal_state = torch.cat([goal_pos, goal_pos.new_zeros(E, A, S - 3)], dim=-1)

        if O > 0:
            obstacle_xy = observations[:, layout["goal_end"] : layout["obstacles_end"]]
            obstacle_xy = obstacle_xy.reshape(E, O, 2)
            obs_state = torch.cat([obstacle_xy, obstacle_xy.new_zeros(E, O, S - 2)], dim=-1)
        else:
            obs_state = observations.new_zeros(E, 0, S)

        return agent_state, goal_state, obs_state

    def _build_graph(self, observations: torch.Tensor, states: torch.Tensor | None) -> GraphData:
        """Parse the flat policy-obs tensor into structured node states and build the graph."""
        agent_state, goal_state, obs_state = self._extract_graph_states(observations)

        return build_graph_data(
            agent_state=agent_state,
            goal_state=goal_state,
            obs_state=obs_state,
            obs_radius=self.obs_radius,
        )




# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Node-type integer IDs
AGENT_TYPE: int = 0   # moving agents
GOAL_TYPE:  int = 1   # goal positions (one per agent)
OBS_TYPE:   int = 2   # static obstacles
PAD_TYPE:   int = -1  # dummy padding node (absorbs masked-out edges)

# One-hot indicator length; order: [obstacle_bit, goal_bit, agent_bit]
NUM_TYPE_INDICATORS: int = 3


def build_graph_data(
    agent_state: torch.Tensor,         # (E, A, S)
    goal_state:  torch.Tensor,         # (E, A, S)  one goal per agent
    obs_state:   torch.Tensor | None,  # (E, O, S)  or None when no obstacles
    *,
    obs_radius: float,
) -> GraphData:
    """Build a batched GraphData for E parallel environments.

    Node ordering within each sub-graph: [agents | goals | obstacles | pad].
    Sub-graphs are concatenated along the leading node/edge axis so the whole
    batch is represented by a single flat GraphData (jraph convention).
    
    Vectorized approach for efficiency.

    Args:
        agent_state: Per-agent physical state (E, A, S).
        goal_state:  Goal state, one per agent (E, A, S).
        obs_state:   Obstacle states (E, O, S), or None.
        obs_radius:  Proximity radius for A-A and A-O edge activation.
    """
    assert agent_state.dim() == 3 and goal_state.dim() == 3
    assert agent_state.shape[0] == goal_state.shape[0], "E must match"
    assert agent_state.shape[1] == goal_state.shape[1], "need one goal per agent"
    assert agent_state.shape[2] == goal_state.shape[2], "state dim must match"

    E, A, S = agent_state.shape
    device = agent_state.device

    # Normalise: absent obstacles -> empty tensor 
    if obs_state is None:
        obs_state = agent_state.new_zeros(E, 0, S)
    assert obs_state.shape[0] == E and obs_state.shape[2] == S
    O = obs_state.shape[1]

    # 1. Node features  (E, N_per, node_dim) — then flattened to (E*N_per, node_dim)  
    N_per = A + A + O + 1 # N_per = A agents + A goals + O obstacles + 1 padding node
    nodes, states, node_types = _make_node_features(agent_state, goal_state, obs_state)
    
    nodes_flat      = nodes.reshape(E * N_per, -1)
    states_flat     = states.reshape(E * N_per, -1)
    node_types_flat = node_types.reshape(E * N_per)

    # 2. Edges
    # Global node ids: each env e owns a contiguous block [e*N_per, (e+1)*N_per).
    # Local ids are broadcast across E via env_offsets.
    env_offsets = (torch.arange(E, device=device) * N_per).unsqueeze(1)  # (E, 1)

    agent_ids = torch.arange(A,          device=device).unsqueeze(0) + env_offsets   # (E, A)
    goal_ids  = torch.arange(A,  2*A,    device=device).unsqueeze(0) + env_offsets   # (E, A)
    obs_ids   = torch.arange(2*A, 2*A+O, device=device).unsqueeze(0) + env_offsets   # (E, O)
    pad_ids   = (N_per - 1 + env_offsets.squeeze(1)).long()                          # (E,)

    edges_flat, recvs_flat, sends_flat, n_edges_per_env = _make_edge_list(
        agent_state, goal_state, obs_state,
        agent_ids, goal_ids, obs_ids, pad_ids,
        obs_radius=obs_radius,
    )

    n_nodes = torch.full((E,), N_per,           dtype=torch.long, device=device)
    n_edges = torch.full((E,), n_edges_per_env, dtype=torch.long, device=device)

    return GraphData(
        n_nodes=n_nodes,
        n_edges=n_edges,
        nodes=nodes_flat,
        edges=edges_flat,
        states=states_flat,
        receivers=recvs_flat.long(),
        senders=sends_flat.long(),
        node_types=node_types_flat,
    )


def _make_node_features(
    agent_state: torch.Tensor,   # (E, A, S)
    goal_state:  torch.Tensor,   # (E, A, S)
    obs_state:   torch.Tensor,   # (E, O, S)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble batched node feature, state and node-type tensors.

    Returns:
        nodes:      (E, N_per, S+3)  state vector concatenated with type indicator
        states:     (E, N_per, S)    states (used by CBF)
        node_types: (E, N_per)       integer type ids 
    """
    E, A, S = agent_state.shape
    G = goal_state.shape[1]   # == A
    O = obs_state.shape[1]
    device, dtype = agent_state.device, agent_state.dtype

    # Physical states: concat agents | goals | obstacles | pad(-1 sentinel)
    state_pad = torch.full((E, 1, S), -1.0, dtype=dtype, device=device)
    states = torch.cat([agent_state, goal_state, obs_state, state_pad], dim=1)  # (E, N, S)

    # One-hot type indicators, appended to the state vector.
    N = A + G + O + 1
    indicator = torch.zeros(E, N, NUM_TYPE_INDICATORS, dtype=dtype, device=device)
    indicator[:, :A,        2] = 1.0   # agent    bit
    indicator[:, A:A+G,     1] = 1.0   # goal     bit
    indicator[:, A+G:A+G+O, 0] = 1.0   # obstacle bit
    # pad row stays all-zero

    # nodes = [  x, y, vx, vy, ...  |  obs_bit, goal_bit, agent_bit  ]
    #         <── state (dim S) ──>  <── type indicator (dim 3) ──>
    nodes = torch.cat([states, indicator], dim=-1)  # (E, N, S+3)

    # Integer node-type ids (used by GNN's get_type_nodes)
    node_types = torch.full((E, N), PAD_TYPE, dtype=torch.long, device=device)
    node_types[:, :A]        = AGENT_TYPE
    node_types[:, A:A+G]     = GOAL_TYPE
    node_types[:, A+G:A+G+O] = OBS_TYPE

    return nodes, states, node_types


def _make_edge_list(
    agent_state: torch.Tensor,   # (E, A, S)
    goal_state:  torch.Tensor,   # (E, A, S)
    obs_state:   torch.Tensor,   # (E, O, S)
    agent_ids:   torch.Tensor,   # (E, A)  global node ids
    goal_ids:    torch.Tensor,   # (E, A)
    obs_ids:     torch.Tensor,   # (E, O)
    pad_ids:     torch.Tensor,   # (E,)    id of each env's padding node
    *,
    obs_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the flat edge list for all E environments in one vectorized pass.

    Returns:
        edges_flat:      (E * n_edges_per_env, node_dim)
        recvs_flat:      (E * n_edges_per_env,)
        sends_flat:      (E * n_edges_per_env,)
        n_edges_per_env: fixed edge count per sub-graph (masked ones route to pad)

    Inactive edges (outside obs_radius, non-diagonal A-G pairs) are redirected to the
    per-env pad node so the edge count is identical across all environments.
    """
    E, A, S = agent_state.shape
    O = obs_state.shape[1]
    device = agent_state.device

    a_pos = agent_state[..., :2]   # (E, A, 2) — 2-D positions for cdist

    # -- Agent → Agent: within obs_radius, no self-loops --
    dist_aa  = torch.cdist(a_pos, a_pos)                                       # (E, A, A)
    aa_mask  = (dist_aa < obs_radius) & ~torch.eye(A, dtype=torch.bool, device=device)
    aa_feats = agent_state[:, :, None, :] - agent_state[:, None, :, :]        # (E, A, A, S)
    aa_f, aa_r, aa_s = _flatten_dense_edge_block(aa_feats, aa_mask, agent_ids, agent_ids, pad_ids)

    # -- Agent → Goal: identity pairing only (agent i → goal i) --
    diag     = torch.arange(A, device=device)
    ag_feats = agent_state.new_zeros(E, A, A, S)
    ag_feats[:, diag, diag, :] = agent_state - goal_state                      # diagonal only
    ag_mask  = torch.eye(A, dtype=torch.bool, device=device).unsqueeze(0)      # (1, A, A) broadcasts
    ag_f, ag_r, ag_s = _flatten_dense_edge_block(ag_feats, ag_mask, agent_ids, goal_ids, pad_ids)

    edge_f_parts    = [aa_f, ag_f]
    recv_parts      = [aa_r, ag_r]
    send_parts      = [aa_s, ag_s]
    n_edges_per_env = A * A + A * A   # fixed; masked entries route to pad

    # -- Agent → Obstacle: within obs_radius (skipped when O == 0) --
    if O > 0:
        o_pos    = obs_state[..., :2]
        dist_ao  = torch.cdist(a_pos, o_pos)                                   # (E, A, O)
        ao_mask  = dist_ao < obs_radius
        ao_feats = agent_state[:, :, None, :] - obs_state[:, None, :, :]      # (E, A, O, S)
        ao_f, ao_r, ao_s = _flatten_dense_edge_block(ao_feats, ao_mask, agent_ids, obs_ids, pad_ids)
        edge_f_parts.append(ao_f)
        recv_parts.append(ao_r)
        send_parts.append(ao_s)
        n_edges_per_env += A * O

    edges_flat = torch.cat(edge_f_parts, dim=0)
    recvs_flat = torch.cat(recv_parts,   dim=0)
    sends_flat = torch.cat(send_parts,   dim=0)

    # Zero-pad edge features to node_dim (type-indicator slots stay 0)
    edges_flat = torch.cat(
        [edges_flat, edges_flat.new_zeros(edges_flat.shape[0], NUM_TYPE_INDICATORS)],
        dim=-1,
    )

    return edges_flat, recvs_flat, sends_flat, n_edges_per_env


def _flatten_dense_edge_block(
    edge_feats: torch.Tensor,   # (E, n_recv, n_send, F)  dense feature grid
    edge_mask:  torch.Tensor,   # (E, n_recv, n_send)     True = active edge
    recv_ids:   torch.Tensor,   # (E, n_recv)             global receiver node ids
    send_ids:   torch.Tensor,   # (E, n_send)             global sender node ids
    pad_ids:    torch.Tensor,   # (E,)                    id of each env's pad node
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a batched dense edge block; redirect inactive edges to the pad node.

    This is the batched analogue of EdgeBlock.make_edges: same mask→pad contract,
    but applied to all E environments in a single set of tensor ops.

    Returns flat tensors of shape (E * n_recv * n_send, ...).
    """
    E, R, Sn, F = edge_feats.shape

    # Broadcast receiver/sender ids and pad_ids to an (E, R, Sn) grid
    recv_grid = recv_ids[:, :,    None].expand(E, R, Sn)
    send_grid = send_ids[:, None, :   ].expand(E, R, Sn)
    pad_grid  = pad_ids[:,  None, None].expand(E, R, Sn)

    # Inactive edges → pad node; active edges → their actual receiver/sender
    recv_flat  = torch.where(edge_mask, recv_grid, pad_grid).reshape(-1)
    send_flat  = torch.where(edge_mask, send_grid, pad_grid).reshape(-1)
    feats_flat = edge_feats.reshape(E * R * Sn, F)

    return feats_flat, recv_flat, send_flat