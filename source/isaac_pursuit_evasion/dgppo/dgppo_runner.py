import torch
import sys
import time
import tqdm
from collections import deque
from skrl.utils import ScopedTimer
from .dgppo_models import DGPPOValueNet, DGPPOPolicy
from .dgppo_agent import DGPPOAgent


class DGPPORolloutTrainer:
    """Sequential trainer with rollout-level progress for a more honest ETA."""

    def __init__(self, env, agents, cfg: dict):
        from skrl.trainers.torch import SequentialTrainer

        self._delegate = SequentialTrainer(env=env, agents=agents, cfg=cfg)
        self.env = self._delegate.env
        self.agents = self._delegate.agents
        self.cfg = self._delegate.cfg
        self.num_simultaneous_agents = self._delegate.num_simultaneous_agents

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def train(self) -> None:
        if self.num_simultaneous_agents != 1:
            self._delegate.train()
            return

        self.agents.enable_training_mode(True)
        observations, infos = self.env.reset()
        states = self.env.state()

        total_timesteps = int(self.cfg.timesteps)
        rollout = max(1, int(getattr(self.agents, "rollouts", 1)))
        completed_steps = 0
        cycle_started_at = time.perf_counter()
        cycle_history = deque(maxlen=8)

        progress = tqdm.tqdm(
            total=total_timesteps,
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
            unit="step",
            smoothing=0.0,
            dynamic_ncols=True,
            desc="DGPPO",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        )
        progress.set_postfix_str(f"eta=measuring step=0/{total_timesteps}")

        try:
            for timestep in range(total_timesteps):
                self.agents.pre_interaction(timestep=timestep, timesteps=total_timesteps)

                with torch.no_grad():
                    with ScopedTimer() as timer:
                        actions, outputs = self.agents.act(
                            observations, states, timestep=timestep, timesteps=total_timesteps
                        )
                        self.agents.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)

                    with ScopedTimer() as timer:
                        next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                        next_states = self.env.state()
                        self.agents.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                    if not self.cfg.headless and not timestep % self.cfg.render_interval:
                        self.env.render()

                    self.agents.record_transition(
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
                        timesteps=total_timesteps,
                    )

                    if self.cfg.environment_info in infos:
                        for key, value in infos[self.cfg.environment_info].items():
                            if isinstance(value, torch.Tensor) and value.numel() == 1:
                                self.agents.track_data(key if "/" in key else f"Info / {key}", value.item())

                self.agents.post_interaction(timestep=timestep, timesteps=total_timesteps)

                if self.env.num_envs > 1:
                    observations = next_observations
                    states = next_states
                else:
                    should_reset = terminated.any() or truncated.any()
                    if should_reset:
                        with torch.no_grad():
                            observations, infos = self.env.reset()
                            states = self.env.state()
                    else:
                        observations = next_observations
                        states = next_states

                current_step = timestep + 1
                if (current_step % rollout == 0) or (current_step == total_timesteps):
                    cycle_elapsed = time.perf_counter() - cycle_started_at
                    cycle_steps = current_step - completed_steps
                    completed_steps = current_step
                    cycle_started_at = time.perf_counter()
                    cycle_history.append((cycle_steps, cycle_elapsed))

                    progress.update(cycle_steps)
                    history_steps = sum(item[0] for item in cycle_history)
                    history_elapsed = sum(item[1] for item in cycle_history)
                    rolling_seconds_per_step = history_elapsed / max(history_steps, 1)
                    global_seconds_per_step = progress.format_dict["elapsed"] / max(completed_steps, 1)
                    last_seconds_per_step = cycle_elapsed / max(cycle_steps, 1)
                    seconds_per_step = max(
                        rolling_seconds_per_step,
                        global_seconds_per_step,
                        last_seconds_per_step,
                    )
                    remaining = max(total_timesteps - completed_steps, 0)
                    eta = self._format_duration(seconds_per_step * remaining)
                    progress.set_postfix_str(
                        f"eta={eta} cycle={cycle_elapsed:.1f}s step={completed_steps}/{total_timesteps}"
                    )
        finally:
            progress.close()

class DGPPORunner:

    def __init__(self, env, cfg: dict):

        self.env = env
        agent_cfg   = cfg.get("agent", cfg)
        trainer_cfg = cfg.get("trainer", {})
        seed        = cfg.get("seed", agent_cfg.get("seed", None))

        self._agent_cfg   = agent_cfg
        self._trainer_cfg = trainer_cfg

        # set random seed
        from skrl.utils import set_seed
        set_seed(seed)

        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        device = torch.device(env.device)

        n_agents = env.num_agents
        n_envs = env.num_envs
        n_constraints = int(getattr(base_env, "n_constraints", 1))
        layout = base_env.graph_obs_layout
        state_dim = int(layout["state_dim"])  # physical graph state (for CBF)
        action_dim = env.action_space.shape[0]  # per agent action size
       
        edge_dim = state_dim
        node_dim = state_dim + 3  # physical state + [obstacle, goal, agent] indicators
        

        # - Policy
        gnn_cfg = agent_cfg.get("gnn", {})
        rnn_cfg = agent_cfg.get("rnn", {})
        use_rnn = bool(agent_cfg.get("use_rnn", False))
        policy = DGPPOPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            gnn_layers=int(gnn_cfg.get("policy_layers", 1)),
            gnn_out_dim=int(gnn_cfg.get("out_dim", 64)),
            gnn_msg_dim=int(gnn_cfg.get("msg_dim", 32)),
            gnn_heads=int(gnn_cfg.get("n_heads", 3)),
            use_rnn=use_rnn,
            rnn_cell=str(rnn_cfg.get("cell", "gru")),
            rnn_hidden=int(rnn_cfg.get("hidden", 64)),
            rnn_layers=int(rnn_cfg.get("layers", 1)),
            device=device,
        )

        # - Critics
        critic_kwargs = dict(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gnn_layers=int(gnn_cfg.get("critic_layers", 1)),
            gnn_out_dim=int(gnn_cfg.get("out_dim", 64)),
            gnn_msg_dim=int(gnn_cfg.get("msg_dim", 32)),
            gnn_heads=int(gnn_cfg.get("n_heads", 3)),
            mlp_hid=(64, 64),
            use_rnn=use_rnn,
            rnn_cell=str(rnn_cfg.get("cell", "gru")),
            rnn_hidden=int(rnn_cfg.get("hidden", 64)),
            rnn_layers=int(rnn_cfg.get("layers", 1)),
            device=device,
        )
        Vl = DGPPOValueNet(**critic_kwargs, n_out=1,             decompose=False)
        Vh = DGPPOValueNet(**critic_kwargs, n_out=n_constraints, decompose=True)

        # - Agent
        self.agent = DGPPOAgent(
            policy=policy,
            Vl=Vl,
            Vh=Vh,
            env=env,
            cfg=agent_cfg,
            observation_space=base_env.observation_space,
            action_space=base_env.action_space,
            device=device,
        )

        # - Trainer 
        self.trainer = DGPPORolloutTrainer(env=env, agents=self.agent, cfg=trainer_cfg)
        self.agent.init(trainer_cfg=trainer_cfg)

    def run(self) -> None:
        self.trainer.train()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        
