"""
Thin runner shim that mirrors :class:`skrl.utils.runner.torch.Runner`'s
public surface so [master_thesis/scripts/skrl/train.py](master_thesis/scripts/skrl/train.py)
can drive DGPPO without any structural changes.

The existing train script only ever touches:
    runner.agent
    runner.trainer
    runner.run()
...plus the hooks it monkey-patches on ``agent`` (``post_interaction``,
``write_checkpoint``). Re-implementing these three attributes is enough.

The shim is also responsible for:
    * instantiating :class:`DGPPOPolicy` / :class:`DGPPOCritic` with the
      dimensions exposed by the env,
    * handing a :class:`EnvAccessor` to the agent,
    * wiring a :class:`skrl.trainers.torch.SequentialTrainer` over the wrapped
      skrl vec-env (so the trainer ticks ``pre_interaction`` / ``act`` /
      ``record_transition`` / ``post_interaction`` for us).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .dgppo_agent import DGPPOAgent, EnvAccessor
from .dgppo_models import DGPPOCritic, DGPPOPolicy
from .graph_builder import GraphLayout, NUM_TYPE_INDICATORS


class DGPPORunner:
    """skrl Runner-compatible driver for DGPPO.

    The constructor takes the ``env`` already wrapped by
    :class:`isaaclab_rl.skrl.SkrlVecEnvWrapper` and the hydra-loaded
    ``agent_cfg`` dict (same shape as :file:`skrl_dgppo_cfg.yaml`).
    """

    def __init__(self, env, *args) -> None:
        if len(args) == 1:
            det_env = env
            agent_cfg = args[0]
        elif len(args) == 2:
            det_env, agent_cfg = args
        else:
            raise TypeError("DGPPORunner expects (env, agent_cfg) or (env, det_env, agent_cfg)")
        self.env = env
        self.det_env = det_env
        self.cfg = dict(agent_cfg)

        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        det_base_env = det_env.unwrapped if hasattr(det_env, "unwrapped") else det_env
        env_accessor = EnvAccessor(base_env)
        det_env_accessor = EnvAccessor(det_base_env)
        device = torch.device(getattr(env, "device", getattr(base_env, "device", "cpu")))

        # Shapes come from the env (the agent has no prior knowledge of the env).
        n_agents = env_accessor.n_agents
        n_obs = env_accessor.n_obs
        n_constraints = env_accessor.n_constraints
        state_dim = env_accessor.state_dim
        action_dim = int(env.single_action_space.shape[-1]) if hasattr(env, "single_action_space") else int(env.action_space.shape[-1])

        layout = GraphLayout(
            n_agents=n_agents, n_goals=n_agents, n_obs=n_obs, state_dim=state_dim
        )
        node_dim = layout.node_dim
        edge_dim = layout.edge_dim

        # Build nets from config.
        gnn_cfg = self.cfg.get("gnn", {})
        rnn_cfg = self.cfg.get("rnn", {})
        use_rnn = bool(self.cfg.get("use_rnn", False))

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
        )
        critic = DGPPOCritic(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_constraints=n_constraints,
            vl_gnn_layers=int(gnn_cfg.get("vl_layers", 1)),
            vh_gnn_layers=int(gnn_cfg.get("vh_layers", 1)),
            gnn_out_dim=int(gnn_cfg.get("out_dim", 64)),
            gnn_msg_dim=int(gnn_cfg.get("msg_dim", 32)),
            gnn_heads=int(gnn_cfg.get("n_heads", 3)),
            use_rnn=use_rnn,
            rnn_cell=str(rnn_cfg.get("cell", "gru")),
            rnn_hidden=int(rnn_cfg.get("hidden", 64)),
            rnn_layers=int(rnn_cfg.get("layers", 1)),
        )

        agent_hparams = dict(self.cfg.get("agent", {}))
        agent_hparams.setdefault("experiment", self.cfg.get("agent", {}).get("experiment", {}))
        # Promote top-level DGPPO knobs into the agent cfg for convenience.
        for key in ("obs_radius",):
            if key in self.cfg:
                agent_hparams[key] = self.cfg[key]

        self.agent = DGPPOAgent(
            policy=policy,
            critic=critic,
            env_accessor=env_accessor,
            det_env=det_env,
            det_env_accessor=det_env_accessor,
            graph_layout=layout,
            cfg=agent_hparams,
            observation_space=getattr(env, "single_observation_space", getattr(env, "observation_space", None)),
            action_space=getattr(env, "single_action_space", getattr(env, "action_space", None)),
            device=device,
        )

        # Trainer: use skrl's SequentialTrainer for the loop semantics.
        from skrl.trainers.torch import SequentialTrainer  # deferred import (skrl lives in the container)

        trainer_cfg = dict(self.cfg.get("trainer", {}))
        # SequentialTrainer expects ``timesteps`` and optional logging keys.
        self.trainer = SequentialTrainer(env=env, agents=self.agent, cfg=trainer_cfg)
        # ``init`` hands the trainer config back to the agent so it can size
        # checkpoint/write intervals.
        self.agent.init(trainer_cfg=trainer_cfg)

    def run(self) -> None:
        self.trainer.train()

    def close(self) -> None:
        if self.det_env is self.env:
            return
        try:
            self.det_env.close()
        except Exception:
            pass
