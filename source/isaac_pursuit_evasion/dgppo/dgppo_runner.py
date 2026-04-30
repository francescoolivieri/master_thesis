import torch
from .dgppo_models import DGPPOValueNet, DGPPOPolicy
from .dgppo_agent import DGPPOAgent

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
        state_dim = base_env.state_space.shape[0]  # physical state (for CBF)
        action_dim = env.action_space.shape[0]  # per agent action size
       
        edge_dim = 4   # 3-D relative position + distance | DEPENDS ON GRAPH BUILDER
        node_dim = base_env.observation_space.shape[0] # per agent obs -> GNN node features
        

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
        from skrl.trainers.torch import SequentialTrainer
        self.trainer = SequentialTrainer(env=env, agents=self.agent, cfg=trainer_cfg)
        self.agent.init(trainer_cfg=trainer_cfg)

    def run(self) -> None:
        self.trainer.train()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        