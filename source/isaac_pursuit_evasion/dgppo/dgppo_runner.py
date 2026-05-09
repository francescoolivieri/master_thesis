import torch
from .dgppo_models import DGPPOValueNet, DGPPOPolicy
from .dgppo_agent import DGPPOAgent, DGPPOAgentCfg
from .utils import NUM_TYPE_INDICATORS


def _as_int_tuple(values, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    if isinstance(values, int):
        return (int(values),)
    return tuple(int(v) for v in values)

class DGPPORunner:

    def __init__(self, env, cfg: dict):

        self.env = env
        agent_cfg_data = cfg.get("agent", cfg)
        agent_cfg = DGPPOAgentCfg.from_dict(agent_cfg_data)
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
        action_dim = env.action_space.shape[0]  # per agent action size
        agent_cfg.num_envs = int(n_envs)
        agent_cfg._raw["num_envs"] = int(n_envs)
       
        layout = base_env.graph_obs_layout
        graph_state_dim = int(layout["state_dim"])
        node_dim = graph_state_dim + NUM_TYPE_INDICATORS
        edge_dim = graph_state_dim + NUM_TYPE_INDICATORS
        

        # - Policy
        gnn_cfg = agent_cfg.gnn
        rnn_cfg = agent_cfg.rnn
        model_cfg = agent_cfg.model
        use_rnn = bool(agent_cfg.use_rnn)
        policy = DGPPOPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            gnn_layers=int(gnn_cfg.get("policy_layers", 1)),
            gnn_out_dim=int(gnn_cfg.get("policy_out_dim", gnn_cfg.get("out_dim", 64))),
            gnn_msg_dim=int(gnn_cfg.get("msg_dim", 32)),
            gnn_heads=int(gnn_cfg.get("n_heads", 3)),
            mlp_hid=_as_int_tuple(model_cfg.get("policy_mlp_hid"), (128, 64)),
            scale_hid=int(model_cfg.get("scale_hid", 64)),
            scale_final=float(model_cfg.get("scale_final", 0.01)),
            std_dev_init=float(model_cfg.get("std_dev_init", 0.5)),
            std_dev_min=float(model_cfg.get("std_dev_min", 1e-5)),
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
            gnn_out_dim=int(gnn_cfg.get("critic_out_dim", gnn_cfg.get("out_dim", 64))),
            gnn_msg_dim=int(gnn_cfg.get("msg_dim", 32)),
            gnn_heads=int(gnn_cfg.get("n_heads", 3)),
            mlp_hid=_as_int_tuple(model_cfg.get("critic_mlp_hid"), (128, 64)),
            use_rnn=use_rnn,
            rnn_cell=str(rnn_cfg.get("cell", "gru")),
            rnn_hidden=int(rnn_cfg.get("hidden", 64)),
            rnn_layers=int(rnn_cfg.get("layers", 1)),
            device=device,
        )
        Vl = DGPPOValueNet(
            **critic_kwargs,
            gnn_layers=int(gnn_cfg.get("vl_layers", 1)),
            n_out=1,
            decompose=False,
        )
        Vh = DGPPOValueNet(
            **critic_kwargs,
            gnn_layers=int(gnn_cfg.get("vh_layers", 1)),
            n_out=n_constraints,
            decompose=True,
        )

        # - Agent
        self.agent = DGPPOAgent(
            policy=policy,
            Vl=Vl,
            Vh=Vh,
            env=env,
            cfg=agent_cfg,
            observation_space=base_env.observation_space,
            state_space=base_env.state_space,
            action_space=base_env.action_space,
            device=device,
        )

        # - Trainer 
        from skrl.trainers.torch import SequentialTrainer
        self.trainer = SequentialTrainer(env=env, agents=self.agent, cfg=trainer_cfg)

    def run(self) -> None:
        self.trainer.train()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        
