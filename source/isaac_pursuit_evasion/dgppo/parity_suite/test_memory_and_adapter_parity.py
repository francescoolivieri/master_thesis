from __future__ import annotations

import pytest

from .parity_test_utils import (
    assert_parity_close,
    importorskip,
    load_update_fixture_for_num_envs,
)

torch = importorskip("torch")

from dgppo.utils import (
    NUM_TYPE_INDICATORS,
    build_graph_data,
    extract_graph_states_from_flat_obs,
)


@pytest.mark.parametrize("num_envs", [2, 6])
def test_rollout_memory_layout_matches_jax_update_fixture(num_envs: int) -> None:
    importorskip("skrl.memories.torch")
    from dgppo.dgppo_memory import DGPPORolloutMemory

    fixture = load_update_fixture_for_num_envs(num_envs)
    agent_state = fixture.tensor("inputs/rollout/graph/8/agent")
    goal_state = fixture.tensor("inputs/rollout/graph/8/goal")
    actions = fixture.tensor("inputs/rollout/actions")
    logp = fixture.tensor("inputs/rollout/log_pis")
    rewards = fixture.tensor("inputs/rollout/rewards")
    costs = fixture.tensor("inputs/rollout/costs")
    values_l = fixture.tensor("checkpoints/update/value/bT_Vl")
    values_h = fixture.tensor("checkpoints/update/value/bTah_Vh")

    B, T, A, S = agent_state.shape
    action_dim = actions.shape[-1]
    n_cost = costs.shape[-1]
    obs_state = torch.zeros(B, 1, S)

    memory = DGPPORolloutMemory(
        rollout_length=T,
        num_det_envs=B,
        num_stc_envs=B,
        n_agents=A,
        n_obs=1,
        state_dim=S,
        action_dim=action_dim,
        n_constraints=n_cost,
        device=torch.device("cpu"),
        use_rnn=False,
    )

    for t in range(T):
        memory.add(
            stc_agent_state=agent_state[:, t],
            stc_goal_state=goal_state[:, t],
            stc_obs_state=obs_state,
            stc_action=actions[:, t],
            stc_log_prob=logp[:, t],
            stc_reward=rewards[:, t],
            stc_cost=costs[:, t],
            stc_value_l=values_l[:, t],
            stc_value_h=values_h[:, t],
            det_agent_state=agent_state[:, t],
            det_goal_state=goal_state[:, t],
            det_obs_state=obs_state,
            det_action=actions[:, t],
            det_log_prob=logp[:, t],
            det_reward=rewards[:, t],
            det_cost=costs[:, t],
            det_value_l=values_l[:, t],
            det_value_h=values_h[:, t],
        )

    memory.set_final_values(
        "stc",
        fixture.tensor("checkpoints/update/value/bTp1_Vl")[:, -1],
        fixture.tensor("checkpoints/update/value/bTp1ah_Vh")[:, -1],
    )
    view = memory.as_bTah_view("stc")

    assert_parity_close(view["bT_l"], -rewards, stage=f"num_envs={num_envs}/memory", tensor_name="bT_l")
    assert_parity_close(view["bTah_hs"], costs, stage=f"num_envs={num_envs}/memory", tensor_name="bTah_hs")
    assert_parity_close(view["bT_Vl"], values_l, stage=f"num_envs={num_envs}/memory", tensor_name="bT_Vl")
    assert_parity_close(
        view["bTp1_Vl"],
        fixture.arrays["checkpoints/update/value/bTp1_Vl"],
        stage=f"num_envs={num_envs}/memory",
        tensor_name="bTp1_Vl",
    )
    assert_parity_close(view["bTah_Vh"], values_h, stage=f"num_envs={num_envs}/memory", tensor_name="bTah_Vh")
    assert_parity_close(
        view["bTp1ah_Vh"],
        fixture.arrays["checkpoints/update/value/bTp1ah_Vh"],
        stage=f"num_envs={num_envs}/memory",
        tensor_name="bTp1ah_Vh",
    )
    assert not view["bT_terminated"].any()
    assert not view["bT_truncated"].any()


def test_rollout_memory_stores_split_done_masks() -> None:
    importorskip("skrl.memories.torch")
    from dgppo.dgppo_memory import DGPPORolloutMemory

    T, B, A, S, O, Da, NH = 2, 3, 1, 4, 2, 2, 2
    memory = DGPPORolloutMemory(
        rollout_length=T,
        num_det_envs=B,
        num_stc_envs=B,
        n_agents=A,
        n_obs=O,
        state_dim=S,
        action_dim=Da,
        n_constraints=NH,
        device=torch.device("cpu"),
        use_rnn=False,
    )

    zeros_agent = torch.zeros(B, A, S)
    zeros_obs = torch.zeros(B, O, S)
    zeros_action = torch.zeros(B, A, Da)
    zeros_cost = torch.zeros(B, A, NH)
    zeros_value_h = torch.zeros(B, A, NH)
    zeros_env = torch.zeros(B)
    zeros_logp = torch.zeros(B, A)

    stc_terminated = [torch.tensor([False, True, False]), torch.tensor([True, False, False])]
    stc_truncated = [torch.tensor([False, False, True]), torch.tensor([False, True, False])]
    det_terminated = [torch.tensor([True, False, False]), torch.tensor([False, False, True])]
    det_truncated = [torch.tensor([False, True, False]), torch.tensor([True, False, False])]

    for t in range(T):
        memory.add(
            stc_agent_state=zeros_agent,
            stc_goal_state=zeros_agent,
            stc_obs_state=zeros_obs,
            stc_action=zeros_action,
            stc_log_prob=zeros_logp,
            stc_reward=zeros_env,
            stc_cost=zeros_cost,
            stc_value_l=zeros_env,
            stc_value_h=zeros_value_h,
            stc_terminated=stc_terminated[t],
            stc_truncated=stc_truncated[t],
            det_agent_state=zeros_agent,
            det_goal_state=zeros_agent,
            det_obs_state=zeros_obs,
            det_action=zeros_action,
            det_log_prob=zeros_logp,
            det_reward=zeros_env,
            det_cost=zeros_cost,
            det_value_l=zeros_env,
            det_value_h=zeros_value_h,
            det_terminated=det_terminated[t],
            det_truncated=det_truncated[t],
        )

    stc_view = memory.as_bTah_view("stc")
    det_view = memory.as_bTah_view("det")

    assert torch.equal(stc_view["bT_terminated"], torch.stack(stc_terminated, dim=1))
    assert torch.equal(stc_view["bT_truncated"], torch.stack(stc_truncated, dim=1))
    assert torch.equal(det_view["bT_terminated"], torch.stack(det_terminated, dim=1))
    assert torch.equal(det_view["bT_truncated"], torch.stack(det_truncated, dim=1))
    assert torch.equal(stc_view["bT_done"], stc_view["bT_terminated"] | stc_view["bT_truncated"])


def test_rollout_memory_stores_policy_and_vl_rnn_carries() -> None:
    importorskip("skrl.memories.torch")
    from dgppo.dgppo_memory import DGPPORolloutMemory

    T, B, A, S, O, Da, NH = 2, 3, 2, 4, 1, 2, 1
    L, C, H = 1, 1, 5
    memory = DGPPORolloutMemory(
        rollout_length=T,
        num_det_envs=B,
        num_stc_envs=B,
        n_agents=A,
        n_obs=O,
        state_dim=S,
        action_dim=Da,
        n_constraints=NH,
        device=torch.device("cpu"),
        use_rnn=True,
        use_vl_rnn=True,
        rnn_layers=L,
        rnn_hidden=H,
        rnn_cell="gru",
    )

    zeros_agent = torch.zeros(B, A, S)
    zeros_obs = torch.zeros(B, O, S)
    zeros_action = torch.zeros(B, A, Da)
    zeros_cost = torch.zeros(B, A, NH)
    zeros_value_h = torch.zeros(B, A, NH)
    zeros_env = torch.zeros(B)
    zeros_logp = torch.zeros(B, A)

    stc_policy_states = []
    stc_vl_states = []
    for t in range(T):
        policy_state = torch.arange(L * B * A * C * H, dtype=torch.float32).reshape(L, B * A, C, H) + 100 * t
        vl_state = torch.arange(L * B * C * H, dtype=torch.float32).reshape(L, B, C, H) + 1000 * t
        stc_policy_states.append(policy_state.reshape(L, B, A, C, H).permute(1, 0, 2, 3, 4))
        stc_vl_states.append(vl_state.permute(1, 0, 2, 3))
        memory.add(
            stc_agent_state=zeros_agent,
            stc_goal_state=zeros_agent,
            stc_obs_state=zeros_obs,
            stc_action=zeros_action,
            stc_log_prob=zeros_logp,
            stc_reward=zeros_env,
            stc_cost=zeros_cost,
            stc_value_l=zeros_env,
            stc_value_h=zeros_value_h,
            det_agent_state=zeros_agent,
            det_goal_state=zeros_agent,
            det_obs_state=zeros_obs,
            det_action=zeros_action,
            det_log_prob=zeros_logp,
            det_reward=zeros_env,
            det_cost=zeros_cost,
            det_value_l=zeros_env,
            det_value_h=zeros_value_h,
            stc_rnn_state=policy_state,
            det_rnn_state=policy_state,
            stc_vl_rnn_state=vl_state,
            det_vl_rnn_state=vl_state,
        )

    view = memory.as_bTah_view("stc")
    assert torch.equal(view["bTa_rnn_states"], torch.stack(stc_policy_states, dim=1))
    assert torch.equal(view["bT_vl_rnn_states"], torch.stack(stc_vl_states, dim=1))


@pytest.mark.parametrize("num_envs", [2, 6])
def test_flat_observation_adapter_and_graph_shapes(num_envs: int) -> None:
    n_agents = 3
    n_obstacles = 2
    state_dim = 4
    E = num_envs

    agent = torch.arange(E * n_agents * state_dim, dtype=torch.float32).reshape(E, n_agents, state_dim)
    goal_pos = torch.arange(E * n_agents * 3, dtype=torch.float32).reshape(E, n_agents, 3) / 10.0
    obstacles = torch.arange(E * n_obstacles * 2, dtype=torch.float32).reshape(E, n_obstacles, 2) / 20.0
    observations = torch.cat([agent.reshape(E, -1), goal_pos.reshape(E, -1), obstacles.reshape(E, -1)], dim=-1)

    layout = {
        "state_dim": state_dim,
        "n_agents": n_agents,
        "n_obstacles": n_obstacles,
        "agent_end": n_agents * state_dim,
        "goal_end": n_agents * state_dim + n_agents * 3,
        "obstacles_end": n_agents * state_dim + n_agents * 3 + n_obstacles * 2,
    }

    agent_state, goal_state, obs_state = extract_graph_states_from_flat_obs(
        observations,
        layout,
        n_agents=n_agents,
    )
    graph = build_graph_data(agent_state, goal_state, obs_state, obs_radius=1.0)

    assert_parity_close(agent_state, agent, stage="adapter", tensor_name="agent_state")
    assert_parity_close(goal_state[..., :3], goal_pos, stage="adapter", tensor_name="goal_position")
    assert_parity_close(obs_state[..., :2], obstacles, stage="adapter", tensor_name="obstacle_xy")
    assert graph.n_graphs == num_envs
    assert graph.nodes.shape == (num_envs * (n_agents * 2 + n_obstacles + 1), state_dim + 3)
    assert graph.edges.shape == (num_envs * (n_agents * n_agents * 2 + n_agents * n_obstacles), state_dim + 3)


def test_gnn_policy_and_value_shapes_for_rollout_and_chunk_graphs() -> None:
    importorskip("torch_geometric")

    from dgppo.dgppo_models import DGPPOPolicy, DGPPOValueNet
    from dgppo.update_helpers import build_rollout_graph, rollout_graph_chunks

    B, T, A, O, S = 2, 4, 3, 2, 6
    action_dim = 4
    n_constraints = 5
    gnn_out_dim = 8
    rnn_hidden = 7

    agent_state = torch.randn(B, T, A, S)
    goal_state = torch.randn(B, T, A, S)
    obs_state = torch.randn(B, T, O, S)
    graph = build_rollout_graph(
        view={
            "bTa_agent_state": agent_state,
            "bTa_goal_state": goal_state,
            "bTo_obs_state": obs_state,
        },
        obs_radius=10.0,
    )
    chunk_ids = torch.tensor([[0, 1], [2, 3]])
    chunk_graph = rollout_graph_chunks(graph, chunk_ids=chunk_ids, T=T, B=B)

    node_dim = S + NUM_TYPE_INDICATORS
    common_kwargs = {
        "node_dim": node_dim,
        "edge_dim": node_dim,
        "gnn_out_dim": gnn_out_dim,
        "gnn_msg_dim": 6,
        "gnn_heads": 2,
        "mlp_hid": (9,),
        "use_rnn": True,
        "rnn_hidden": rnn_hidden,
    }
    policy = DGPPOPolicy(
        **common_kwargs,
        action_dim=action_dim,
        gnn_layers=2,
    )
    Vl = DGPPOValueNet(
        **common_kwargs,
        gnn_layers=2,
        n_out=1,
        decompose=False,
    )
    Vh = DGPPOValueNet(
        **common_kwargs,
        gnn_layers=1,
        n_out=n_constraints,
        decompose=True,
    )

    assert policy.gnn(graph, node_type=0, n_type=A).shape == (B * T, A, gnn_out_dim)
    assert policy.gnn(chunk_graph, node_type=0, n_type=A).shape == (
        B,
        chunk_ids.shape[0],
        chunk_ids.shape[1],
        A,
        gnn_out_dim,
    )

    dist, policy_state = policy.distribution(graph, policy.initialize_carry(B * T * A), A)
    assert dist.mean.shape == (B * T, A, action_dim)
    assert dist.std.shape == (B * T, A, action_dim)
    assert policy_state.shape == (1, B * T * A, 1, rnn_hidden)

    vl, vl_state = Vl(graph, Vl.rnn.initialize_carry(B * T), A)
    vh, vh_state = Vh(graph, policy.initialize_carry(B * T * A), A)
    assert vl.shape == (B * T, 1)
    assert vl_state.shape == (1, B * T, 1, rnn_hidden)
    assert vh.shape == (B * T, A, n_constraints)
    assert vh_state.shape == (1, B * T * A, 1, rnn_hidden)
