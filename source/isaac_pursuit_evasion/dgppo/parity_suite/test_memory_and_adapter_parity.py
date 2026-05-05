from __future__ import annotations

import pytest

from .parity_test_utils import assert_parity_close, importorskip, load_update_fixture_for_num_envs

torch = importorskip("torch")

from dgppo.utils import build_graph_data, extract_graph_states_from_flat_obs


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
