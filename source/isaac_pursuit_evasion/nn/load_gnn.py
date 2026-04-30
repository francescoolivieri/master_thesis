"""
Scratch notes and utilities for integrating the DGPPO GNN with IsaacLab.

Contains a small helper to load the JAX fixture (for introspection /
debugging), plus a block of commented-out reference code that sketches how
to build a :class:`GraphData` from an environment state (agents, goals,
obstacles) and how to step a simple double-integrator dynamics model.

Everything below the helper is intentionally kept as comments: the real
implementation will be written against the IsaacLab env and will replace
these notes.
"""

import numpy as np


def load_data(path: str = "update_fixture.npz") -> dict:
    """Load every array in the JAX fixture into a plain ``{name: ndarray}`` dict.

    Useful for poking at the shapes and contents of a fixture before wiring
    up a new env (e.g. ``load_data()['metadata/config/num_agents']``).
    """
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Reference sketch (keep as notes; not meant to run as-is).
#
# Shows the intended layout for a per-env step function that:
#   1. Decodes agents / goals / obstacles from a GraphData,
#   2. Integrates the dynamics,
#   3. Rebuilds a fresh GraphData around the new state.
#
# In the IsaacLab port, steps 1 and 2 will be handled by the simulator; only
# step 3 (graph construction) has to be ported.
# ---------------------------------------------------------------------------
#
# import torch
# from dataclasses import dataclass
# from typing import NamedTuple, Optional
# from gnn import EdgeBlock, GraphData
#
# AGENT_ID, GOAL_ID, OBS_ID = 0, 1, 2
# dt = 0.03
#
# @dataclass
# class AgentState(NamedTuple):
#     position: torch.Tensor
#     velocity: torch.Tensor
#     orientation: torch.Tensor
#     angular_velocity: torch.Tensor
#     force: torch.Tensor      # applied force  (from policy action)
#     torque: torch.Tensor     # applied torque (from policy action)
#
# def state_lim(state):
#     area_size = 1.5
#     lower_lim = torch.tensor([0., 0., -0.5, -0.5])
#     upper_lim = torch.tensor([area_size, area_size, 0.5, 0.5])
#     return lower_lim, upper_lim
#
# def clip_state(state):
#     lower, upper = state_lim(state)
#     return torch.clip(state, lower, upper)
#
# def action_lim():
#     return -torch.ones(2), torch.ones(2)
#
# def clip_action(action):
#     lower, upper = action_lim()
#     return torch.clip(action, lower, upper)
#
# def agent_step_euler(agent_states, action) -> AgentState:
#     """Forward-Euler integration of a 2D double-integrator."""
#     x_dot = torch.cat([agent_states[:, 2:], action * 10.], dim=1)
#     return AgentState(clip_state(x_dot * dt + agent_states))
#
# def step(graph: GraphData, action):
#     n_agents, n_goals = 10, 1
#     agent_state = graph.get_type_states(type_idx=AGENT_ID, n_states=n_agents)
#     env_state   = graph.get_type_states(type_idx=GOAL_ID,  n_states=n_goals)
#     # obstacles are supplied by the env; in IsaacLab they would come from
#     # the raycaster / lidar sensor rather than an in-graph tensor.
#     action = clip_action(action)
#     next_agent_state = agent_step_euler(agent_state, action)
#     return get_graph_from_env(next_agent_state)
#
# obs_radius = 0.5
#
# def edge_blocks(state) -> list[EdgeBlock]:
#     """Build Agent-Agent, Agent-Goal and Agent-Obstacle edge blocks.
#
#     * Agent-Agent: active if agents are within ``obs_radius``.
#     * Agent-Goal:  1-vs-1 matching (identity mask, one goal per agent).
#     * Agent-Obs:   active if obstacle is within ``obs_radius``.
#     """
#     num_agents = state.num_agents
#
#     # A-A
#     agent_pos = state.agent[:, :2]
#     dist = torch.cdist(agent_pos, agent_pos)
#     edge_feats = state.agent[:, None, :] - state.agent[None, :, :]
#     mask = dist < obs_radius
#     id_agent = torch.arange(num_agents)
#     aa = EdgeBlock(edge_feats, mask, id_agent, id_agent)
#
#     # A-G (one-to-one matching along the diagonal)
#     diff_feats = state.agent - state.goal
#     edge_feats = torch.zeros((num_agents, num_agents, diff_feats.shape[-1]))
#     diag = torch.arange(num_agents)
#     edge_feats[diag, diag, :] = diff_feats
#     mask = torch.eye(num_agents, dtype=torch.bool)
#     id_goal = torch.arange(num_agents) + num_agents
#     ag = EdgeBlock(edge_feats, mask, id_agent, id_goal)
#
#     # A-O
#     obs_pos = state.obs[:, :2]
#     dist = torch.cdist(agent_pos, obs_pos)
#     mask = dist < obs_radius
#     id_obs = torch.arange(state.num_obs) + num_agents + 1
#     state_diff = state.agent[:, None, :] - state.obs[None, :, :]
#     ao = EdgeBlock(state_diff, mask, id_agent, id_obs)
#
#     return [aa, ag, ao]
#
# def get_graph_from_env(env_state) -> GraphData:
#     """Concatenate agent/goal/obstacle states into a padded GraphData.
#
#     Node features are [state | type-one-hot]. A dummy "padding" node is
#     appended so that masked-out edges can be routed to a valid index
#     (see :meth:`EdgeBlock.make_edges`).
#     """
#     ...
