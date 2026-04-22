"""
Env-agnostic helper that turns per-env agent/goal/obstacle tensors into a
batched :class:`GraphData` compatible with :class:`GraphTransformerGNN`.

All nodes from every parallel environment are concatenated along the leading
axis (same flat layout used by the JAX reference in
``dgppo/env/mpe/base.py::MPE.get_graph``):

    [env0_agents | env0_goals | env0_obs | env0_pad] ++ [env1_agents | ...]

Edges are built per-env as dense :class:`EdgeBlock`s then flattened into a
single (sender, receiver, feature) list. Masked-out edges are routed to the
env's padding node so that fixed-shape tensors can be used even when
connectivity varies.

Node type convention (kept in ``graph.node_types``):
    0  -> agent
    1  -> goal
    2  -> obstacle
   -1  -> padding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .gnn import EdgeBlock, GraphData


AGENT_TYPE = 0
GOAL_TYPE = 1
OBS_TYPE = 2
PAD_TYPE = -1

# Indicator one-hot offsets appended to every node feature vector.
# Layout: [state_dim features | agent_flag | goal_flag | obs_flag]
NUM_TYPE_INDICATORS = 3


@dataclass
class GraphLayout:
    """Shapes of the graph produced by :func:`build_graph_data`.

    Exposed so that callers (policy / critic nets, config) can size the GNN
    without re-deriving the numbers.
    """

    n_agents: int
    n_goals: int
    n_obs: int
    state_dim: int

    @property
    def node_dim(self) -> int:
        return self.state_dim + NUM_TYPE_INDICATORS

    @property
    def edge_dim(self) -> int:
        # The GNN in gnn.py uses the same ``in_dim`` for both node and edge
        # linears in each layer, so edges are padded to ``node_dim``. The
        # leading ``state_dim`` entries carry the state-difference features;
        # the remaining ``NUM_TYPE_INDICATORS`` slots are zero-padded.
        return self.node_dim

    @property
    def nodes_per_env(self) -> int:
        # +1 for the padding node per sub-graph.
        return self.n_agents + self.n_goals + self.n_obs + 1


def _make_node_features(
    agent_state: torch.Tensor,   # (E, A, S)
    goal_state: torch.Tensor,    # (E, G, S)
    obs_state: torch.Tensor,     # (E, O, S)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble per-env node feature, state, and type tensors (with padding)."""
    E, A, S = agent_state.shape
    G = goal_state.shape[1]
    O = obs_state.shape[1]
    device = agent_state.device
    dtype = agent_state.dtype

    # States per node: raw state vectors for agents/goals/obstacles, and a
    # -1 sentinel vector for the pad node (mirrors ``GetGraph.to_padded``).
    state_pad = torch.full((E, 1, S), -1.0, dtype=dtype, device=device)
    states = torch.cat([agent_state, goal_state, obs_state, state_pad], dim=1)  # (E, N, S)

    # Type-indicator one-hots appended to the state.
    type_ids = torch.empty((E, A + G + O + 1), dtype=torch.long, device=device)
    type_ids[:, :A] = AGENT_TYPE
    type_ids[:, A:A + G] = GOAL_TYPE
    type_ids[:, A + G:A + G + O] = OBS_TYPE
    type_ids[:, A + G + O:] = PAD_TYPE

    indicator = torch.zeros((E, A + G + O + 1, NUM_TYPE_INDICATORS), dtype=dtype, device=device)
    indicator[:, :A, 0] = 1.0
    indicator[:, A:A + G, 1] = 1.0
    indicator[:, A + G:A + G + O, 2] = 1.0
    # Padding row stays all zero; its state is -1 and its type id is -1.

    nodes = torch.cat([states, indicator], dim=-1)  # (E, N, node_dim)
    return nodes, states, type_ids


def _edge_blocks_for_env(
    agent_state: torch.Tensor,        # (A, S)
    goal_state: torch.Tensor,         # (G, S) with G == A (one goal per agent)
    obs_state: torch.Tensor,          # (O, S)
    *,
    obs_radius: float,
    agent_ids: torch.Tensor,          # (A,)
    goal_ids: torch.Tensor,           # (G,)
    obs_ids: torch.Tensor,            # (O,)
    self_loops_in_agent_block: bool = False,
) -> list[EdgeBlock]:
    """Build the three canonical edge blocks (A-A, A-G, A-O) for one env.

    Mirrors ``dgppo/env/mpe/mpe_target.py::edge_blocks``: distance-gated
    connectivity for A-A and A-O, and a one-to-one diagonal matching for A-G.
    """
    A = agent_state.shape[0]
    O = obs_state.shape[0]
    device = agent_state.device

    # A - A
    a_pos = agent_state[:, :2]
    dist_aa = torch.cdist(a_pos, a_pos)
    aa_mask = dist_aa < obs_radius
    if not self_loops_in_agent_block:
        aa_mask = aa_mask & ~torch.eye(A, dtype=torch.bool, device=device)
    aa_feats = agent_state[:, None, :] - agent_state[None, :, :]  # (A, A, S)
    aa = EdgeBlock(aa_feats, aa_mask, agent_ids, agent_ids)

    # A - G (identity matching along diagonal)
    ag_feats = torch.zeros((A, A, agent_state.shape[1]), dtype=agent_state.dtype, device=device)
    diag = torch.arange(A, device=device)
    ag_feats[diag, diag, :] = agent_state - goal_state
    ag_mask = torch.eye(A, dtype=torch.bool, device=device)
    ag = EdgeBlock(ag_feats, ag_mask, agent_ids, goal_ids)

    # A - O
    if O > 0:
        o_pos = obs_state[:, :2]
        dist_ao = torch.cdist(a_pos, o_pos)
        ao_mask = dist_ao < obs_radius
        ao_feats = agent_state[:, None, :] - obs_state[None, :, :]
        ao = EdgeBlock(ao_feats, ao_mask, agent_ids, obs_ids)
        return [aa, ag, ao]
    return [aa, ag]


def build_graph_data(
    agent_state: torch.Tensor,
    goal_state: torch.Tensor,
    obs_state: Optional[torch.Tensor],
    *,
    obs_radius: float,
) -> GraphData:
    """Build a batched :class:`GraphData` for ``E`` parallel environments.

    Args:
        agent_state: ``(E, A, S)`` physical state per agent.
        goal_state:  ``(E, G, S)`` with ``G == A`` (one goal per agent).
        obs_state:   ``(E, O, S)`` or ``None`` if the env exposes no obstacles.
        obs_radius:  proximity threshold for A-A and A-O edges.

    Returns:
        A flat batched :class:`GraphData`. Node ordering per sub-graph is
        ``[agents | goals | obstacles | pad]`` and sub-graphs are concatenated
        along the leading node axis.
    """
    assert agent_state.dim() == 3
    assert goal_state.dim() == 3
    assert agent_state.shape[0] == goal_state.shape[0]
    assert agent_state.shape[1] == goal_state.shape[1], "need one goal per agent"
    assert agent_state.shape[2] == goal_state.shape[2]

    E, A, S = agent_state.shape
    if obs_state is None:
        obs_state = agent_state.new_zeros((E, 0, S))
    assert obs_state.dim() == 3 and obs_state.shape[0] == E and obs_state.shape[2] == S
    O = obs_state.shape[1]
    device = agent_state.device

    nodes_pe, states_pe, types_pe = _make_node_features(agent_state, goal_state, obs_state)

    N_per = A + A + O + 1  # nodes per sub-graph (includes padding)
    # Global node ids for each env are contiguous; pad is the last local id.
    # Per-env local id helpers, then offset by env * N_per when flattening.
    local_agent_ids = torch.arange(A, device=device)
    local_goal_ids = torch.arange(A, device=device) + A
    local_obs_ids = torch.arange(O, device=device) + 2 * A
    local_pad_id = N_per - 1

    edge_feats_all = []
    recv_all = []
    send_all = []
    edges_per_env = []

    for e in range(E):
        offset = e * N_per
        agent_ids = local_agent_ids + offset
        goal_ids = local_goal_ids + offset
        obs_ids = local_obs_ids + offset
        pad_id = local_pad_id + offset

        blocks = _edge_blocks_for_env(
            agent_state[e],
            goal_state[e],
            obs_state[e],
            obs_radius=obs_radius,
            agent_ids=agent_ids,
            goal_ids=goal_ids,
            obs_ids=obs_ids,
        )
        n_e = 0
        for block in blocks:
            feats, recvs, sends = block.make_edges(pad_id=pad_id)
            edge_feats_all.append(feats)
            recv_all.append(recvs)
            send_all.append(sends)
            n_e += feats.shape[0]
        edges_per_env.append(n_e)

    nodes_flat = nodes_pe.reshape(E * N_per, -1)
    states_flat = states_pe.reshape(E * N_per, -1)
    types_flat = types_pe.reshape(E * N_per)

    node_dim = S + NUM_TYPE_INDICATORS
    if edge_feats_all:
        edges_flat = torch.cat(edge_feats_all, dim=0)
        # Pad edges with zeros so their feature dim matches ``node_dim``.
        pad = edges_flat.new_zeros(edges_flat.shape[0], NUM_TYPE_INDICATORS)
        edges_flat = torch.cat((edges_flat, pad), dim=-1)
        recv_flat = torch.cat(recv_all, dim=0)
        send_flat = torch.cat(send_all, dim=0)
    else:
        edges_flat = agent_state.new_zeros((0, node_dim))
        recv_flat = torch.zeros(0, dtype=torch.long, device=device)
        send_flat = torch.zeros(0, dtype=torch.long, device=device)

    n_nodes = torch.full((E,), N_per, dtype=torch.long, device=device)
    n_edges = torch.tensor(edges_per_env, dtype=torch.long, device=device)

    return GraphData(
        n_nodes=n_nodes,
        n_edges=n_edges,
        nodes=nodes_flat,
        edges=edges_flat,
        states=states_flat,
        receivers=recv_flat.long(),
        senders=send_flat.long(),
        node_types=types_flat,
    )
