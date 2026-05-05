from __future__ import annotations

import torch
from typing import Optional, Sequence, cast

"""
Variables naming convention :
- b: batch of envs
- T: time steps
- a: agents
- h: costs/constraints

ex. bTah_Vh: [b, T, a, h] for the safety critic
"""

def compute_dec_ocp_gae(
    Tah_hs: torch.Tensor,
    T_l: torch.Tensor,
    Tp1ah_Vh: torch.Tensor,
    Tp1_Vl: torch.Tensor,
    disc_gamma: float,
    gae_lambda: float,
    discount_to_max: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the decomposed-OCP GAE target for both the constraint (high-level)
    value ``Vh`` and the reward (low-level) value ``Vl``.

    Args:
        Tah_hs:    ``[B, T, A, NH]`` constraint costs ``h_i(s_t)``.
        T_l:       ``[B, T]`` scalar reward-to-cost ``l(s_t)``.
        Tp1ah_Vh:  ``[B, T+1, A, NH]`` bootstrap values for ``Vh``.
        Tp1_Vl:    ``[B, T+1]`` bootstrap values for ``Vl``.
        disc_gamma: discount factor.
        gae_lambda: GAE lambda.
        discount_to_max: if True, bootstrap ``Vh`` towards the max over heads
            (standard in DGPPO); otherwise per-head.

    Returns:
        ``Qh`` with shape ``[B, T, A, NH]`` and ``Ql`` with shape ``[B, T]``.
    """
    B, T, A, NH = Tah_hs.shape
    device, dtype = Tah_hs.device, Tah_hs.dtype

    Qh_all = torch.zeros_like(Tah_hs)
    Ql_all = torch.zeros_like(T_l)

    time_ids = torch.arange(T + 1, device=device)

    for b in range(B):
        Tah_hs_b = Tah_hs[b]
        T_l_b = T_l[b]
        # Values along the trajectory and terminal bootstrap.
        Tah_Vh = Tp1ah_Vh[b, :-1]
        T_Vl = Tp1_Vl[b, :-1].unsqueeze(-1).expand(T, A)
        Vh_final = Tp1ah_Vh[b, -1]
        Vl_final = Tp1_Vl[b, -1]

        # Rolling buffer of "next values". Position 0 always holds the most
        # recently computed bootstrap (starts at the terminal value); later
        # positions hold values collected at earlier iterations of the loop.
        next_Vhs_row = torch.zeros((T + 1, A, NH), device=device, dtype=dtype)
        next_Vl_row = torch.zeros((T + 1, A), device=device, dtype=dtype)
        next_Vhs_row[0] = Vh_final
        next_Vl_row[0] = Vl_final

        # GAE weighting coefficients over the rolling buffer.
        gae_coeffs = torch.zeros((T + 1,), device=device, dtype=dtype)
        gae_coeffs[0] = 1.0

        Qs = torch.zeros((T, A, NH + 1), device=device, dtype=dtype)

        # We iterate backwards in time via 't' (accesses the trajectory) while
        # 'step' is simply the iteration counter 0..T-1 used to size the
        # mask and GAE coefficients
        for step, t in enumerate(reversed(range(T))):
            hs = Tah_hs_b[t]
            l = T_l_b[t]
            Vhs = Tah_Vh[t]
            Vl = T_Vl[t]

            # Only the first 'step + 1' buffer positions contain valid data.
            mask = (time_ids <= step).to(dtype)
            mask_h = mask[:, None, None]
            mask_l = mask[:, None]

            # For constraint values: bootstrap towards max-over-heads. Attribute True for DGPPO.
            if discount_to_max:
                h_disc = hs.max(dim=-1).values[:, None]
            else:
                h_disc = hs

            disc_to_h = (1.0 - disc_gamma) * h_disc[None] + disc_gamma * next_Vhs_row
            Vhs_row = mask_h * torch.maximum(hs[None], disc_to_h)
            Vl_row = mask_l * (l + disc_gamma * next_Vl_row)

            # Stack Vh (NH heads) and Vl into one tensor for a single einsum.
            cat_V_row = torch.cat([Vhs_row, Vl_row[:, :, None]], dim=-1)
            Qs[t] = torch.einsum("tah,t->ah", cat_V_row, gae_coeffs)

            # Advance the rolling buffer: the Vhs / Vl of this step becomes
            # the "next value" at distance 'step + 1' for the following iter.
            # Note: we intentionally write into the local 'Vhs_row' / 'Vl_row' (which still carries the mask) and then rebind the buffer to them.
            Vhs_row[step + 1] = Vhs
            Vl_row[step + 1] = Vl
            next_Vhs_row = Vhs_row
            next_Vl_row = Vl_row

            # Shift the GAE coefficients one slot deeper and rewrite the two
            # leading entries to match the current truncation length.
            gae_coeffs = torch.roll(gae_coeffs, shifts=1, dims=0)
            gae_coeffs[0] = gae_lambda ** (step + 1)
            gae_coeffs[1] = (gae_lambda ** step) * (1.0 - gae_lambda)

        Qh_all[b] = Qs[:, :, :NH]
        # Vl is shared across agents, take the agent-0 slice.
        Ql_all[b] = Qs[:, 0, NH]

    return Qh_all, Ql_all


def compute_cbf_advantages(
    bT_Ql: torch.Tensor,
    bT_Vl: torch.Tensor,
    bTah_Vh: torch.Tensor,
    bTp1ah_Vh: torch.Tensor,
    alpha: float,
    cbf_eps: float,
    cbf_weight: float,
    dt: float = 0.03,
    cbf_scale: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """
    CBF-based advantage used by DGPPO.

    Reward advantage is standardized across time, and the constraint term
    (``V_{t+1} - V_t) / dt + alpha * V_t``) enters as an additive penalty
    whenever any head is unsafe.
    """
    
    # calculate cost advantage and normalize
    bT_Al_raw = bT_Ql - bT_Vl    
    bT_Al_norm = (bT_Al_raw - bT_Al_raw.mean(dim=1, keepdim=True)) / (
        bT_Al_raw.std(dim=1, keepdim=True, unbiased=False) + 1e-8
    )
    bTa_Al = bT_Al_norm[:, :, None].expand(-1, -1, bTah_Vh.shape[2])

    # Discrete CBF derivative: (V_{t+1} - V_t) / dt + alpha * V_t.
    bTah_cbf_deriv = (bTp1ah_Vh[:, 1:] - bTah_Vh) / dt + alpha * bTah_Vh
    bTah_Acbf = torch.clamp(bTah_cbf_deriv + cbf_eps, min=0.0)
    
    # check if the safety constraint is satisfied (check for all constraints)
    bTa_is_safe = (bTah_cbf_deriv <= 0).all(dim=-1)

    # Use reward advantage only in safe states; otherwise zero and put CBF term (added below).
    bTa_A = torch.where(bTa_is_safe, bTa_Al, torch.zeros_like(bTa_Al))

    # add CBF term (note that bTah_Acbf is zero when satisfied)
    scale = cbf_weight if cbf_scale is None else cbf_scale
    bTa_A = bTa_A + bTah_Acbf.max(dim=-1).values * scale
    
    # flip for PPO use
    bTa_A = -bTa_A

    return {
        "bT_Al_raw": bT_Al_raw,
        "bT_Al_norm": bT_Al_norm,
        "bTah_cbf_deriv": bTah_cbf_deriv,
        "bTah_Acbf": bTah_Acbf,
        "bTa_is_safe": bTa_is_safe,
        "bTa_A": bTa_A,
    }

def compute_policy_surrogate(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_eps: float
) -> dict[str, torch.Tensor]:
    """Standard clipped PPO surrogate (pessimistic max over clipped/unclipped)."""
    loss_policy1 = -ratio * advantages
    loss_policy2 = -torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss_policy = torch.maximum(loss_policy1, loss_policy2).mean()
    clip_frac = (loss_policy2 > loss_policy1).float().mean()
    return {
        "loss_policy1": loss_policy1,
        "loss_policy2": loss_policy2,
        "loss_policy": loss_policy,
        "clip_frac": clip_frac,
    }
    
    
    
    
from dataclasses import dataclass, replace
from typing import Callable, Generic, NamedTuple, Optional, TypeVar

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as softmax_pyg



@dataclass
class GraphData:
    """
    Flat, padded batched-graph representation used by the GNN.

    All nodes from every sub-graph are concatenated along the first axis; the
    per-sub-graph sizes are tracked in ``n_nodes`` / ``n_edges``. ``receivers``
    and ``senders`` index into the flat ``nodes`` tensor. ``node_types`` marks
    the role of each node (by convention: 0 = agent, 1 = goal, 2 = obstacle,
    -1 = padding).
    """

    n_nodes: torch.Tensor            # (n_graphs,) nodes per sub-graph
    n_edges: torch.Tensor            # (n_graphs,) edges per sub-graph
    nodes: torch.Tensor              # (sum_n_nodes, node_feat_dim)
    edges: Optional[torch.Tensor]    # (sum_n_edges, edge_feat_dim)
    states: torch.Tensor             # per-node physical state, (sum_n_nodes, state_dim)
    receivers: torch.Tensor          # (sum_n_edges,) -- indexes ``nodes``
    senders: torch.Tensor            # (sum_n_edges,) -- indexes ``nodes``
    node_types: torch.Tensor         # (sum_n_nodes,) node type ids, -1 for padding

    @property
    def n_graphs(self) -> int:
        # number of sub-graphs (i.e. parallel simulations) in this batch
        if self.n_nodes.ndim == 0:
            return 1
        return int(self.n_nodes.numel())

    @property
    def batch_shape(self) -> torch.Size:
        # same info as n_graphs but as a shape tuple (convenient for reshape)
        return self.n_nodes.shape

    def get_type_nodes(self, type_idx: int, n_nodes: int) -> torch.Tensor:
        '''
        Get #'n_nodes' nodes of a given type 'type_idx'.
        '''
        # TODO: check dymensionality
        tot_n_feats = self.nodes.shape[1]

        n_is_type = self.node_types == type_idx
        idx = torch.cumsum(n_is_type.long(), dim=0) - 1

        cumulative_n_type = self.n_graphs * n_nodes
        type_feats = self.nodes.new_zeros(cumulative_n_type, tot_n_feats)

        # Note: "n_is_type" masks valid nodes to assign at the correct index
        type_feats[idx[n_is_type]] = self.nodes[n_is_type]

        return type_feats.reshape(self.batch_shape + (n_nodes, tot_n_feats))

    def get_type_states(self, type_idx: int, n_states: int) -> torch.Tensor:
        '''
        Get #'n_states' states of the nodes of a given type 'type_idx'.
        '''
        # TODO: check dymensionality
        assert isinstance(self.states, torch.Tensor)
        tot_n_states = self.states.shape[1]

        n_is_type = self.node_types == type_idx
        idx = torch.cumsum(n_is_type.long(), dim=0) - 1

        cumulative_n_type = self.n_graphs * n_states
        type_feats = self.states.new_zeros(cumulative_n_type, tot_n_states)

        # Note: "n_is_type" masks valid nodes to assign at the correct index
        type_feats[idx[n_is_type]] = self.states[n_is_type]

        return type_feats.reshape(self.batch_shape + (n_states, tot_n_states))
    
    def get_envs_graphs(self, env_ids: torch.Tensor) -> GraphData:
        """
        Get the graphs for the given environment ids.
        """
        edges = None if self.edges is None else self.edges[env_ids]
        return GraphData(
            n_nodes=self.n_nodes[env_ids],
            n_edges=self.n_edges[env_ids],
            nodes=self.nodes[env_ids],
            edges=edges,
            states=self.states[env_ids],
            receivers=self.receivers[env_ids],
            senders=self.senders[env_ids],
            node_types=self.node_types[env_ids],
        )
    

    def _replace(self, **kwargs):
        return replace(self, **kwargs)


def graph_data_slice(graph: GraphData, index: tuple[int, ...] | int) -> GraphData:
    """Return one local sub-graph from a padded, flattened ``GraphData`` batch."""
    if isinstance(index, int):
        index = (index,)
    if graph.n_nodes.ndim == 0:
        if index not in ((), (0,)):
            raise IndexError(f"cannot slice scalar GraphData with index={index}")
        return graph

    if len(index) != graph.n_nodes.ndim:
        raise IndexError(f"GraphData index rank {len(index)} does not match batch rank {graph.n_nodes.ndim}")

    flat_index = 0
    for axis, idx in enumerate(index):
        axis_size = int(graph.n_nodes.shape[axis])
        if idx < 0:
            idx += axis_size
        if idx < 0 or idx >= axis_size:
            raise IndexError(f"GraphData index {index} is out of bounds for batch shape {tuple(graph.n_nodes.shape)}")
        flat_index = flat_index * axis_size + idx

    n_graphs = graph.n_graphs
    padded_nodes = graph.nodes.shape[0] // n_graphs
    padded_edges = graph.receivers.shape[0] // n_graphs
    node_start = flat_index * padded_nodes
    node_stop = node_start + padded_nodes
    edge_start = flat_index * padded_edges
    edge_stop = edge_start + padded_edges

    edges = None if graph.edges is None else graph.edges[edge_start:edge_stop]
    return GraphData(
        n_nodes=graph.n_nodes[index],
        n_edges=graph.n_edges[index],
        nodes=graph.nodes[node_start:node_stop],
        edges=edges,
        states=graph.states[node_start:node_stop],
        receivers=graph.receivers[edge_start:edge_stop] - node_start,
        senders=graph.senders[edge_start:edge_stop] - node_start,
        node_types=graph.node_types[node_start:node_stop],
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Node-type integer IDs.
AGENT_TYPE: int = 0
GOAL_TYPE: int = 1
OBS_TYPE: int = 2
PAD_TYPE: int = -1

# One-hot indicator length; order: [obstacle_bit, goal_bit, agent_bit].
NUM_TYPE_INDICATORS: int = 3


def extract_graph_states_from_flat_obs(
    observations: torch.Tensor,
    layout: dict,
    *,
    n_agents: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode IsaacLab flat policy observations into graph state tensors."""
    E = observations.shape[0]
    S = int(layout["state_dim"])
    A = int(layout.get("n_agents", n_agents))
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


def build_graph_data(
    agent_state: torch.Tensor,
    goal_state: torch.Tensor,
    obs_state: torch.Tensor | None,
    *,
    obs_radius: float,
) -> GraphData:
    """Build a batched ``GraphData`` with jraph-style concatenated sub-graphs."""
    assert agent_state.dim() == 3 and goal_state.dim() == 3
    assert agent_state.shape[0] == goal_state.shape[0], "E must match"
    assert agent_state.shape[1] == goal_state.shape[1], "need one goal per agent"
    assert agent_state.shape[2] == goal_state.shape[2], "state dim must match"

    E, A, S = agent_state.shape
    device = agent_state.device

    if obs_state is None:
        obs_state = agent_state.new_zeros(E, 0, S)
    assert obs_state.shape[0] == E and obs_state.shape[2] == S
    O = obs_state.shape[1]

    N_per = A + A + O + 1
    nodes, states, node_types = _make_node_features(agent_state, goal_state, obs_state)

    nodes_flat = nodes.reshape(E * N_per, -1)
    states_flat = states.reshape(E * N_per, -1)
    node_types_flat = node_types.reshape(E * N_per)

    env_offsets = (torch.arange(E, device=device) * N_per).unsqueeze(1)
    agent_ids = torch.arange(A, device=device).unsqueeze(0) + env_offsets
    goal_ids = torch.arange(A, 2 * A, device=device).unsqueeze(0) + env_offsets
    obs_ids = torch.arange(2 * A, 2 * A + O, device=device).unsqueeze(0) + env_offsets
    pad_ids = (N_per - 1 + env_offsets.squeeze(1)).long()

    edges_flat, recvs_flat, sends_flat, n_edges_per_env = _make_edge_list(
        agent_state,
        goal_state,
        obs_state,
        agent_ids,
        goal_ids,
        obs_ids,
        pad_ids,
        obs_radius=obs_radius,
    )

    n_nodes = torch.full((E,), N_per, dtype=torch.long, device=device)
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
    agent_state: torch.Tensor,
    goal_state: torch.Tensor,
    obs_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble batched node features, physical states, and node-type ids."""
    E, A, S = agent_state.shape
    G = goal_state.shape[1]
    O = obs_state.shape[1]
    device, dtype = agent_state.device, agent_state.dtype

    state_pad = torch.full((E, 1, S), -1.0, dtype=dtype, device=device)
    states = torch.cat([agent_state, goal_state, obs_state, state_pad], dim=1)

    N = A + G + O + 1
    indicator = torch.zeros(E, N, NUM_TYPE_INDICATORS, dtype=dtype, device=device)
    indicator[:, :A, 2] = 1.0
    indicator[:, A : A + G, 1] = 1.0
    indicator[:, A + G : A + G + O, 0] = 1.0
    nodes = torch.cat([states, indicator], dim=-1)

    node_types = torch.full((E, N), PAD_TYPE, dtype=torch.long, device=device)
    node_types[:, :A] = AGENT_TYPE
    node_types[:, A : A + G] = GOAL_TYPE
    node_types[:, A + G : A + G + O] = OBS_TYPE

    return nodes, states, node_types


def _make_edge_list(
    agent_state: torch.Tensor,
    goal_state: torch.Tensor,
    obs_state: torch.Tensor,
    agent_ids: torch.Tensor,
    goal_ids: torch.Tensor,
    obs_ids: torch.Tensor,
    pad_ids: torch.Tensor,
    *,
    obs_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build fixed-size edge blocks, redirecting inactive edges to padding nodes."""
    E, A, S = agent_state.shape
    O = obs_state.shape[1]
    device = agent_state.device

    a_pos = agent_state[..., :2]

    dist_aa = torch.cdist(a_pos, a_pos)
    aa_mask = (dist_aa < obs_radius) & ~torch.eye(A, dtype=torch.bool, device=device)
    aa_feats = agent_state[:, :, None, :] - agent_state[:, None, :, :]
    aa_f, aa_r, aa_s = _flatten_dense_edge_block(aa_feats, aa_mask, agent_ids, agent_ids, pad_ids)

    diag = torch.arange(A, device=device)
    ag_feats = agent_state.new_zeros(E, A, A, S)
    ag_feats[:, diag, diag, :] = agent_state - goal_state
    ag_mask = torch.eye(A, dtype=torch.bool, device=device).unsqueeze(0)
    ag_f, ag_r, ag_s = _flatten_dense_edge_block(ag_feats, ag_mask, agent_ids, goal_ids, pad_ids)

    edge_f_parts = [aa_f, ag_f]
    recv_parts = [aa_r, ag_r]
    send_parts = [aa_s, ag_s]
    n_edges_per_env = A * A + A * A

    if O > 0:
        o_pos = obs_state[..., :2]
        dist_ao = torch.cdist(a_pos, o_pos)
        ao_mask = dist_ao < obs_radius
        ao_feats = agent_state[:, :, None, :] - obs_state[:, None, :, :]
        ao_f, ao_r, ao_s = _flatten_dense_edge_block(ao_feats, ao_mask, agent_ids, obs_ids, pad_ids)
        edge_f_parts.append(ao_f)
        recv_parts.append(ao_r)
        send_parts.append(ao_s)
        n_edges_per_env += A * O

    edges_flat = torch.cat(edge_f_parts, dim=0)
    recvs_flat = torch.cat(recv_parts, dim=0)
    sends_flat = torch.cat(send_parts, dim=0)
    edges_flat = torch.cat(
        [edges_flat, edges_flat.new_zeros(edges_flat.shape[0], NUM_TYPE_INDICATORS)],
        dim=-1,
    )

    return edges_flat, recvs_flat, sends_flat, n_edges_per_env


def _flatten_dense_edge_block(
    edge_feats: torch.Tensor,
    edge_mask: torch.Tensor,
    recv_ids: torch.Tensor,
    send_ids: torch.Tensor,
    pad_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a dense edge grid and route inactive entries to each env's pad node."""
    E, R, Sn, F = edge_feats.shape
    recv_grid = recv_ids[:, :, None].expand(E, R, Sn)
    send_grid = send_ids[:, None, :].expand(E, R, Sn)
    pad_grid = pad_ids[:, None, None].expand(E, R, Sn)

    recv_flat = torch.where(edge_mask, recv_grid, pad_grid).reshape(-1)
    send_flat = torch.where(edge_mask, send_grid, pad_grid).reshape(-1)
    feats_flat = edge_feats.reshape(E * R * Sn, F)

    return feats_flat, recv_flat, send_flat


class GraphTransformer(MessagePassing):
    """
    Single multi-head self-attention layer over a graph.
    
    Edge features are added to the value vector before the attention mixing, 
    and the outputs of the heads are averaged (not concatenated) so the output 
    dimension stays 'out_dim' regardless of 'n_heads'. 
    
    A residual-style 'node_proj' branch is added to the aggregated messages before the activation.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        out_dim: int,
        n_heads: int,
        act: Callable = torch.relu,
    ):
        super().__init__(aggr='add')  # "Add" aggregation. (equal to "sum"?)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.act = act

        def init_linear(layer: nn.Linear, bias: bool = True):
            nn.init.orthogonal_(layer.weight)  # nn.init.xavier_uniform_(query.weight)
            if bias:
                nn.init.zeros_(layer.bias)

        self.query = nn.Linear(in_dim, out_dim * n_heads)
        self.key = nn.Linear(in_dim, out_dim * n_heads)
        self.value = nn.Linear(in_dim, out_dim * n_heads)
        self.edge_feats = nn.Linear(edge_dim, out_dim * n_heads, bias=False)
        
        self.node_proj = nn.Linear(in_dim, out_dim)

        for layer in (self.query, self.key, self.value, self.node_proj):
            init_linear(layer)
        init_linear(self.edge_feats, bias=False)

    def forward(self, graph: GraphData) -> GraphData:
        # torch_geometric expects edge_index with row 0 = source, row 1 = target
        edge_index = torch.stack([graph.senders, graph.receivers], dim=0)
        msgs = self.propagate(edge_index=edge_index, x=graph.nodes, edge_attr=graph.edges)
        out = self.act(self.node_proj(graph.nodes) + msgs)
        return graph._replace(nodes=out)

    def message(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x_i: torch.Tensor,       # receiver features (pyg convention)
        x_j: torch.Tensor,       # sender features
        edge_attr: torch.Tensor,
        index: torch.Tensor,     # receiver index per edge (for softmax grouping)
    ) -> torch.Tensor:
        Q = self.query(x_i).reshape(x_i.shape[0], self.n_heads, self.out_dim)
        K = self.key(x_j).reshape(x_j.shape[0], self.n_heads, self.out_dim)
        V = self.value(x_j).reshape(x_j.shape[0], self.n_heads, self.out_dim)
        E = self.edge_feats(edge_attr).reshape(edge_attr.shape[0], self.n_heads, self.out_dim) # edge features

        # scaled dot-product attention; softmax groups over edges sharing a receiver
        attn = (Q * K).sum(dim=-1) / (self.out_dim ** 0.5)
        attn = softmax_pyg(attn, index)

        msgs = attn.unsqueeze(-1) * (V + E)
        return msgs.mean(dim=1)  # average over heads


class GraphTransformerGNN(nn.Module):
    """
    Stack of 'n_layers' of 'GraphTransformer'. 
    
    Hidden layers use 'msg_dim'; final layer projects to 'out_dim'. 
    
    If 'node_type' is passed to 'forward', only nodes of that type are returned, 
    reshaped per sub-graph as '(batch_shape, n_type, out_dim)'.
    """

    def __init__(self, in_dim: int, edge_dim: int, msg_dim: int, out_dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.msg_dim = msg_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        layers = []
        cur_dim = in_dim
        for i in range(n_layers):
            layer_out_dim = out_dim if i == n_layers - 1 else msg_dim
            layers.append(GraphTransformer(cur_dim, edge_dim, layer_out_dim, n_heads, torch.relu))
            cur_dim = layer_out_dim
        self.gnn_layers = nn.ModuleList(layers)

    def forward(
        self,
        graph: GraphData,
        node_type: Optional[int] = None,
        n_type: Optional[int] = None,
    ) -> torch.Tensor:
        for layer in self.gnn_layers:
            graph = layer(graph)

        if node_type is None:
            return graph.nodes
        assert n_type is not None, "n_type must be provided when filtering by node_type"
        return graph.get_type_nodes(node_type, n_type)



class MLP(nn.Module):
    """
    Fully-connected network with orthogonal init and optional per-layer LayerNorm. 
   
    Activation is applied after each layer; the final layer's activation can be disabled via 'act_final=False' 
    and its weight can be rescaled via 'scale_final' (typical trick for policy/value heads so that the initial output is small).
    """

    def __init__(
        self,
        hid_sizes: Sequence[int],
        in_dim: int,
        act: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu,
        act_final: bool = True,
        use_layernorm: bool = True,
        scale_final: Optional[float] = None,
    ):
        super().__init__()
        self.hid_sizes = tuple(hid_sizes)
        self.in_dim = in_dim
        self.act = act
        self.act_final = act_final
        self.use_layernorm = use_layernorm
        self.scale_final = scale_final

        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for hid_size in self.hid_sizes:
            self.layers.append(nn.Linear(prev_dim, hid_size))
            prev_dim = hid_size

        self.layer_norms = (
            nn.ModuleList([nn.LayerNorm(h) for h in self.hid_sizes]) if self.use_layernorm else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, layer in enumerate(self.layers):
            layer = cast(nn.Linear, layer)
            is_last = i == len(self.layers) - 1
            nn.init.orthogonal_(layer.weight)
            
            if is_last and self.scale_final is not None:
                with torch.no_grad():
                    layer.weight.mul_(self.scale_final)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            x = layer(x)
            no_activation = is_last and not self.act_final
            if not no_activation:
                if self.use_layernorm:
                    assert self.layer_norms is not None
                    x = self.layer_norms[i](x)
                x = self.act(x)
        return x


class RNN(nn.Module):
    """
    Multi-layer stateful RNN over a batch of agents.

    Inputs/outputs:
        x:         [n_agents, in_dim]
        rnn_state: [n_layers, n_agents, n_carries, hid_size]
                   - GRU:  n_carries = 1 (just 'h')
                   - LSTM: n_carries = 2 ('h', 'c')
    """

    def __init__(self, rnn_cell: str, input_size: int, hidden_size: int, rnn_layers: int):
        super().__init__()
        assert rnn_cell in {"gru", "lstm"}
        self.rnn_cell = rnn_cell
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

        cells = []
        for i in range(rnn_layers):
            in_size = input_size if i == 0 else hidden_size
            
            if rnn_cell == "gru":
                cells.append(nn.GRUCell(in_size, hidden_size))
            else:
                cells.append(nn.LSTMCell(in_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor, rnn_state: torch.Tensor):
        # L -> n_layers, N -> n_agents, C -> n_carries, H -> hid_size
        
        new_states = []
        for i, cell in enumerate(self.cells):
            if self.rnn_cell == "gru":
                h_i = rnn_state[i, :, 0, :]                # [N, H]
                h_next = cell(x, h_i)                      # [N, H]
                x = h_next
                new_states.append(h_next.unsqueeze(1))     # [N, 1, H]
            else:  # lstm
                h_i = rnn_state[i, :, 0, :]                # [N, H]
                c_i = rnn_state[i, :, 1, :]                # [N, H]
                h_next, c_next = cell(x, (h_i, c_i))
                x = h_next
                new_states.append(torch.stack([h_next, c_next], dim=1))  # [N, 2, H]
        return x, torch.stack(new_states, dim=0)  # [L, N, C, H]

    @torch.no_grad()
    def initialize_carry(self, n_agents: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        n_carries = 1 if self.rnn_cell == "gru" else 2
        return torch.zeros(self.rnn_layers, n_agents, n_carries, self.hidden_size, device=device)
