"""
Graph neural network primitives used by the DGPPO.

Ported from the JAX/Flax reference in ``dgppo/nn/gnn.py``. The representation
of a batched graph mirrors `jraph.GraphsTuple`: nodes from every sub-graph
(one per parallel simulation) are concatenated along the leading axis, and
``receivers`` / ``senders`` are flat indices into that concatenated tensor.
"""

from dataclasses import dataclass, replace
from typing import Callable, Generic, NamedTuple, Optional, TypeVar

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as softmax_pyg


# Type of the node-level physical state carried in GraphData.states
# (e.g. an agent / goal / obstacle state tensor). Kept generic so callers
# can plug in whatever state representation their environment uses.
_State = TypeVar("_State")


class EdgeBlock(NamedTuple):
    """
    Dense description of edges between a group of receiver nodes and a group
    of sender nodes (e.g. agent->agent, agent->goal). Used as an intermediate
    representation before being flattened into a single (sender, receiver,
    feature) edge list via `make_edges` method.
    """

    edge_feats: torch.Tensor   # (n_recv, n_send, n_edge_feat)
    edge_mask: torch.Tensor    # (n_recv, n_send) -- which edges are active
    ids_recv: torch.Tensor     # (n_recv,) global node ids of receivers
    ids_send: torch.Tensor     # (n_send,) global node ids of senders

    @property
    def n_recv(self) -> int:
        assert self.edge_feats.shape[0] == self.edge_mask.shape[0] == len(self.ids_recv)
        return len(self.ids_recv)

    @property
    def n_send(self) -> int:
        assert self.edge_feats.shape[1] == self.edge_mask.shape[1] == len(self.ids_send)
        return len(self.ids_send)

    @property
    def n_edges(self) -> int:
        return self.n_recv * self.n_send

    def make_edges(self, pad_id: int, edge_mask: Optional[torch.Tensor] = None):
        """
        Flatten the dense edge block into a (sender, receiver, feature) edge list.

        Inactive edges (``edge_mask == False``) are redirected to ``pad_id``,
        the id of a dummy padding node appended to the graph, so that fixed-shape
        tensors can be used even with variable connectivity.

        Args:
            pad_id: id of the padding node to route masked-out edges to.
            edge_mask: optional override of ``self.edge_mask``.
        Returns:
            (edge_feats, receiver_ids, sender_ids), each of length ``n_edges``.
        """
        edge_mask = self.edge_mask if edge_mask is None else edge_mask

        # broadcast ids to a (n_recv, n_send) grid so they align with the mask / feats
        id_recv_grid = self.ids_recv[:, None].expand(self.n_recv, self.n_send)
        id_send_grid = self.ids_send[None, :].expand(self.n_recv, self.n_send)

        e_recvs = torch.where(edge_mask, id_recv_grid, pad_id).reshape(-1)   # -1 flattens the tensor
        e_sends = torch.where(edge_mask, id_send_grid, pad_id).reshape(-1)
        e_edge_feats = self.edge_feats.reshape(self.n_edges, -1)

        assert e_recvs.shape == e_sends.shape == e_edge_feats.shape[:1] == (self.n_edges,)
        
        return e_edge_feats, e_recvs, e_sends


@dataclass
class GraphData(Generic[_State]):
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
    states: _State                   # per-node physical state, (sum_n_nodes, state_dim)
    receivers: torch.Tensor          # (sum_n_edges,) -- indexes ``nodes``
    senders: torch.Tensor            # (sum_n_edges,) -- indexes ``nodes``
    node_types: torch.Tensor         # (sum_n_nodes,) node type ids, -1 for padding

    @property
    def n_graphs(self) -> int:
        # number of sub-graphs (i.e. parallel simulations) in this batch
        if self.n_nodes.ndim == 0:
            return 1
        return int(self.n_nodes.shape[0])

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

    def _replace(self, **kwargs):
        return replace(self, **kwargs)


class GraphTransformer(MessagePassing):
    """
    Single multi-head self-attention layer over a graph.
    
    Edge features are added to the value vector before the attention mixing, 
    and the outputs of the heads are averaged (not concatenated) so the output 
    dimension stays 'out_dim' regardless of 'n_heads'. 
    
    A residual-style 'node_proj' branch is added to the aggregated messages before the activation.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int, act: Callable = torch.relu):
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
        self.edge_feats = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        
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

    def message(
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

    def __init__(self, in_dim: int, msg_dim: int, out_dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.in_dim = in_dim
        self.msg_dim = msg_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        layers = []
        cur_dim = in_dim
        for i in range(n_layers):
            layer_out_dim = out_dim if i == n_layers - 1 else msg_dim
            layers.append(GraphTransformer(cur_dim, layer_out_dim, n_heads, torch.relu))
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
