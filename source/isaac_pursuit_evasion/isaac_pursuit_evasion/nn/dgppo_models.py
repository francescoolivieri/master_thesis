"""
Torch network building blocks for the DGPPO port.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

try:
    from .gnn import GraphTransformerGNN
except ImportError:
    from gnn import GraphTransformerGNN


class MLP(nn.Module):
    """
    Fully-connected network with orthogonal init and optional per-layer LayerNorm.
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
                    x = self.layer_norms[i](x)
                x = self.act(x)
        return x


class RNN(nn.Module):
    """
    Multi-layer stateful RNN over a batch of agents.

    Inputs/outputs:
        x:         [n_agents, in_dim]
        rnn_state: [n_layers, n_agents, n_carries, hid_size]
                   - GRU:  n_carries = 1 (h)
                   - LSTM: n_carries = 2 (h, c)
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
        new_states = []
        for i, cell in enumerate(self.cells):
            if self.rnn_cell == "gru":
                h_i = rnn_state[i, :, 0, :]
                h_next = cell(x, h_i)
                x = h_next
                new_states.append(h_next.unsqueeze(1))
            else:
                h_i = rnn_state[i, :, 0, :]
                c_i = rnn_state[i, :, 1, :]
                h_next, c_next = cell(x, (h_i, c_i))
                x = h_next
                new_states.append(torch.stack([h_next, c_next], dim=1))
        return x, torch.stack(new_states, dim=0)

    @torch.no_grad()
    def initialize_carry(self, n_agents: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        n_carries = 1 if self.rnn_cell == "gru" else 2
        return torch.zeros(self.rnn_layers, n_agents, n_carries, self.hidden_size, device=device)


class DecStateFn(nn.Module):
    """
    Decentralized value head: one output per agent.
    """

    def __init__(self, gnn: nn.Module, mlp: nn.Module, rnn: Optional[nn.Module] = None, n_out: int = 1):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.rnn = rnn
        self.n_out = n_out
        self.value_out = nn.Linear(mlp.hid_sizes[-1], n_out)
        nn.init.orthogonal_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)

    def forward(self, graph, rnn_state: torch.Tensor, n_agents: int):
        x = self.gnn(graph, node_type=0, n_type=n_agents)
        x = self.mlp(x)
        assert x.shape[0] == n_agents

        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)

        x = self.value_out(x)
        assert x.shape == (n_agents, self.n_out)
        return x, rnn_state


class RStateFn(nn.Module):
    """
    Centralized value head: one output per sub-graph.
    """

    def __init__(self, gnn: nn.Module, mlp: nn.Module, rnn: Optional[nn.Module] = None, n_out: int = 1):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.rnn = rnn
        self.n_out = n_out

        self.value_out = nn.Linear(mlp.hid_sizes[-1], n_out)
        nn.init.orthogonal_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)

    def forward(self, graph, rnn_state: torch.Tensor, n_agents: int):
        x = self.gnn(graph, node_type=0, n_type=n_agents)
        x = x.mean(dim=0, keepdim=True)
        x = self.mlp(x)

        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)

        x = self.value_out(x)
        assert x.shape == (1, self.n_out)
        return x, rnn_state


class ValueNet(nn.Module):
    """
    Critic network wrapper: GNN + MLP head + optional RNN.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_agents: int,
        n_out: int = 1,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        gnn_layers: int = 1,
        gnn_out_dim: int = 16,
        use_lstm: bool = False,
        decompose: bool = False,
        n_heads: int = 3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.n_out = n_out
        self.decompose = decompose

        self.gnn = GraphTransformerGNN(
            in_dim=node_dim,
            msg_dim=32,
            out_dim=gnn_out_dim,
            n_heads=n_heads,
            n_layers=gnn_layers,
        )
        self.head = MLP(
            hid_sizes=(64, 64),
            in_dim=gnn_out_dim,
            act=nn.functional.relu,
            act_final=True,
        )
        self.rnn = (
            RNN(
                rnn_cell="lstm" if use_lstm else "gru",
                input_size=64,
                hidden_size=64,
                rnn_layers=rnn_layers,
            )
            if use_rnn
            else None
        )

        head_cls = DecStateFn if decompose else RStateFn
        self.net = head_cls(gnn=self.gnn, mlp=self.head, rnn=self.rnn, n_out=n_out)

    def forward(self, graph, rnn_state: torch.Tensor, n_agents: int):
        return self.net(graph, rnn_state, n_agents)
