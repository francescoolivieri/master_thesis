import torch
import torch.nn as nn
from dgppo.update_helpers import _evaluate_policy_chunks, _evaluate_vl_chunks
from dgppo.utils import GraphData


class _IdentityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hid_sizes = (1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _AccumulatingRNN(nn.Module):
    def initialize_carry(self, n_agents: int, device=None) -> torch.Tensor:
        return torch.zeros(1, n_agents, 1, 1, device=device)

    def forward(self, x: torch.Tensor, rnn_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = rnn_state[0, :, 0, :] + x
        return h, h.reshape(1, x.shape[0], 1, 1)


class _ConstantGNN(nn.Module):
    def forward(self, graph: GraphData, node_type: int | None = None, n_type: int | None = None) -> torch.Tensor:
        return graph.nodes.reshape(tuple(graph.n_nodes.shape) + (1, 1))


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = _AccumulatingRNN()
        self.gnn = _ConstantGNN()
        self.mlp = _IdentityHead()
        self.scale_hid = nn.Identity()
        self.mean_head = nn.Identity()
        self.std_head = nn.Linear(1, 1)
        nn.init.zeros_(self.std_head.weight)
        nn.init.zeros_(self.std_head.bias)
        self.std_dev_init_inv = 0.0
        self.std_dev_min = 1.0e-5

    def initialize_carry(self, n_agents_total: int, device=None) -> torch.Tensor:
        return self.rnn.initialize_carry(n_agents_total, device=device)


class _ToyValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = _AccumulatingRNN()
        self.gnn = _ConstantGNN()
        self.head = _IdentityHead()
        self.net = nn.Module()
        self.net.n_out = 1
        self.net.value_out = nn.Identity()


def _toy_graph(*, B: int, C: int, R: int, A: int) -> GraphData:
    n_graphs = B * C * R
    return GraphData(
        n_nodes=torch.ones(B, C, R, dtype=torch.long),
        n_edges=torch.zeros(B, C, R, dtype=torch.long),
        nodes=torch.ones(n_graphs * A, 1),
        edges=torch.zeros(0, 1),
        states=torch.ones(n_graphs * A, 1),
        receivers=torch.zeros(0, dtype=torch.long),
        senders=torch.zeros(0, dtype=torch.long),
        node_types=torch.zeros(n_graphs * A, dtype=torch.long),
    )


def test_recurrent_policy_replay_resets_after_done_inside_chunk() -> None:
    B, C, R, A = 1, 1, 3, 1
    graph = _toy_graph(B=B, C=C, R=R, A=A)
    actions = torch.zeros(B, C, R, A, 1)

    logp_no_done, _ = _evaluate_policy_chunks(
        policy=_ToyPolicy(),
        graph=graph,
        action_chunks=actions,
        done_chunks=None,
        B=B,
        C=C,
        R=R,
        A=A,
        compute_entropy=False,
    )
    logp_with_done, _ = _evaluate_policy_chunks(
        policy=_ToyPolicy(),
        graph=graph,
        action_chunks=actions,
        done_chunks=torch.tensor([[[True, False, False]]]),
        B=B,
        C=C,
        R=R,
        A=A,
        compute_entropy=False,
    )

    assert logp_with_done[0, 0, 1, 0] > logp_no_done[0, 0, 1, 0]


def test_recurrent_vl_replay_resets_after_done_inside_chunk() -> None:
    B, C, R, A = 1, 1, 3, 1
    graph = _toy_graph(B=B, C=C, R=R, A=A)

    values_no_done = _evaluate_vl_chunks(
        Vl=_ToyValueNet(),
        graph=graph,
        B=B,
        C=C,
        R=R,
        A=A,
        device=torch.device("cpu"),
        done_chunks=None,
    )
    values_with_done = _evaluate_vl_chunks(
        Vl=_ToyValueNet(),
        graph=graph,
        B=B,
        C=C,
        R=R,
        A=A,
        device=torch.device("cpu"),
        done_chunks=torch.tensor([[[True, False, False]]]),
    )

    assert values_no_done.tolist() == [[[1.0, 2.0, 3.0]]]
    assert values_with_done.tolist() == [[[1.0, 1.0, 2.0]]]
