"""
Torch-side building blocks for the DGPPO port:

* :class:`MLP`, :class:`RNN` -- vanilla MLP and multi-layer RNN matching the
  Flax reference (orthogonal init, optional LayerNorm in MLP, GRU/LSTM cells).
* :class:`DecStateFn`, :class:`RStateFn` -- value/policy heads that combine
  the GNN, MLP, optional RNN and a final linear projection. ``Dec`` returns
  one output per agent, ``R`` returns one output per sub-graph
  (centralized value / reward head).
* :class:`ValueNet` -- convenience wrapper around a GNN + head pair.
* Torch ports of the DGPPO update-time helpers: :func:`compute_dec_ocp_gae`,
  :func:`compute_cbf_advantages`, :func:`compute_policy_surrogate`.
* :func:`run_update_fixture_parity` -- compares the torch implementations
  of the helpers against the JAX fixture stored in ``update_fixture.npz``.

Run as ``python train_dgppo.py --fixture update_fixture.npz`` to execute
the torch-backend parity check.
"""

from __future__ import annotations

import argparse
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    # Package import (used when train_dgppo is imported as part of the
    # ``isaac_pursuit_evasion.nn`` package, e.g. from the DGPPO agent).
    from .gnn import GraphTransformerGNN
except ImportError:
    # Script import (used when this file is executed directly, e.g. the
    # parity harness: ``python train_dgppo.py --fixture ...``).
    from gnn import GraphTransformerGNN


# ---------------------------------------------------------------------------
# Network primitives
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Value / policy heads
# ---------------------------------------------------------------------------


class DecStateFn(nn.Module):
    """
    Decentralized head: one output per agent. 
    
    Applies the shared GNN, filters to agent nodes, passes each agent through the shared MLP/RNN, and projects to ``n_out``.
    """

    def __init__(self, gnn: nn.Module, mlp: nn.Module, rnn: Optional[nn.Module] = None, n_out: int = 1):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.rnn = rnn
        self.n_out = n_out

        # final projection 
        self.value_out = nn.Linear(mlp.hid_sizes[-1], n_out)
        nn.init.orthogonal_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)

    def forward(self, graph, rnn_state: torch.Tensor, n_agents: int):
        x = self.gnn(graph, node_type=0, n_type=n_agents)    # (n_agents, gnn_out_dim)
        x = self.mlp(x)                                       # (n_agents, hid)
        assert x.shape[0] == n_agents

        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)

        x = self.value_out(x)                                 # (n_agents, n_out)
        assert x.shape == (n_agents, self.n_out)
        return x, rnn_state


class RStateFn(nn.Module):
    """
    Centralized head: one output per sub-graph.
    
    Same as 'DecStateFn' but aggregates per-agent GNN outputs with a mean before the MLP/RNN.
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
        x = self.gnn(graph, node_type=0, n_type=n_agents)    # (n_agents, gnn_out_dim)
        x = x.mean(dim=0, keepdim=True)                       # (1, gnn_out_dim)
        x = self.mlp(x)

        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)

        x = self.value_out(x)                                 # (1, n_out)
        assert x.shape == (1, self.n_out)
        return x, rnn_state


class ValueNet(nn.Module):
    """
    Critic network wrapper: GNN + MLP head + optional RNN
    
    Can be in either a 'decentralized' or 'centralized' configuration, 
    kept for possible future multi-agent tests.

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


# ---------------------------------------------------------------------------
# DGPPO update-time helpers
# ---------------------------------------------------------------------------


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
    cbf_scale: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """
    CBF-based advantage used by DGPPO.

    Reward advantage is standardized across time, and the constraint term
    (``V_{t+1} - V_t) / dt + alpha * V_t``) enters as an additive penalty
    whenever any head is unsafe. The env fixture uses ``dt = 1/3``, hence
    the factor of 3 on the finite difference.
    """
    
    # calculate cost advantage and normalize
    bT_Al_raw = bT_Ql - bT_Vl    
    bT_Al_norm = (bT_Al_raw - bT_Al_raw.mean(dim=1, keepdim=True)) / (
        bT_Al_raw.std(dim=1, keepdim=True, unbiased=False) + 1e-8
    )
    bTa_Al = bT_Al_norm[:, :, None].expand(-1, -1, bTah_Vh.shape[2])

    # Discrete CBF derivative: (V_{t+1} - V_t) / dt + alpha * V_t 
    dt = 0.03  # self.env.dt
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


# ---------------------------------------------------------------------------
# Torch parity harness (runs the torch helpers against the JAX fixture)
# ---------------------------------------------------------------------------


def _to_torch(x) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_update_fixture_parity(
    fixture_path: str, rtol: float = 1e-4, atol: float = 1e-5
) -> tuple[bool, list]:
    """Check the torch update helpers against the JAX fixture."""
    z = np.load(fixture_path)

    Tah_hs = _to_torch(z["inputs/rollout/costs"]).float()
    T_l = -_to_torch(z["inputs/rollout/rewards"]).float()
    bTp1ah_Vh = _to_torch(z["checkpoints/update/value/bTp1ah_Vh"]).float()
    bTp1_Vl = _to_torch(z["checkpoints/update/value/bTp1_Vl"]).float()
    bTah_Vh = _to_torch(z["checkpoints/update/value/bTah_Vh"]).float()
    bT_Vl = _to_torch(z["checkpoints/update/value/bT_Vl"]).float()
    ratio = _to_torch(z["checkpoints/update/policy/ratio"]).float()
    rnn_chunk_ids = _to_torch(z["inputs/batching/rnn_chunk_ids"]).long()

    gamma = float(np.asarray(z["metadata/config/gamma"]))
    gae_lambda = float(np.asarray(z["metadata/config/gae_lambda"]))
    alpha = float(np.asarray(z["metadata/config/alpha"]))
    cbf_eps = float(np.asarray(z["metadata/config/cbf_eps"]))
    cbf_weight = float(np.asarray(z["metadata/config/cbf_weight"]))
    clip_eps = float(np.asarray(z["metadata/config/clip_eps"]))

    Qh, Ql = compute_dec_ocp_gae(
        Tah_hs=Tah_hs,
        T_l=T_l,
        Tp1ah_Vh=bTp1ah_Vh,
        Tp1_Vl=bTp1_Vl,
        disc_gamma=gamma,
        gae_lambda=gae_lambda,
    )

    adv = compute_cbf_advantages(
        bT_Ql=Ql,
        bT_Vl=bT_Vl,
        bTah_Vh=bTah_Vh,
        bTp1ah_Vh=bTp1ah_Vh,
        alpha=alpha,
        cbf_eps=cbf_eps,
        cbf_weight=cbf_weight,
    )

    bTa_A_chunked = adv["bTa_A"][:, rnn_chunk_ids]
    ppo = compute_policy_surrogate(ratio=ratio, advantages=bTa_A_chunked, clip_eps=clip_eps)

    def ref(key: str) -> torch.Tensor:
        return _to_torch(z[f"checkpoints/{key}"]).float()

    checks = {
        "update/gae/bTah_Qh": (Qh, ref("update/gae/bTah_Qh")),
        "update/gae/bT_Ql": (Ql, ref("update/gae/bT_Ql")),
        "update/adv/bT_Al_raw": (adv["bT_Al_raw"], ref("update/adv/bT_Al_raw")),
        "update/adv/bT_Al_norm": (adv["bT_Al_norm"], ref("update/adv/bT_Al_norm")),
        "update/adv/bTah_cbf_deriv": (adv["bTah_cbf_deriv"], ref("update/adv/bTah_cbf_deriv")),
        "update/adv/bTah_Acbf": (adv["bTah_Acbf"], ref("update/adv/bTah_Acbf")),
        "update/adv/bTa_is_safe": (adv["bTa_is_safe"].float(), ref("update/adv/bTa_is_safe")),
        "update/adv/bTa_A": (adv["bTa_A"], ref("update/adv/bTa_A")),
        "update/policy/loss_policy1": (ppo["loss_policy1"], ref("update/policy/loss_policy1")),
        "update/policy/loss_policy2": (ppo["loss_policy2"], ref("update/policy/loss_policy2")),
        "update/policy/loss_policy": (ppo["loss_policy"], ref("update/policy/loss_policy")),
        "update/policy/clip_frac": (ppo["clip_frac"], ref("update/policy/clip_frac")),
    }

    rows = []
    all_ok = True
    for name, (got, r) in checks.items():
        ok = torch.allclose(got, r, rtol=rtol, atol=atol)
        all_ok = all_ok and bool(ok)
        rows.append((name, ok, _max_abs(got, r), tuple(got.shape), tuple(r.shape)))

    print("=== DGPPO torch parity checks (fixture) ===")
    for name, ok, max_abs, got_shape, ref_shape in rows:
        status = "PASS" if ok else "FAIL"
        print(f"{status:4} | {name:34} | max_abs={max_abs:.6e} | got={got_shape} ref={ref_shape}")
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")

    return all_ok, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=str,
        default="source/isaac_pursuit_evasion/nn/update_fixture.npz",
        help="Path to update_fixture.npz",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    ok, _ = run_update_fixture_parity(args.fixture, rtol=args.rtol, atol=args.atol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
