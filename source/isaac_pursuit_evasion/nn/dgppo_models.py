"""
Torch policy/critic networks for the DGPPO IsaacLab port.

These are intentionally *not* skrl :class:`GaussianMixin` / :class:`DeterministicMixin`
models: those mixins assume flat observation tensors, whereas DGPPO operates
on :class:`GraphData`. We expose a small, explicit interface
(``act`` / ``evaluate`` / ``get_value``) that the custom :class:`DGPPOAgent`
calls directly.

Structure mirrors the JAX reference in ``dgppo/algo/module/{policy,value}.py``:

* :class:`DGPPOPolicy` -- GNN + MLP + (optional) RNN + ``ScaleHid`` + mean/std
  heads producing a squashed Gaussian (``TanhTransformedNormal``) over
  per-agent actions.
* :class:`DGPPOCritic` -- two value heads sharing a fresh GNN each
  (same pattern as the JAX reference): ``Vl`` centralized scalar via
  :class:`RStateFn`, ``Vh`` per-agent NH-dimensional via :class:`DecStateFn`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GraphData, GraphTransformerGNN
from .train_dgppo import MLP, RNN, DecStateFn, RStateFn


# ---------------------------------------------------------------------------
# Squashed Gaussian (Tanh-transformed Normal) helper
# ---------------------------------------------------------------------------


class TanhNormal:
    """Squashed-Gaussian distribution used by the DGPPO policy.

    Matches the JAX reference (``dgppo/algo/module/distribution.py``): a
    diagonal Normal over a pre-squash action ``u``, with ``a = tanh(u)`` being
    the actual action sent to the env. Provides ``sample``/``log_prob``/
    ``entropy``/``mode`` over the transformed action.

    Reparameterized sampling is used so that gradients flow through the
    policy parameters on-policy only (PPO uses the log-prob form so this is
    not strictly required here, but keeping the standard convention is
    cheap and future-proofs against auxiliary losses).
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def sample(self) -> torch.Tensor:
        noise = torch.randn_like(self.mean)
        u = self.mean + self.std * noise
        return torch.tanh(u)

    def log_prob(self, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Log-density of a squashed action under this distribution.

        ``action`` lives in ``(-1, 1)``; we invert the tanh, compute the
        Gaussian log-density, and add the tanh change-of-variable correction.
        """
        # Clip to avoid atanh(+/-1) -> inf.
        a_clipped = action.clamp(-1.0 + eps, 1.0 - eps)
        u = torch.atanh(a_clipped)
        log_p_u = (
            -0.5 * ((u - self.mean) / self.std).pow(2)
            - torch.log(self.std)
            - 0.5 * math.log(2.0 * math.pi)
        )
        # Tanh Jacobian: log|d a / d u| = log(1 - tanh(u)^2); sum over action dims.
        log_det = torch.log(torch.clamp(1.0 - a_clipped.pow(2), min=eps))
        return (log_p_u - log_det).sum(dim=-1)

    def entropy(self, n_samples: int = 1) -> torch.Tensor:
        """MC estimate of the differential entropy of the squashed Gaussian.

        Closed form does not exist; a single reparam sample is enough for
        the small regularization weight DGPPO uses (``coef_ent``).
        """
        entropies = []
        for _ in range(n_samples):
            u = self.mean + self.std * torch.randn_like(self.mean)
            a = torch.tanh(u)
            log_p_u = (
                -0.5 * ((u - self.mean) / self.std).pow(2)
                - torch.log(self.std)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_det = torch.log(torch.clamp(1.0 - a.pow(2), min=1e-6))
            entropies.append(-(log_p_u - log_det).sum(dim=-1))
        return torch.stack(entropies, dim=0).mean(dim=0)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)


# ---------------------------------------------------------------------------
# Shared backbone (GNN + MLP + optional RNN) used by policy and critic heads
# ---------------------------------------------------------------------------


class _GNNBackbone(nn.Module):
    """GNN + MLP + optional RNN. Outputs per-agent features.

    Ported from ``dgppo/algo/module/policy.py::PolicyNet`` + the ``head``
    MLP in ``PPOPolicy``. Keeping it reusable lets the critic re-instantiate
    its own GNN (the reference does not share GNN weights between policy and
    critic, and neither do we).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        gnn_layers: int,
        gnn_out_dim: int,
        gnn_msg_dim: int,
        gnn_heads: int,
        mlp_hid: tuple[int, ...],
        use_rnn: bool,
        rnn_cell: str,
        rnn_hidden: int,
        rnn_layers: int,
    ) -> None:
        super().__init__()
        self.gnn = GraphTransformerGNN(
            in_dim=node_dim,
            msg_dim=gnn_msg_dim,
            out_dim=gnn_out_dim,
            n_heads=gnn_heads,
            n_layers=gnn_layers,
        )
        self.mlp = MLP(hid_sizes=mlp_hid, in_dim=gnn_out_dim, act=F.relu, act_final=True)
        self.use_rnn = bool(use_rnn)
        if self.use_rnn:
            self.rnn: Optional[RNN] = RNN(
                rnn_cell=rnn_cell,
                input_size=mlp_hid[-1],
                hidden_size=rnn_hidden,
                rnn_layers=rnn_layers,
            )
            self.out_dim = rnn_hidden
        else:
            self.rnn = None
            self.out_dim = mlp_hid[-1]

    def forward(
        self,
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents_total: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns per-agent features ``(n_agents_total, out_dim)``."""
        x = self.gnn(graph, node_type=0, n_type=n_agents_total // graph.n_graphs)
        # ``GraphTransformerGNN`` reshapes to (batch_shape, n_type, feat_dim);
        # flatten to (n_agents_total, feat_dim).
        x = x.reshape(-1, x.shape[-1])
        x = self.mlp(x)
        if self.rnn is not None:
            x, rnn_state = self.rnn(x, rnn_state)
        return x, rnn_state

    @torch.no_grad()
    def initialize_carry(self, n_agents_total: int, device=None) -> Optional[torch.Tensor]:
        if self.rnn is None:
            return None
        return self.rnn.initialize_carry(n_agents_total, device=device)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class DGPPOPolicy(nn.Module):
    """Decentralized policy producing one TanhNormal per agent.

    Matches ``PPOPolicy`` / ``TanhNormal`` in the JAX reference: a scaled
    hidden layer is followed by mean/std heads, ``std = softplus(u + c) + eps``
    with ``c = log(exp(std_init) - 1)`` so that at init std equals
    ``std_dev_init``.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        action_dim: int,
        *,
        gnn_layers: int = 1,
        gnn_out_dim: int = 64,
        gnn_msg_dim: int = 32,
        gnn_heads: int = 3,
        mlp_hid: tuple[int, ...] = (64, 64),
        scale_hid: int = 64,
        scale_final: float = 0.01,
        std_dev_init: float = 0.5,
        std_dev_min: float = 1e-5,
        use_rnn: bool = False,
        rnn_cell: str = "gru",
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.std_dev_min = float(std_dev_min)
        # ``softplus(std_init_inv) == std_dev_init``.
        self.std_init_inv = float(math.log(math.exp(std_dev_init) - 1.0))

        self.backbone = _GNNBackbone(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gnn_layers=gnn_layers,
            gnn_out_dim=gnn_out_dim,
            gnn_msg_dim=gnn_msg_dim,
            gnn_heads=gnn_heads,
            mlp_hid=mlp_hid,
            use_rnn=use_rnn,
            rnn_cell=rnn_cell,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
        )
        self.scale_hid = nn.Linear(self.backbone.out_dim, scale_hid)
        nn.init.orthogonal_(self.scale_hid.weight)
        with torch.no_grad():
            self.scale_hid.weight.mul_(scale_final)
        nn.init.zeros_(self.scale_hid.bias)

        self.mean_head = nn.Linear(scale_hid, action_dim)
        self.std_head = nn.Linear(scale_hid, action_dim)
        for layer in (self.mean_head, self.std_head):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def distribution(
        self,
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents_total: int,
    ) -> tuple[TanhNormal, Optional[torch.Tensor]]:
        feats, rnn_state = self.backbone(graph, rnn_state, n_agents_total)
        h = self.scale_hid(feats)
        mean = self.mean_head(h)
        std = F.softplus(self.std_head(h) + self.std_init_inv) + self.std_dev_min
        return TanhNormal(mean, std), rnn_state

    def act(
        self,
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents_total: int,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sample an action (or take the mode) and return ``(a, log_prob, mean_a, rnn_state)``."""
        dist, rnn_state = self.distribution(graph, rnn_state, n_agents_total)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.mode(), rnn_state

    def evaluate(
        self,
        graph: GraphData,
        action: torch.Tensor,
        rnn_state: Optional[torch.Tensor],
        n_agents_total: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(log_prob, entropy, rnn_state)`` for a previously taken action."""
        dist, rnn_state = self.distribution(graph, rnn_state, n_agents_total)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, rnn_state

    @torch.no_grad()
    def initialize_carry(self, n_agents_total: int, device=None) -> Optional[torch.Tensor]:
        return self.backbone.initialize_carry(n_agents_total, device=device)


# ---------------------------------------------------------------------------
# Critic (dual-head)
# ---------------------------------------------------------------------------


class DGPPOCritic(nn.Module):
    """Two value heads used by DGPPO.

    * ``Vl``: centralized scalar value (mean-aggregated GNN -> MLP -> scalar),
      via :class:`RStateFn`.
    * ``Vh``: per-agent NH-dimensional safety value (no aggregation), via
      :class:`DecStateFn`. ``NH`` equals the number of constraint heads the
      env exposes in ``get_costs()``.

    As in the JAX reference the two heads have independent GNN backbones.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_constraints: int,
        *,
        vl_gnn_layers: int = 1,
        vh_gnn_layers: int = 1,
        gnn_out_dim: int = 64,
        gnn_msg_dim: int = 32,
        gnn_heads: int = 3,
        mlp_hid: tuple[int, ...] = (64, 64),
        use_rnn: bool = False,
        rnn_cell: str = "gru",
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_constraints = int(n_constraints)
        self.use_rnn = bool(use_rnn)

        self.vl_gnn = GraphTransformerGNN(
            in_dim=node_dim,
            msg_dim=gnn_msg_dim,
            out_dim=gnn_out_dim,
            n_heads=gnn_heads,
            n_layers=vl_gnn_layers,
        )
        self.vh_gnn = GraphTransformerGNN(
            in_dim=node_dim,
            msg_dim=gnn_msg_dim,
            out_dim=gnn_out_dim,
            n_heads=gnn_heads,
            n_layers=vh_gnn_layers,
        )

        vl_mlp = MLP(hid_sizes=mlp_hid, in_dim=gnn_out_dim, act=F.relu, act_final=True)
        vh_mlp = MLP(hid_sizes=mlp_hid, in_dim=gnn_out_dim, act=F.relu, act_final=True)

        vl_rnn = (
            RNN(rnn_cell=rnn_cell, input_size=mlp_hid[-1], hidden_size=rnn_hidden, rnn_layers=rnn_layers)
            if use_rnn
            else None
        )
        vh_rnn = (
            RNN(rnn_cell=rnn_cell, input_size=mlp_hid[-1], hidden_size=rnn_hidden, rnn_layers=rnn_layers)
            if use_rnn
            else None
        )

        self.vl_head = RStateFn(gnn=self.vl_gnn, mlp=vl_mlp, rnn=vl_rnn, n_out=1)
        self.vh_head = DecStateFn(gnn=self.vh_gnn, mlp=vh_mlp, rnn=vh_rnn, n_out=max(1, self.n_constraints))

    def get_vl(
        self,
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # ``RStateFn`` reduces agents via a mean and outputs (1, n_out) per
        # sub-graph. For a batched graph we call per sub-graph.
        return self._per_subgraph(graph, rnn_state, n_agents, head=self.vl_head)

    def get_vh(
        self,
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._per_subgraph(graph, rnn_state, n_agents, head=self.vh_head)

    @staticmethod
    def _per_subgraph(
        graph: GraphData,
        rnn_state: Optional[torch.Tensor],
        n_agents: int,
        *,
        head: nn.Module,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply ``head`` per sub-graph; stacks the per-env outputs.

        The underlying GNN already operates on the flattened batched graph,
        but the ``RStateFn`` / ``DecStateFn`` heads in ``train_dgppo.py`` are
        written for a single sub-graph (they apply the ``mean`` / per-agent
        filtering with a hard-coded ``n_agents``). Splitting explicitly keeps
        their semantics intact and avoids silent shape bugs.
        """
        # Fast path used by the default IsaacLab setup (no recurrent critic):
        # keep all operations batched and avoid the Python per-env loop.
        if rnn_state is None and getattr(head, "rnn", None) is None:
            per_agent = head.gnn(graph, node_type=0, n_type=n_agents)  # (E, n_agents, gnn_out)
            if isinstance(head, RStateFn):
                x = per_agent.mean(dim=1)  # (E, gnn_out)
                x = head.mlp(x)            # (E, hid)
                v = head.value_out(x).unsqueeze(1)  # (E, 1, n_out)
                return v, None

            E = per_agent.shape[0]
            x = per_agent.reshape(E * n_agents, -1)
            x = head.mlp(x).reshape(E, n_agents, -1)
            v = head.value_out(x)  # (E, n_agents, n_out)
            return v, None

        # Fallback for recurrent critics: preserve sub-graph semantics exactly.
        per_agent = head.gnn(graph, node_type=0, n_type=n_agents)  # (E, n_agents, gnn_out)
        E = per_agent.shape[0]

        vals = []
        new_rnn_states = []
        for e in range(E):
            x = per_agent[e]  # (n_agents, gnn_out)
            rs_e = None
            if rnn_state is not None:
                # rnn_state shape: (n_layers, E*n_agents_for_rnn, carries, hid)
                start = e * x.shape[0] if isinstance(head, DecStateFn) else e
                stride = x.shape[0] if isinstance(head, DecStateFn) else 1
                rs_e = rnn_state[:, start:start + stride]
            elif getattr(head, "rnn", None) is not None:
                # The caller may run value inference with no recurrent carry
                # (e.g. bootstrap/value-only paths). Initialize a zero carry
                # for this sub-graph so RNN-enabled critics remain usable.
                rs_e = head.rnn.initialize_carry(n_agents=x.shape[0], device=x.device)
            if isinstance(head, RStateFn):
                x = x.mean(dim=0, keepdim=True)  # (1, gnn_out)
                x = head.mlp(x)
                if head.rnn is not None:
                    x, rs_e = head.rnn(x, rs_e)
                v = head.value_out(x)  # (1, n_out)
            else:
                x = head.mlp(x)
                if head.rnn is not None:
                    x, rs_e = head.rnn(x, rs_e)
                v = head.value_out(x)  # (n_agents, n_out)
            vals.append(v)
            if rs_e is not None:
                new_rnn_states.append(rs_e)

        stacked = torch.stack(vals, dim=0)  # (E, *, n_out)
        new_rnn_state = None
        if new_rnn_states:
            new_rnn_state = torch.cat(new_rnn_states, dim=1)
        return stacked, new_rnn_state
