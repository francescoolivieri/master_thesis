"""
Numpy-backend forward-pass parity check for the DGPPO value networks.

Re-implements one layer of :class:`GraphTransformer`, the MLP head, the GRU
cell and the final linear projection in pure numpy, using the weights stored
in the JAX fixture. Comparing against ``checkpoints/update/value/...`` tells
us whether the Flax value networks have been faithfully reproduced.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Flax-equivalent primitives (implemented in pure numpy)
# ---------------------------------------------------------------------------


def dense(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = x @ kernel
    if bias is not None:
        y = y + bias
    return y


def layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * scale + bias


def segment_sum(values: np.ndarray, segment_ids: np.ndarray, num_segments: int) -> np.ndarray:
    # naive reference implementation of jraph.segment_sum
    out = np.zeros((num_segments,) + values.shape[1:], dtype=values.dtype)
    for i in range(values.shape[0]):
        out[segment_ids[i]] += values[i]
    return out


def segment_softmax(logits: np.ndarray, segment_ids: np.ndarray, num_segments: int) -> np.ndarray:
    # per-segment softmax, with the usual max-subtraction for stability
    out = np.zeros_like(logits)
    for seg in range(num_segments):
        mask = segment_ids == seg
        if not np.any(mask):
            continue
        seg_logits = logits[mask]
        seg_logits = seg_logits - np.max(seg_logits, axis=0, keepdims=True)
        seg_exp = np.exp(seg_logits)
        out[mask] = seg_exp / np.sum(seg_exp, axis=0, keepdims=True)
    return out


# Weight-key layout for a single GraphTransformer layer (Flax naming):
#   Dense_0: query, Dense_1: key, Dense_2: value,
#   Dense_3: edge projection (no bias), Dense_4: node projection (residual).
_GT_KEYS = [
    "Dense_0/kernel", "Dense_0/bias",
    "Dense_1/kernel", "Dense_1/bias",
    "Dense_2/kernel", "Dense_2/bias",
    "Dense_3/kernel",
    "Dense_4/kernel", "Dense_4/bias",
]


def _load_gt_params(z, prefix):
    return {k: z[f"{prefix}/{k}"] for k in _GT_KEYS}


def graph_transformer_layer(nodes, edges, senders, receivers, params):
    """Single GraphTransformer layer, faithful numpy re-implementation."""
    q = dense(nodes[receivers], params["Dense_0/kernel"], params["Dense_0/bias"])
    k = dense(nodes[senders], params["Dense_1/kernel"], params["Dense_1/bias"])
    v = dense(nodes[senders], params["Dense_2/kernel"], params["Dense_2/bias"])
    e = dense(edges, params["Dense_3/kernel"], None)

    n_heads = 3
    out_dim = q.shape[-1] // n_heads
    q = q.reshape(q.shape[0], n_heads, out_dim)
    k = k.reshape(k.shape[0], n_heads, out_dim)
    v = v.reshape(v.shape[0], n_heads, out_dim)
    e = e.reshape(e.shape[0], n_heads, out_dim)

    # scaled dot-product attention, grouped softmax per receiver, mean over heads
    attn = np.sum(q * k, axis=-1) / math.sqrt(out_dim)
    attn = segment_softmax(attn, receivers, num_segments=nodes.shape[0])
    attn = attn[..., None]
    msgs = attn * (v + e)
    msgs = np.mean(msgs, axis=1)
    aggr = segment_sum(msgs, receivers, num_segments=nodes.shape[0])

    # residual-style node projection + ReLU
    node_proj = dense(nodes, params["Dense_4/kernel"], params["Dense_4/bias"])
    return np.maximum(node_proj + aggr, 0.0)


def graph_transformer_gnn(nodes, edges, senders, receivers, layer_params):
    x = nodes
    for p in layer_params:
        x = graph_transformer_layer(x, edges, senders, receivers, p)
    return x


def flax_gru_cell(x, h, p):
    """Flax-style GRU cell (note the separate biases on reset/update gates)."""
    pre_r = dense(x, p["ir/kernel"], p["ir/bias"]) + dense(h, p["hr/kernel"], None)
    pre_z = dense(x, p["iz/kernel"], p["iz/bias"]) + dense(h, p["hz/kernel"], None)
    pre_r = np.asarray(pre_r, dtype=np.float32)
    pre_z = np.asarray(pre_z, dtype=np.float32)
    r = 1.0 / (1.0 + np.exp(-pre_r))
    z = 1.0 / (1.0 + np.exp(-pre_z))
    n = np.tanh(dense(x, p["in/kernel"], p["in/bias"]) + r * dense(h, p["hn/kernel"], p["hn/bias"]))
    return (1.0 - z) * n + z * h


_GRU_KEYS = [
    "in/kernel", "in/bias",
    "ir/kernel", "ir/bias",
    "iz/kernel", "iz/bias",
    "hn/kernel", "hn/bias",
    "hr/kernel",
    "hz/kernel",
]


def mlp_head(x, params):
    """Two-layer Dense-LN-ReLU head matching ``MLP(hid_sizes=(64, 64))``."""
    x = dense(x, params["Dense_0/kernel"], params["Dense_0/bias"])
    x = layer_norm(x, params["LayerNorm_0/scale"], params["LayerNorm_0/bias"])
    x = np.maximum(x, 0.0)
    x = dense(x, params["Dense_1/kernel"], params["Dense_1/bias"])
    x = layer_norm(x, params["LayerNorm_1/scale"], params["LayerNorm_1/bias"])
    x = np.maximum(x, 0.0)
    return x


_HEAD_KEYS = [
    "Dense_0/kernel", "Dense_0/bias",
    "Dense_1/kernel", "Dense_1/bias",
    "LayerNorm_0/scale", "LayerNorm_0/bias",
    "LayerNorm_1/scale", "LayerNorm_1/bias",
]


# ---------------------------------------------------------------------------
# End-to-end forward passes for the Vh and Vl networks
# ---------------------------------------------------------------------------


@dataclass
class ForwardParity:
    """Pulls graph inputs and weights from the fixture and runs the numpy
    re-implementations of the Vh (per-agent, 1 GNN layer) and Vl (centralized,
    2 GNN layers) value networks."""

    z: Any
    n_agents: int

    def _graph_at(self, prefix: str, t: int):
        # Returns the graph components at time ``t`` for sub-graph 0.
        nodes = self.z[f"inputs/{prefix}/2"][0, t]
        edges = self.z[f"inputs/{prefix}/3"][0, t]
        receivers = self.z[f"inputs/{prefix}/5"][0, t]
        senders = self.z[f"inputs/{prefix}/6"][0, t]
        node_type = self.z[f"inputs/{prefix}/7"][0, t]
        return nodes, edges, senders, receivers, node_type

    def _type_nodes(self, nodes, node_type, type_idx=0):
        # Equivalent of GraphData.get_type_nodes(type_idx, n_agents)[0].
        return nodes[node_type == type_idx][: self.n_agents]

    def forward_vh_single(self, t: int):
        """Decentralized Vh head: one output per agent (single timestep)."""
        nodes, edges, senders, receivers, node_type = self._graph_at("rollout/graph", t)

        gnn_prefix = "inputs/params_before_update/Vh/params/GraphTransformerGNN_0/GraphTransformer_0"
        gnn_p = _load_gt_params(self.z, gnn_prefix)
        x = graph_transformer_gnn(nodes, edges, senders, receivers, [gnn_p])
        x = self._type_nodes(x, node_type, type_idx=0)

        head_prefix = "inputs/params_before_update/Vh/params/ValueGNNHead"
        head_p = {k: self.z[f"{head_prefix}/{k}"] for k in _HEAD_KEYS}
        x = mlp_head(x, head_p)

        # Vh is per-agent: RNN state is indexed at ``[0, t, 0, :, 0, :]``.
        rnn_state = self.z["inputs/rollout/rnn_states"][0, t, 0, :, 0, :]
        rnn_prefix = "inputs/params_before_update/Vh/params/RNN_0/GRUCell_1"
        rnn_p = {k: self.z[f"{rnn_prefix}/{k}"] for k in _GRU_KEYS}
        x = flax_gru_cell(x, rnn_state, rnn_p)

        out_prefix = "inputs/params_before_update/Vh/params/Dense_0"
        return dense(x, self.z[f"{out_prefix}/kernel"], self.z[f"{out_prefix}/bias"])

    def forward_vl_scan(self):
        """Centralized Vl head: a single scalar per timestep, computed via a
        recurrent scan over the trajectory (mean over agents at each step)."""
        T = self.z["inputs/rollout/rewards"].shape[1]
        h = np.zeros((1, 64), dtype=np.float32)
        values = np.zeros((T,), dtype=np.float32)

        # Vl has 2 GNN layers (shared across time).
        gnn_layers = [
            _load_gt_params(
                self.z,
                f"inputs/params_before_update/Vl/params/GraphTransformerGNN_0/GraphTransformer_{i}",
            )
            for i in (0, 1)
        ]
        head_prefix = "inputs/params_before_update/Vl/params/ValueGNNHead"
        rnn_prefix = "inputs/params_before_update/Vl/params/RNN_0/GRUCell_1"
        out_prefix = "inputs/params_before_update/Vl/params/Dense_0"
        head_params = {k: self.z[f"{head_prefix}/{k}"] for k in _HEAD_KEYS}
        rnn_params = {k: self.z[f"{rnn_prefix}/{k}"] for k in _GRU_KEYS}
        out_w = self.z[f"{out_prefix}/kernel"]
        out_b = self.z[f"{out_prefix}/bias"]

        for t in range(T):
            nodes, edges, senders, receivers, node_type = self._graph_at("rollout/graph", t)
            x = graph_transformer_gnn(nodes, edges, senders, receivers, gnn_layers)
            x = self._type_nodes(x, node_type, type_idx=0)
            x = np.mean(x, axis=0, keepdims=True)
            x = mlp_head(x, head_params)
            h = flax_gru_cell(x, h, rnn_params)
            values[t] = dense(h, out_w, out_b).squeeze()
        return values[None, :]


def run_forward_parity(fixture_path: str, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    z = np.load(fixture_path)
    n_agents = int(np.asarray(z["metadata/config/num_agents"]))
    fwd = ForwardParity(z=z, n_agents=n_agents)

    T = z["inputs/rollout/rewards"].shape[1]
    vh = np.stack([fwd.forward_vh_single(t) for t in range(T)], axis=0)[None, ...]
    vl = fwd.forward_vl_scan()

    ref_vh = np.asarray(z["checkpoints/update/value/bTah_Vh"], dtype=np.float32)
    ref_vl = np.asarray(z["checkpoints/update/value/bT_Vl"], dtype=np.float32)

    checks = {
        "forward/value/bTah_Vh": (vh, ref_vh),
        "forward/value/bT_Vl": (vl, ref_vl),
    }

    all_ok = True
    print("=== Forward Parity (Vl, Vh) ===")
    for name, (got, r) in checks.items():
        ok = np.allclose(got, r, atol=atol, rtol=rtol)
        all_ok = all_ok and bool(ok)
        max_abs = float(np.max(np.abs(got.astype(np.float64) - r.astype(np.float64))))
        print(f"{'PASS' if ok else 'FAIL':4} | {name:24} | max_abs={max_abs:.6e}")
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=str, default="source/isaac_pursuit_evasion/nn/update_fixture.npz")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    ok = run_forward_parity(args.fixture, atol=args.atol, rtol=args.rtol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
