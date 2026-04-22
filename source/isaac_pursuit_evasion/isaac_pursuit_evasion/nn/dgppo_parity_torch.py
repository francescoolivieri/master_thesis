"""
Torch parity harness for DGPPO update-time helpers.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from .dgppo_losses import compute_cbf_advantages, compute_dec_ocp_gae, compute_policy_surrogate
except ImportError:
    from dgppo_losses import compute_cbf_advantages, compute_dec_ocp_gae, compute_policy_surrogate


def _to_torch(x) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_update_fixture_parity(
    fixture_path: str, rtol: float = 1e-4, atol: float = 1e-5, cbf_dt: float = 0.03
) -> tuple[bool, list]:
    """Check DGPPO torch helpers against the stored JAX fixture."""
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
        dt=cbf_dt,
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
