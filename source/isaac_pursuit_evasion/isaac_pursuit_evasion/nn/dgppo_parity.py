"""
Numpy-backend parity check for the DGPPO update-time helpers.

This mirrors the JAX reference exactly (reverse scan with a growing rolling
buffer of "next values"), and is used as a ground truth for the torch port
in ``dgppo_losses.py`` / ``dgppo_parity_torch.py``. All checks in
``run_update_fixture_parity`` should PASS against the bundled
``update_fixture.npz``.
"""

import argparse

import numpy as np


def compute_dec_ocp_gae(Tah_hs, T_l, Tp1ah_Vh, Tp1_Vl, disc_gamma, gae_lambda, discount_to_max=True):
    """
    Decomposed-OCP GAE target for both the constraint value ``Vh`` and the
    reward value ``Vl``. See the torch port in ``dgppo_losses.py`` for the
    full docstring of shapes and semantics.
    """
    B, T, A, NH = Tah_hs.shape
    Qh_all = np.zeros_like(Tah_hs)
    Ql_all = np.zeros_like(T_l)
    time_ids = np.arange(T + 1)

    for b in range(B):
        Tah_hs_b = Tah_hs[b]
        T_l_b = T_l[b]
        Tah_Vh = Tp1ah_Vh[b, :-1]
        T_Vl = np.repeat(Tp1_Vl[b, :-1, None], A, axis=1)
        Vh_final = Tp1ah_Vh[b, -1]
        Vl_final = Tp1_Vl[b, -1]

        # Rolling buffers of "next values"; position 0 always holds the most
        # recent bootstrap (starts at the terminal value), deeper positions
        # store values collected at earlier reverse-scan iterations.
        next_Vhs_row = np.zeros((T + 1, A, NH), dtype=Tah_hs.dtype)
        next_Vl_row = np.zeros((T + 1, A), dtype=Tah_hs.dtype)
        next_Vhs_row[0] = Vh_final
        next_Vl_row[0] = Vl_final

        gae_coeffs = np.zeros((T + 1,), dtype=Tah_hs.dtype)
        gae_coeffs[0] = 1.0
        Qs = np.zeros((T, A, NH + 1), dtype=Tah_hs.dtype)

        # ``t`` walks the trajectory backwards, ``step`` is the iteration
        # counter 0..T-1 used to size the mask / GAE coefficient window
        # (matches the JAX reverse-scan semantics).
        for step, t in enumerate(reversed(range(T))):
            hs = Tah_hs_b[t]
            l = T_l_b[t]
            Vhs = Tah_Vh[t]
            Vl = T_Vl[t]

            # Only the first ``step + 1`` buffer positions contain valid data.
            mask = (time_ids <= step).astype(Tah_hs.dtype)
            mask_h = mask[:, None, None]
            mask_l = mask[:, None]

            h_disc = hs.max(axis=-1, keepdims=True) if discount_to_max else hs
            disc_to_h = (1.0 - disc_gamma) * h_disc[None] + disc_gamma * next_Vhs_row
            Vhs_row = mask_h * np.maximum(hs[None], disc_to_h)
            Vl_row = mask_l * (l + disc_gamma * next_Vl_row)

            cat_V_row = np.concatenate([Vhs_row, Vl_row[:, :, None]], axis=-1)
            Qs[t] = np.einsum("tah,t->ah", cat_V_row, gae_coeffs)

            # Push the current Vhs/Vl into the rolling buffer and rebind
            # ``next_*_row`` to the (locally-modified) Vhs_row/Vl_row.
            Vhs_row[step + 1] = Vhs
            Vl_row[step + 1] = Vl
            next_Vhs_row = Vhs_row
            next_Vl_row = Vl_row

            # Shift coefficients one slot deeper and refresh the two leading
            # entries to match the current truncation length.
            gae_coeffs = np.roll(gae_coeffs, shift=1)
            gae_coeffs[0] = gae_lambda ** (step + 1)
            gae_coeffs[1] = (gae_lambda ** step) * (1.0 - gae_lambda)

        Qh_all[b] = Qs[:, :, :NH]
        Ql_all[b] = Qs[:, 0, NH]

    return Qh_all, Ql_all


def compute_cbf_advantages(bT_Ql, bT_Vl, bTah_Vh, bTp1ah_Vh, alpha, cbf_eps, cbf_weight):
    """CBF-based PPO advantage. See ``dgppo_losses.py`` for details."""
    bT_Al_raw = bT_Ql - bT_Vl
    bT_Al_norm = (bT_Al_raw - bT_Al_raw.mean(axis=1, keepdims=True)) / (
        bT_Al_raw.std(axis=1, keepdims=True) + 1e-8
    )
    bTa_Al = np.repeat(bT_Al_norm[:, :, None], bTah_Vh.shape[2], axis=2)

    # Discrete CBF derivative: (V_{t+1} - V_t) / dt + alpha * V_t ; the
    # fixture env uses dt = 0.03 s, hence the 1/dt factor ~ 33.33.
    inv_dt = 1.0 / 0.03
    bTah_cbf_deriv = inv_dt * (bTp1ah_Vh[:, 1:] - bTah_Vh) + alpha * bTah_Vh
    bTah_Acbf = np.maximum(bTah_cbf_deriv + cbf_eps, 0.0)
    bTa_is_safe = (bTah_cbf_deriv <= 0).all(axis=-1)

    bTa_A = np.where(bTa_is_safe, bTa_Al, 0.0)
    bTa_A = bTa_A + bTah_Acbf.max(axis=-1) * cbf_weight
    bTa_A = -bTa_A

    return {
        "bT_Al_raw": bT_Al_raw,
        "bT_Al_norm": bT_Al_norm,
        "bTah_cbf_deriv": bTah_cbf_deriv,
        "bTah_Acbf": bTah_Acbf,
        "bTa_is_safe": bTa_is_safe,
        "bTa_A": bTa_A,
    }


def compute_policy_surrogate(ratio, advantages, clip_eps):
    """Standard clipped PPO surrogate (pessimistic max over clipped/unclipped)."""
    loss_policy1 = -ratio * advantages
    loss_policy2 = -np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss_policy = np.maximum(loss_policy1, loss_policy2).mean()
    clip_frac = (loss_policy2 > loss_policy1).mean()
    return {
        "loss_policy1": loss_policy1,
        "loss_policy2": loss_policy2,
        "loss_policy": loss_policy,
        "clip_frac": clip_frac,
    }


def run_update_fixture_parity(fixture_path, rtol=1e-4, atol=1e-5):
    """Run the numpy helpers against the JAX fixture and report per-key errors."""
    z = np.load(fixture_path)

    Tah_hs = np.asarray(z["inputs/rollout/costs"], dtype=np.float32)
    T_l = -np.asarray(z["inputs/rollout/rewards"], dtype=np.float32)
    bTp1ah_Vh = np.asarray(z["checkpoints/update/value/bTp1ah_Vh"], dtype=np.float32)
    bTp1_Vl = np.asarray(z["checkpoints/update/value/bTp1_Vl"], dtype=np.float32)
    bTah_Vh = np.asarray(z["checkpoints/update/value/bTah_Vh"], dtype=np.float32)
    bT_Vl = np.asarray(z["checkpoints/update/value/bT_Vl"], dtype=np.float32)
    ratio = np.asarray(z["checkpoints/update/policy/ratio"], dtype=np.float32)
    rnn_chunk_ids = np.asarray(z["inputs/batching/rnn_chunk_ids"], dtype=np.int64)

    gamma = float(np.asarray(z["metadata/config/gamma"]))
    gae_lambda = float(np.asarray(z["metadata/config/gae_lambda"]))
    alpha = float(np.asarray(z["metadata/config/alpha"]))
    cbf_eps = float(np.asarray(z["metadata/config/cbf_eps"]))
    cbf_weight = float(np.asarray(z["metadata/config/cbf_weight"]))
    clip_eps = float(np.asarray(z["metadata/config/clip_eps"]))

    Qh, Ql = compute_dec_ocp_gae(Tah_hs, T_l, bTp1ah_Vh, bTp1_Vl, gamma, gae_lambda)
    adv = compute_cbf_advantages(Ql, bT_Vl, bTah_Vh, bTp1ah_Vh, alpha, cbf_eps, cbf_weight)
    ppo = compute_policy_surrogate(ratio, adv["bTa_A"][:, rnn_chunk_ids], clip_eps)

    def ref(key):
        return np.asarray(z[f"checkpoints/{key}"], dtype=np.float32)

    checks = {
        "update/gae/bTah_Qh": (Qh, ref("update/gae/bTah_Qh")),
        "update/gae/bT_Ql": (Ql, ref("update/gae/bT_Ql")),
        "update/adv/bT_Al_raw": (adv["bT_Al_raw"], ref("update/adv/bT_Al_raw")),
        "update/adv/bT_Al_norm": (adv["bT_Al_norm"], ref("update/adv/bT_Al_norm")),
        "update/adv/bTah_cbf_deriv": (adv["bTah_cbf_deriv"], ref("update/adv/bTah_cbf_deriv")),
        "update/adv/bTah_Acbf": (adv["bTah_Acbf"], ref("update/adv/bTah_Acbf")),
        "update/adv/bTa_is_safe": (adv["bTa_is_safe"], np.asarray(z["checkpoints/update/adv/bTa_is_safe"])),
        "update/adv/bTa_A": (adv["bTa_A"], ref("update/adv/bTa_A")),
        "update/policy/loss_policy1": (ppo["loss_policy1"], ref("update/policy/loss_policy1")),
        "update/policy/loss_policy2": (ppo["loss_policy2"], ref("update/policy/loss_policy2")),
        "update/policy/loss_policy": (ppo["loss_policy"], ref("update/policy/loss_policy")),
        "update/policy/clip_frac": (ppo["clip_frac"], ref("update/policy/clip_frac")),
    }

    all_ok = True
    print("=== DGPPO parity checks (numpy backend) ===")
    for name, (got, r) in checks.items():
        ok = np.allclose(got, r, rtol=rtol, atol=atol)
        all_ok = all_ok and bool(ok)
        max_abs = float(np.max(np.abs(got.astype(np.float64) - r.astype(np.float64))))
        print(f"{'PASS' if ok else 'FAIL':4} | {name:34} | max_abs={max_abs:.6e}")
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=str, default="source/isaac_pursuit_evasion/nn/update_fixture.npz")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    ok = run_update_fixture_parity(args.fixture, rtol=args.rtol, atol=args.atol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
