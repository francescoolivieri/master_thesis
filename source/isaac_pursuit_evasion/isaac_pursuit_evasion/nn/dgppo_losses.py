"""
DGPPO update-time math helpers.
"""

from __future__ import annotations

from typing import Optional

import torch


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
    Compute decomposed-OCP GAE targets for constraint-value heads (Vh) and
    reward value (Vl).
    """
    B, T, A, NH = Tah_hs.shape
    device, dtype = Tah_hs.device, Tah_hs.dtype

    Qh_all = torch.zeros_like(Tah_hs)
    Ql_all = torch.zeros_like(T_l)

    time_ids = torch.arange(T + 1, device=device)

    for b in range(B):
        Tah_hs_b = Tah_hs[b]
        T_l_b = T_l[b]
        Tah_Vh = Tp1ah_Vh[b, :-1]
        T_Vl = Tp1_Vl[b, :-1].unsqueeze(-1).expand(T, A)
        Vh_final = Tp1ah_Vh[b, -1]
        Vl_final = Tp1_Vl[b, -1]

        next_Vhs_row = torch.zeros((T + 1, A, NH), device=device, dtype=dtype)
        next_Vl_row = torch.zeros((T + 1, A), device=device, dtype=dtype)
        next_Vhs_row[0] = Vh_final
        next_Vl_row[0] = Vl_final

        gae_coeffs = torch.zeros((T + 1,), device=device, dtype=dtype)
        gae_coeffs[0] = 1.0

        Qs = torch.zeros((T, A, NH + 1), device=device, dtype=dtype)

        for step, t in enumerate(reversed(range(T))):
            hs = Tah_hs_b[t]
            l = T_l_b[t]
            Vhs = Tah_Vh[t]
            Vl = T_Vl[t]

            mask = (time_ids <= step).to(dtype)
            mask_h = mask[:, None, None]
            mask_l = mask[:, None]

            if discount_to_max:
                h_disc = hs.max(dim=-1).values[:, None]
            else:
                h_disc = hs

            disc_to_h = (1.0 - disc_gamma) * h_disc[None] + disc_gamma * next_Vhs_row
            Vhs_row = mask_h * torch.maximum(hs[None], disc_to_h)
            Vl_row = mask_l * (l + disc_gamma * next_Vl_row)

            cat_V_row = torch.cat([Vhs_row, Vl_row[:, :, None]], dim=-1)
            Qs[t] = torch.einsum("tah,t->ah", cat_V_row, gae_coeffs)

            Vhs_row[step + 1] = Vhs
            Vl_row[step + 1] = Vl
            next_Vhs_row = Vhs_row
            next_Vl_row = Vl_row

            gae_coeffs = torch.roll(gae_coeffs, shifts=1, dims=0)
            gae_coeffs[0] = gae_lambda ** (step + 1)
            gae_coeffs[1] = (gae_lambda ** step) * (1.0 - gae_lambda)

        Qh_all[b] = Qs[:, :, :NH]
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
    dt: float = 0.03,
) -> dict[str, torch.Tensor]:
    """
    Compute CBF-shaped advantages used by DGPPO.

    The finite-difference CBF term is `(V_{t+1} - V_t) / dt + alpha * V_t`.
    """
    bT_Al_raw = bT_Ql - bT_Vl
    bT_Al_norm = (bT_Al_raw - bT_Al_raw.mean(dim=1, keepdim=True)) / (
        bT_Al_raw.std(dim=1, keepdim=True, unbiased=False) + 1e-8
    )
    bTa_Al = bT_Al_norm[:, :, None].expand(-1, -1, bTah_Vh.shape[2])

    bTah_cbf_deriv = (bTp1ah_Vh[:, 1:] - bTah_Vh) / dt + alpha * bTah_Vh
    bTah_Acbf = torch.clamp(bTah_cbf_deriv + cbf_eps, min=0.0)

    bTa_is_safe = (bTah_cbf_deriv <= 0).all(dim=-1)
    bTa_A = torch.where(bTa_is_safe, bTa_Al, torch.zeros_like(bTa_Al))

    scale = cbf_weight if cbf_scale is None else cbf_scale
    bTa_A = bTa_A + bTah_Acbf.max(dim=-1).values * scale
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
    """Compute clipped PPO policy surrogate."""
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
