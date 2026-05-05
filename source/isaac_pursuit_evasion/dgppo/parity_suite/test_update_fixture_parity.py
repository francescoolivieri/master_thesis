from __future__ import annotations

import pytest

from .parity_test_utils import assert_parity_close, importorskip, load_update_fixture_for_num_envs

torch = importorskip("torch")

from dgppo.update_helpers import build_update_graph_batch
from dgppo.utils import compute_cbf_advantages, compute_policy_surrogate


@pytest.mark.parametrize("num_envs", [2, 6])
def test_cbf_advantage_matches_jax_update_fixture(num_envs: int) -> None:
    fixture = load_update_fixture_for_num_envs(num_envs)
    cfg = fixture.metadata["config"]

    advantages = compute_cbf_advantages(
        bT_Ql=fixture.tensor("checkpoints/update/gae/bT_Ql"),
        bT_Vl=fixture.tensor("checkpoints/update/value/bT_Vl"),
        bTah_Vh=fixture.tensor("checkpoints/update/value/bTah_Vh"),
        bTp1ah_Vh=fixture.tensor("checkpoints/update/value/bTp1ah_Vh"),
        alpha=float(cfg["alpha"]),
        cbf_eps=float(cfg["cbf_eps"]),
        cbf_weight=float(cfg["cbf_weight"]),
        dt=0.03,
        cbf_scale=float(cfg["cbf_weight"]),
    )

    comparisons = {
        "bT_Al_raw": "checkpoints/update/adv/bT_Al_raw",
        "bT_Al_norm": "checkpoints/update/adv/bT_Al_norm",
        "bTah_cbf_deriv": "checkpoints/update/adv/bTah_cbf_deriv",
        "bTah_Acbf": "checkpoints/update/adv/bTah_Acbf",
        "bTa_is_safe": "checkpoints/update/adv/bTa_is_safe",
        "bTa_A": "checkpoints/update/adv/bTa_A",
    }
    for name, key in comparisons.items():
        assert_parity_close(
            advantages[name],
            fixture.arrays[key],
            stage=f"num_envs={num_envs}/advantage",
            tensor_name=name,
        )


@pytest.mark.parametrize("num_envs", [2, 6])
def test_ppo_surrogate_matches_jax_update_fixture(num_envs: int) -> None:
    fixture = load_update_fixture_for_num_envs(num_envs)
    cfg = fixture.metadata["config"]

    surrogate = compute_policy_surrogate(
        ratio=fixture.tensor("checkpoints/update/policy/ratio"),
        advantages=fixture.tensor("checkpoints/update/adv/bTa_A")[
            :, fixture.tensor("checkpoints/aux/rnn_chunk_ids").long()
        ],
        clip_eps=float(cfg["clip_eps"]),
    )

    assert_parity_close(
        surrogate["loss_policy1"],
        fixture.arrays["checkpoints/update/policy/loss_policy1"],
        stage=f"num_envs={num_envs}/policy",
        tensor_name="loss_policy1",
        atol=1e-5,
        rtol=1e-4,
    )
    assert_parity_close(
        surrogate["loss_policy2"],
        fixture.arrays["checkpoints/update/policy/loss_policy2"],
        stage=f"num_envs={num_envs}/policy",
        tensor_name="loss_policy2",
        atol=1e-5,
        rtol=1e-4,
    )
    assert_parity_close(
        surrogate["loss_policy"],
        fixture.arrays["checkpoints/update/policy/loss_policy"],
        stage=f"num_envs={num_envs}/policy",
        tensor_name="loss_policy",
        atol=1e-5,
        rtol=1e-4,
    )
    assert_parity_close(
        surrogate["clip_frac"],
        fixture.arrays["checkpoints/update/policy/clip_frac"],
        stage=f"num_envs={num_envs}/policy",
        tensor_name="clip_frac",
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.parametrize("num_envs", [2, 6])
def test_build_update_graph_batch_uses_production_graph_builder(num_envs: int) -> None:
    fixture = load_update_fixture_for_num_envs(num_envs)
    B, T, A, S = fixture.arrays["inputs/rollout/graph/8/agent"].shape
    action_dim = fixture.arrays["inputs/rollout/actions"].shape[-1]
    n_cost = fixture.arrays["inputs/rollout/costs"].shape[-1]

    view = {
        "bTa_agent_state": fixture.tensor("inputs/rollout/graph/8/agent"),
        "bTa_goal_state": fixture.tensor("inputs/rollout/graph/8/goal"),
        "bTo_obs_state": torch.zeros(B, T, 0, S),
        "bTa_actions": fixture.tensor("inputs/rollout/actions"),
        "bTa_logp": fixture.tensor("inputs/rollout/log_pis"),
    }
    det_view = view
    idx = torch.arange(B, dtype=torch.long)

    batch = build_update_graph_batch(
        idx=idx,
        view=view,
        det_view=det_view,
        qh_det=fixture.tensor("checkpoints/update/gae/bTah_Qh_det"),
        ql=fixture.tensor("checkpoints/update/gae/bT_Ql"),
        advantages=fixture.tensor("checkpoints/update/adv/bTa_A"),
        obs_radius=2.0,
    )

    assert batch.actions.shape == (B, T, A, action_dim)
    assert batch.qh_det_targets.shape == (B, T, A, n_cost)
    assert batch.graph.n_graphs == B * T
    assert batch.graph.get_type_states(0, A).shape == (B * T, A, S)
