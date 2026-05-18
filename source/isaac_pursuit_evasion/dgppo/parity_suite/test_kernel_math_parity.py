from __future__ import annotations

import pytest

from .parity_test_utils import assert_parity_close, importorskip, load_kernel_fixture

torch = importorskip("torch")

from dgppo.update_helpers import _reset_env_carry_after_done, _reset_policy_carry_after_done
from dgppo.utils import (
    compute_cbf_advantages,
    compute_dec_ocp_gae,
    compute_policy_surrogate,
    compute_pos_tracking_safety_costs,
    zero_env_rnn_states_for_done,
    zero_policy_rnn_states_for_done,
)


def test_dec_ocp_gae_matches_jax_kernel_fixture() -> None:
    fixture = load_kernel_fixture()
    cfg = fixture.metadata["kernel_fixture_config"]

    qh, ql = compute_dec_ocp_gae(
        Tah_hs=fixture.tensor("inputs/gae/Tah_hs").unsqueeze(0),
        T_l=fixture.tensor("inputs/gae/T_l").unsqueeze(0),
        Tp1ah_Vh=fixture.tensor("inputs/gae/Tp1ah_Vh").unsqueeze(0),
        Tp1_Vl=fixture.tensor("inputs/gae/Tp1_Vl").unsqueeze(0),
        disc_gamma=float(cfg["disc_gamma"]),
        gae_lambda=float(cfg["gae_lambda"]),
    )

    assert_parity_close(qh.squeeze(0), fixture.arrays["checkpoints/kernel/gae/Qhs"], stage="kernel", tensor_name="Qhs")
    assert_parity_close(ql.squeeze(0), fixture.arrays["checkpoints/kernel/gae/Ql"], stage="kernel", tensor_name="Ql")


def test_ppo_surrogate_matches_jax_kernel_fixture() -> None:
    fixture = load_kernel_fixture()
    ratio = fixture.tensor("inputs/ppo/ratio")
    advantage = fixture.tensor("inputs/ppo/advantage")
    clip_eps = float(fixture.arrays["inputs/ppo/clip_eps"])

    surrogate = compute_policy_surrogate(ratio, advantage, clip_eps)

    assert_parity_close(
        surrogate["loss_policy1"],
        fixture.arrays["checkpoints/kernel/ppo/loss_policy1"],
        stage="kernel",
        tensor_name="loss_policy1",
    )
    assert_parity_close(
        surrogate["loss_policy2"],
        fixture.arrays["checkpoints/kernel/ppo/loss_policy2"],
        stage="kernel",
        tensor_name="loss_policy2",
    )
    assert_parity_close(
        surrogate["loss_policy"],
        fixture.arrays["checkpoints/kernel/ppo/loss_policy"],
        stage="kernel",
        tensor_name="loss_policy",
    )
    assert_parity_close(
        surrogate["clip_frac"],
        fixture.arrays["checkpoints/kernel/ppo/clip_frac"],
        stage="kernel",
        tensor_name="clip_frac",
    )


def test_dec_ocp_gae_masks_true_terminations_without_masking_truncations() -> None:
    Tah_hs = torch.tensor([[[[0.0]], [[0.0]], [[100.0]]]], dtype=torch.float32)
    T_l = torch.tensor([[1.0, 10.0, 100.0]], dtype=torch.float32)
    Tp1ah_Vh = torch.zeros(1, 4, 1, 1)
    Tp1ah_Vh[:, -1] = 1000.0
    Tp1_Vl = torch.zeros(1, 4)
    Tp1_Vl[:, -1] = 1000.0

    qh, ql = compute_dec_ocp_gae(
        Tah_hs=Tah_hs,
        T_l=T_l,
        Tp1ah_Vh=Tp1ah_Vh,
        Tp1_Vl=Tp1_Vl,
        disc_gamma=1.0,
        gae_lambda=1.0,
        T_terminated=torch.tensor([[False, True, False]]),
    )

    assert torch.equal(ql, torch.tensor([[11.0, 10.0, 1100.0]]))
    assert torch.equal(qh.squeeze(-1).squeeze(-1), torch.tensor([[0.0, 0.0, 1000.0]]))

    _, ql_truncated = compute_dec_ocp_gae(
        Tah_hs=Tah_hs,
        T_l=T_l,
        Tp1ah_Vh=Tp1ah_Vh,
        Tp1_Vl=Tp1_Vl,
        disc_gamma=1.0,
        gae_lambda=1.0,
        T_terminated=torch.zeros(1, 3, dtype=torch.bool),
        T_truncated=torch.tensor([[False, True, False]]),
    )
    assert torch.equal(ql_truncated, torch.tensor([[1111.0, 1110.0, 1100.0]]))

    _, ql_truncated_masked = compute_dec_ocp_gae(
        Tah_hs=Tah_hs,
        T_l=T_l,
        Tp1ah_Vh=Tp1ah_Vh,
        Tp1_Vl=Tp1_Vl,
        disc_gamma=1.0,
        gae_lambda=1.0,
        T_truncated=torch.tensor([[False, True, False]]),
        bootstrap_on_truncated=False,
    )
    assert torch.equal(ql_truncated_masked, torch.tensor([[11.0, 10.0, 1100.0]]))


def test_cbf_advantages_do_not_cross_done_boundaries() -> None:
    bT_Ql = torch.zeros(1, 2)
    bT_Vl = torch.zeros(1, 2)
    bTah_Vh = -torch.ones(1, 2, 1, 1)
    bTp1ah_Vh = -torch.ones(1, 3, 1, 1)
    bTp1ah_Vh[:, 1] = 1000.0

    info = compute_cbf_advantages(
        bT_Ql=bT_Ql,
        bT_Vl=bT_Vl,
        bTah_Vh=bTah_Vh,
        bTp1ah_Vh=bTp1ah_Vh,
        alpha=2.0,
        cbf_eps=0.01,
        cbf_weight=1.0,
        dt=0.1,
        bT_done=torch.tensor([[True, False]]),
    )

    assert torch.equal(info["bTah_cbf_deriv"][:, 0], torch.tensor([[[-2.0]]]))
    assert torch.equal(info["bTah_Acbf"][:, 0], torch.zeros(1, 1, 1))


def test_rnn_done_helpers_zero_only_finished_env_slots() -> None:
    policy_state = torch.arange(1 * 6 * 1 * 2, dtype=torch.float32).reshape(1, 6, 1, 2)
    vl_state = torch.arange(1 * 3 * 1 * 2, dtype=torch.float32).reshape(1, 3, 1, 2)
    done = torch.tensor([False, True, False])

    zero_policy_rnn_states_for_done(policy_state, done, n_agents=2)
    zero_env_rnn_states_for_done(vl_state, done)

    assert torch.equal(policy_state.reshape(1, 3, 2, 1, 2)[:, 1], torch.zeros(1, 2, 1, 2))
    assert torch.equal(vl_state[:, 1], torch.zeros(1, 1, 2))
    assert policy_state.reshape(1, 3, 2, 1, 2)[:, 0].abs().sum() > 0
    assert vl_state[:, 0].abs().sum() > 0


def test_recurrent_update_done_masks_zero_post_step_carries_out_of_place() -> None:
    policy_state = torch.arange(1 * 6 * 1 * 2, dtype=torch.float32, requires_grad=True).reshape(1, 6, 1, 2)
    vl_state = torch.arange(1 * 3 * 1 * 2, dtype=torch.float32, requires_grad=True).reshape(1, 3, 1, 2)
    done = torch.tensor([[False], [True], [False]])

    policy_reset = _reset_policy_carry_after_done(policy_state, done, n_agents=2)
    vl_reset = _reset_env_carry_after_done(vl_state, done)

    assert torch.equal(policy_reset.reshape(1, 3, 2, 1, 2)[:, 1], torch.zeros(1, 2, 1, 2))
    assert torch.equal(vl_reset[:, 1], torch.zeros(1, 1, 2))
    assert torch.equal(policy_state.reshape(1, 3, 2, 1, 2)[:, 1], torch.tensor([[[[4.0, 5.0]], [[6.0, 7.0]]]]))
    assert torch.equal(vl_state[:, 1], torch.tensor([[[2.0, 3.0]]]))
    (policy_reset.sum() + vl_reset.sum()).backward()


def test_pos_tracking_safety_costs_emit_vertical_and_pillar_heads() -> None:
    agent_state = torch.zeros(5, 1, 6)
    agent_state[:, 0, :3] = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.2],
            [0.51, 0.0, 1.0],
            [0.0, 0.0, 0.1],
            [2.2, 0.0, 1.0],
        ]
    )
    obs_state = torch.zeros(5, 2, 6)
    obs_state[:, :, :2] = torch.tensor([[0.5, 0.0], [-0.5, 0.0]]).view(1, 2, 2)

    costs = compute_pos_tracking_safety_costs(
        agent_state=agent_state,
        obs_state=obs_state,
        arena_min=(-2.0, -2.0, 0.0),
        arena_max=(2.0, 2.0, 2.0),
        collision_altitude=0.2,
        pillar_collision_radius=0.2,
        pillar_top_z=1.8,
    )

    assert costs.shape == (5, 1, 3)
    assert torch.all(costs[0, 0] < 0.0)
    assert costs[1, 0, 0] > 0.0
    assert costs[2, 0, 1] > 0.0
    assert costs[3, 0, 0] > 0.0
    assert costs[4, 0, 0] < 0.0


def test_pos_tracking_safety_costs_can_reduce_to_nearest_obstacle_head() -> None:
    agent_state = torch.zeros(2, 1, 8)
    agent_state[:, 0, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    obs_state = torch.zeros(2, 3, 8)
    obs_state[:, :, :2] = torch.tensor(
        [
            [[0.25, 0.0], [1.5, 0.0], [2.0, 0.0]],
            [[1.5, 0.0], [2.0, 0.0], [2.5, 0.0]],
        ]
    )

    costs = compute_pos_tracking_safety_costs(
        agent_state=agent_state,
        obs_state=obs_state,
        arena_min=(-2.0, -2.0, 0.0),
        arena_max=(2.0, 2.0, 2.0),
        collision_altitude=0.2,
        pillar_collision_radius=0.3,
        pillar_top_z=1.8,
        obstacle_cost_mode="nearest_obstacle",
    )

    assert costs.shape == (2, 1, 2)
    assert torch.all(costs[:, 0, 0] < 0.0)
    assert costs[0, 0, 1] > 0.0
    assert costs[1, 0, 1] < 0.0


def test_tanh_normal_supports_fixed_noise_sampling() -> None:
    from dgppo.dgppo_models import TanhNormal

    mean = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    std = torch.tensor([[0.5, 1.25]], dtype=torch.float32)
    noise = torch.tensor([[0.3, -0.7]], dtype=torch.float32)

    dist = TanhNormal(mean, std)
    actual = dist.sample(noise=noise)
    expected = torch.tanh(mean + std * noise)

    assert_parity_close(actual, expected, stage="distribution", tensor_name="fixed_noise_action")
