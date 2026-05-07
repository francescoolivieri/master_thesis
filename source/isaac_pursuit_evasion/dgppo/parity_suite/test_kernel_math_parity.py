from __future__ import annotations

from .parity_test_utils import assert_parity_close, importorskip, load_kernel_fixture

torch = importorskip("torch")

from dgppo.utils import compute_dec_ocp_gae, compute_policy_surrogate


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


def test_dec_ocp_gae_respects_episode_boundaries() -> None:
    qh, ql = compute_dec_ocp_gae(
        Tah_hs=torch.zeros(1, 2, 1, 1),
        T_l=torch.tensor([[1.0, 10.0]]),
        Tp1ah_Vh=torch.zeros(1, 3, 1, 1),
        Tp1_Vl=torch.tensor([[0.0, 7.0, 100.0]]),
        disc_gamma=0.9,
        gae_lambda=0.5,
        T_done=torch.tensor([[True, False]]),
    )

    assert_parity_close(qh, torch.zeros_like(qh), stage="kernel/dones", tensor_name="Qh")
    assert_parity_close(ql[:, 0], torch.tensor([1.0]), stage="kernel/dones", tensor_name="Ql_done")


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


def test_tanh_normal_supports_fixed_noise_sampling() -> None:
    from dgppo.dgppo_models import TanhNormal

    mean = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    std = torch.tensor([[0.5, 1.25]], dtype=torch.float32)
    noise = torch.tensor([[0.3, -0.7]], dtype=torch.float32)

    dist = TanhNormal(mean, std)
    actual = dist.sample(noise=noise)
    expected = torch.tanh(mean + std * noise)

    assert_parity_close(actual, expected, stage="distribution", tensor_name="fixed_noise_action")
