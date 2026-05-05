from __future__ import annotations

import copy

from .parity_test_utils import (
    LoadedFixture,
    assert_parity_close,
    fixture_graph_data,
    importorskip,
    load_drift_trace_for_num_envs,
)

torch = importorskip("torch")

from dgppo.update_helpers import (
    compute_policy_loss_from_log_prob,
    compute_rollout_policy_loss,
    compute_rollout_vh_loss,
    compute_rollout_vl_loss,
    compute_value_l2_loss,
    evaluate_vh_values,
    scan_vl_values,
)
from dgppo.utils import compute_cbf_advantages, compute_dec_ocp_gae

from .parameter_mapping import instantiate_fixture_models, load_fixture_params_into_model


def _step_fixture(drift: LoadedFixture, step: int) -> LoadedFixture:
    prefix = f"trace/{step}/"
    arrays = {key[len(prefix) :]: value for key, value in drift.arrays.items() if key.startswith(prefix)}
    metadata = copy.deepcopy(drift.metadata)
    metadata.setdefault("config", {})["update_step"] = step
    metadata["fixture_type"] = "drift_trace_step"
    return LoadedFixture(path=drift.path, metadata=metadata, arrays=arrays)


def _load_step_models(fixture: LoadedFixture):
    models = instantiate_fixture_models(fixture.metadata, fixture.arrays)
    load_fixture_params_into_model("policy", models.policy, fixture.arrays)
    load_fixture_params_into_model("Vl", models.Vl, fixture.arrays)
    load_fixture_params_into_model("Vh", models.Vh, fixture.arrays)
    models.policy.eval()
    models.Vl.eval()
    models.Vh.eval()
    return models


def test_multi_update_drift_trace_replays_production_losses() -> None:
    drift = load_drift_trace_for_num_envs(6)
    cfg = drift.metadata["config"]
    derived = drift.metadata.get("derived", {})
    n_updates = int(derived.get("n_updates", cfg["n_drift_updates"]))
    num_envs = int(derived.get("num_envs_total", 2 * int(cfg["n_env_train"])))

    assert num_envs == 2 * int(cfg["n_env_train"])
    assert n_updates == int(cfg["n_drift_updates"])

    for step in range(n_updates):
        fixture = _step_fixture(drift, step)
        models = _load_step_models(fixture)
        graph = fixture_graph_data(fixture)
        det_graph = fixture_graph_data(fixture, prefix="inputs/det_rollout/graph")
        chunk_ids = fixture.tensor("checkpoints/aux/rnn_chunk_ids").long()
        B, T, A, _action_dim = fixture.arrays["inputs/rollout/actions"].shape

        assert int(fixture.arrays["step"]) == step
        assert B == int(cfg["n_env_train"])

        bT_Vl = scan_vl_values(Vl=models.Vl, graph=graph, B=B, T=T, A=A)
        bTah_Vh = evaluate_vh_values(
            Vh=models.Vh,
            graph=graph,
            rnn_states=fixture.tensor("inputs/rollout/rnn_states"),
            B=B,
            T=T,
            A=A,
        )
        assert_parity_close(
            bT_Vl,
            fixture.arrays["checkpoints/update/value/bT_Vl"],
            stage=f"drift/step={step}/value_scan",
            tensor_name="bT_Vl",
            atol=1e-5,
            rtol=1e-5,
        )
        assert_parity_close(
            bTah_Vh,
            fixture.arrays["checkpoints/update/value/bTah_Vh"],
            stage=f"drift/step={step}/value_scan",
            tensor_name="bTah_Vh",
            atol=1e-5,
            rtol=1e-5,
        )

        bTp1_Vl = torch.cat([bT_Vl, fixture.tensor("checkpoints/update/value/bTp1_Vl")[:, -1:]], dim=1)
        bTp1ah_Vh = torch.cat([bTah_Vh, fixture.tensor("checkpoints/update/value/bTp1ah_Vh")[:, -1:]], dim=1)
        _bTah_Qh, bT_Ql = compute_dec_ocp_gae(
            Tah_hs=fixture.tensor("inputs/rollout/costs"),
            T_l=-fixture.tensor("inputs/rollout/rewards"),
            Tp1ah_Vh=bTp1ah_Vh,
            Tp1_Vl=bTp1_Vl,
            disc_gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
        )
        adv_info = compute_cbf_advantages(
            bT_Ql=bT_Ql,
            bT_Vl=bT_Vl,
            bTah_Vh=bTah_Vh,
            bTp1ah_Vh=bTp1ah_Vh,
            alpha=float(cfg["alpha"]),
            cbf_eps=float(cfg["cbf_eps"]),
            cbf_weight=float(cfg["cbf_weight"]),
            dt=0.03,
            cbf_scale=float(fixture.tensor("checkpoints/update/adv/cbf_scale")),
        )
        assert_parity_close(
            bT_Ql,
            fixture.arrays["checkpoints/update/gae/bT_Ql"],
            stage=f"drift/step={step}/gae",
            tensor_name="bT_Ql",
            atol=1e-5,
            rtol=1e-4,
        )
        assert_parity_close(
            adv_info["bTa_A"],
            fixture.arrays["checkpoints/update/adv/bTa_A"],
            stage=f"drift/step={step}/advantage",
            tensor_name="bTa_A",
            atol=5e-5,
            rtol=2e-3,
        )

        policy_info = compute_rollout_policy_loss(
            policy=models.policy,
            graph=graph,
            actions=fixture.tensor("inputs/rollout/actions"),
            old_logp=fixture.tensor("inputs/rollout/log_pis"),
            advantages=fixture.tensor("checkpoints/update/adv/bTa_A"),
            chunk_ids=chunk_ids,
            clip_eps=float(cfg["clip_eps"]),
            entropy_scale=0.0,
            n_agents=A,
        )
        assert_parity_close(
            policy_info["loss_policy"],
            fixture.arrays["checkpoints/update/policy/loss_policy"],
            stage=f"drift/step={step}/policy",
            tensor_name="loss_policy",
            atol=1e-5,
            rtol=1e-4,
        )
        entropy_scaled = compute_policy_loss_from_log_prob(
            log_prob=policy_info["log_prob"],
            old_logp=fixture.tensor("inputs/rollout/log_pis")[:, chunk_ids],
            advantages=fixture.tensor("checkpoints/update/adv/bTa_A")[:, chunk_ids],
            entropy=torch.full_like(policy_info["entropy"], float(fixture.tensor("checkpoints/update/policy/entropy"))),
            clip_eps=float(cfg["clip_eps"]),
            entropy_scale=float(cfg["coef_ent"]),
        )
        assert_parity_close(
            entropy_scaled["loss_policy_total"],
            # Compare against the exported pre-update batch loss. For the
            # num_envs=6 drift artifact, metrics/policy_loss is the logged
            # post-update last-minibatch scalar from the JAX update loop, not
            # the full-batch pre-update quantity reconstructed here.
            fixture.arrays["checkpoints/update/policy/policy_loss"],
            stage=f"drift/step={step}/policy",
            tensor_name="policy_loss_with_exported_entropy",
            atol=1e-5,
            rtol=1e-4,
        )

        # The drift trace exports these under update/metrics/* as logged
        # post-update last-minibatch values from the JAX PPO loop. This test
        # replays pre-update full-batch quantities from the checkpointed
        # parameters, so those metrics are not comparable here without also
        # replaying the optimizer loop minibatch-by-minibatch.
        compute_rollout_vl_loss(
            Vl=models.Vl,
            graph=graph,
            targets=fixture.tensor("checkpoints/update/gae/bT_Ql"),
            chunk_ids=chunk_ids,
            A=A,
        )
        compute_rollout_vh_loss(
            Vh=models.Vh,
            graph=det_graph,
            rnn_states=fixture.tensor("inputs/det_rollout/rnn_states"),
            targets=fixture.tensor("checkpoints/update/gae/bTah_Qh_det"),
            chunk_ids=chunk_ids,
            A=A,
        )

        assert_parity_close(
            compute_value_l2_loss(bT_Vl, bT_Ql),
            fixture.arrays["checkpoints/update/loss/Vl_global"],
            stage=f"drift/step={step}/global_value",
            tensor_name="Vl_global",
            atol=1e-5,
            rtol=1e-4,
        )
        assert_parity_close(
            compute_value_l2_loss(
                fixture.tensor("checkpoints/update/value/bTah_Vh_det"),
                fixture.tensor("checkpoints/update/gae/bTah_Qh_det"),
            ),
            fixture.arrays["checkpoints/update/loss/Vh_det_global"],
            stage=f"drift/step={step}/global_value",
            tensor_name="Vh_det_global",
            atol=1e-5,
            rtol=1e-4,
        )
