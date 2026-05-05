from __future__ import annotations

import pytest

from .parity_test_utils import (
    assert_parity_close,
    fixture_graph_data,
    importorskip,
    load_update_fixture_for_num_envs,
)

torch = importorskip("torch")

from dgppo.update_helpers import (
    compute_policy_loss_from_log_prob,
    compute_rollout_policy_loss,
    compute_rollout_vl_loss,
    compute_value_l2_loss,
    evaluate_vh_values,
    scan_policy_rnn_states,
    scan_vl_values,
)
from dgppo.utils import compute_cbf_advantages, compute_dec_ocp_gae

from .parameter_mapping import (
    ParameterMappingError,
    expected_tensor_for_entry,
    instantiate_fixture_models,
    load_fixture_params_into_model,
    map_fixture_params_to_state_dict,
)

_NUM_ENVS_WITH_UPDATE_FIXTURES = [2, 6]

_FORWARD_PARITY_KEYS = (
    "inputs/rollout/actions",
    "inputs/rollout/rnn_states",
    "inputs/rollout/graph/0",
    "inputs/rollout/graph/1",
    "inputs/rollout/graph/2",
    "inputs/rollout/graph/3",
    "inputs/rollout/graph/4",
    "inputs/rollout/graph/5",
    "inputs/rollout/graph/6",
    "inputs/rollout/graph/7",
    "checkpoints/actor/rollout/mean",
    "checkpoints/actor/rollout/std",
    "checkpoints/actor/rollout/mode",
    "checkpoints/actor/rollout/log_prob",
    "checkpoints/actor/rollout/fixed_noise",
    "checkpoints/actor/rollout/fixed_noise_action",
    "checkpoints/update/value/bT_Vl",
    "checkpoints/update/value/bTah_Vh",
)


def _prepared_fixture_models(num_envs: int = 2):
    fixture = load_update_fixture_for_num_envs(num_envs)
    if fixture.metadata["config"].get("use_lstm", False):
        pytest.xfail("LSTM forward parity pending GRU/LSTM gate-order parity")

    try:
        models = instantiate_fixture_models(fixture.metadata, fixture.arrays)
    except ParameterMappingError as exc:
        if "GRU/LSTM gate-order parity" in str(exc):
            pytest.xfail("RNN forward parity pending GRU gate-order and scan-state parity")
        raise

    load_fixture_params_into_model("policy", models.policy, fixture.arrays)
    load_fixture_params_into_model("Vl", models.Vl, fixture.arrays)
    load_fixture_params_into_model("Vh", models.Vh, fixture.arrays)
    models.policy.eval()
    models.Vl.eval()
    models.Vh.eval()
    return fixture, models


def _policy_rnn_state(fixture, b: int, t: int, *, use_rnn: bool) -> torch.Tensor | None:
    if not use_rnn:
        return None
    # RNN state is [L, A, C, H] for one B,T graph.
    return fixture.tensor("inputs/rollout/rnn_states")[b, t]


def _det_policy_rnn_state(fixture, b: int, t: int, *, use_rnn: bool) -> torch.Tensor | None:
    if not use_rnn:
        return None
    # RNN state is [L, A, C, H] for one B,T deterministic graph.
    return fixture.tensor("inputs/det_rollout/rnn_states")[b, t]


def _iter_bt(B: int, T: int):
    for b in range(B):
        for t in range(T):
            yield b, t


def _det_rollout_required_keys(include_next_graph: bool = True) -> tuple[str, ...]:
    graph_keys = tuple(f"inputs/det_rollout/graph/{leaf_id}" for leaf_id in range(8))
    next_graph_keys = tuple(f"inputs/det_rollout/next_graph/{leaf_id}" for leaf_id in range(8))
    rollout_keys = (
        "inputs/det_rollout/actions",
        "inputs/det_rollout/rewards",
        "inputs/det_rollout/costs",
        "inputs/det_rollout/rnn_states",
        "checkpoints/update/value/bTah_Vh_det",
        "checkpoints/update/value/bTp1ah_Vh_det",
        "checkpoints/update/gae/bTah_Qh_det",
        "checkpoints/update/loss/Vh_det_global",
    )
    return graph_keys + rollout_keys + (next_graph_keys if include_next_graph else ())


def _compute_det_vh_forward_and_targets(fixture, models, *, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = fixture.metadata["config"]
    B, T, A, _action_dim = fixture.arrays["inputs/det_rollout/actions"].shape
    _B, _Tp1, _A, n_cost = fixture.arrays["checkpoints/update/value/bTp1ah_Vh_det"].shape

    det_graph = fixture_graph_data(fixture, prefix="inputs/det_rollout/graph")
    bTah_Vh_det = evaluate_vh_values(
        Vh=models.Vh,
        graph=det_graph,
        rnn_states=fixture.tensor("inputs/det_rollout/rnn_states"),
        B=B,
        T=T,
        A=A,
    )
    assert_parity_close(
        bTah_Vh_det,
        fixture.arrays["checkpoints/update/value/bTah_Vh_det"],
        stage=f"num_envs={num_envs}/det_replay/value_scan",
        tensor_name="bTah_Vh_det",
        atol=1e-5,
        rtol=1e-5,
    )

    final_Vh_det = bTah_Vh_det.new_empty((B, A, n_cost))
    with torch.no_grad():
        for b in range(B):
            final_graph = fixture_graph_data(fixture, index=(b, T - 1), prefix="inputs/det_rollout/next_graph")
            rnn_state = _det_policy_rnn_state(fixture, b, T - 1, use_rnn=models.spec.use_rnn)
            _action, _log_prob, _mode, final_rnn_state = models.policy.act(
                final_graph,
                rnn_state,
                A,
                deterministic=True,
            )
            value, _ = models.Vh(final_graph, final_rnn_state, A)
            final_Vh_det[b] = value

    bTp1ah_Vh_det = torch.cat([bTah_Vh_det, final_Vh_det[:, None]], dim=1)
    assert_parity_close(
        bTp1ah_Vh_det,
        fixture.arrays["checkpoints/update/value/bTp1ah_Vh_det"],
        stage=f"num_envs={num_envs}/det_replay/value_scan",
        tensor_name="bTp1ah_Vh_det",
        atol=1e-5,
        rtol=1e-5,
    )

    bTah_Qh_det, _ = compute_dec_ocp_gae(
        Tah_hs=fixture.tensor("inputs/det_rollout/costs"),
        T_l=-fixture.tensor("inputs/det_rollout/rewards"),
        Tp1ah_Vh=bTp1ah_Vh_det,
        Tp1_Vl=fixture.tensor("checkpoints/update/value/bTp1_Vl"),
        disc_gamma=float(cfg["gamma"]),
        gae_lambda=float(cfg["gae_lambda"]),
    )
    assert_parity_close(
        bTah_Qh_det,
        fixture.arrays["checkpoints/update/gae/bTah_Qh_det"],
        stage=f"num_envs={num_envs}/det_replay/gae",
        tensor_name="bTah_Qh_det",
        atol=1e-5,
        rtol=1e-4,
    )

    assert_parity_close(
        compute_value_l2_loss(bTah_Vh_det, bTah_Qh_det),
        fixture.arrays["checkpoints/update/loss/Vh_det_global"],
        stage=f"num_envs={num_envs}/det_replay/value_loss",
        tensor_name="Vh_det_global",
        atol=1e-5,
        rtol=1e-4,
    )
    return bTah_Vh_det, bTah_Qh_det


def _value_optimizer_parameters(model) -> list[torch.nn.Parameter]:
    params = list(model.gnn.parameters()) + list(model.head.parameters()) + list(model.net.value_out.parameters())
    if model.rnn is not None:
        params += list(model.rnn.parameters())
    return params


def _apply_first_torch_adam_step(
    *,
    parameters: list[torch.nn.Parameter],
    loss: torch.Tensor,
    learning_rate: float,
    max_grad_norm: float,
) -> torch.Tensor:
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    for param in parameters:
        param.grad = None
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()
    return torch.as_tensor(grad_norm, device=loss.device, dtype=loss.dtype)


def _mapped_model_delta_norm(model_name: str, model, before: dict[str, torch.Tensor], fixture) -> torch.Tensor:
    """Compute a JAX-tree-like delta norm, avoiding duplicate PyTorch module aliases."""
    _mapped_state, report = map_fixture_params_to_state_dict(model_name, model, fixture.arrays)
    after = model.state_dict()
    seen_sources: set[tuple[str, ...]] = set()
    total = None
    for entry in report.entries:
        if entry.source_keys in seen_sources:
            continue
        seen_sources.add(entry.source_keys)
        delta_sq = (after[entry.target_key].detach() - before[entry.target_key]).square().sum()
        total = delta_sq if total is None else total + delta_sq
    if total is None:
        raise ValueError(f"{model_name}: no mapped parameters for delta norm")
    return torch.sqrt(total)


def _assert_delta_norm_contract(
    *,
    fixture,
    model_name: str,
    actual: torch.Tensor,
    max_rel_drift: float,
) -> None:
    expected = fixture.tensor(f"checkpoints/update/param_delta/{model_name}")
    rel_drift = torch.abs(actual - expected) / torch.clamp(expected.abs(), min=1e-12)
    assert float(rel_drift) < max_rel_drift, (
        f"{model_name} optimizer delta drift too large: "
        f"actual={float(actual)} expected={float(expected)} rel={float(rel_drift)}"
    )


def test_forward_parity_fixture_keys_exist() -> None:
    fixture = load_update_fixture_for_num_envs(2)
    missing = [key for key in _FORWARD_PARITY_KEYS if key not in fixture.arrays]
    assert not missing, f"update fixture is missing forward parity keys: {missing}"


def test_parameter_mapping_parity() -> None:
    fixture = load_update_fixture_for_num_envs(2)
    if fixture.metadata["config"].get("use_lstm", False):
        pytest.xfail("RNN parameter mapping pending GRU/LSTM gate-order parity")

    try:
        models = instantiate_fixture_models(fixture.metadata, fixture.arrays)
    except ParameterMappingError as exc:
        if "GRU/LSTM gate-order parity" in str(exc):
            pytest.xfail("RNN parameter mapping pending GRU/LSTM gate-order parity")
        raise

    reports = {
        "policy": load_fixture_params_into_model("policy", models.policy, fixture.arrays),
        "Vl": load_fixture_params_into_model("Vl", models.Vl, fixture.arrays),
        "Vh": load_fixture_params_into_model("Vh", models.Vh, fixture.arrays),
    }
    state_dicts = {
        "policy": models.policy.state_dict(),
        "Vl": models.Vl.state_dict(),
        "Vh": models.Vh.state_dict(),
    }

    for model_name, report in reports.items():
        assert (
            not report.missing_target_keys
        ), f"{model_name}: unmapped PyTorch state_dict keys: {report.missing_target_keys}"
        assert (
            not report.unmapped_source_keys
        ), f"{model_name}: unused JAX fixture parameter keys: {report.unmapped_source_keys}"
        for entry in report.entries:
            assert_parity_close(
                state_dicts[model_name][entry.target_key],
                expected_tensor_for_entry(entry, fixture.arrays),
                stage=f"{model_name}/parameter_mapping",
                tensor_name=entry.target_key,
            )


def test_actor_distribution_mean_std_mode_parity() -> None:
    fixture, models = _prepared_fixture_models()

    # B: rollout envs, T: horizon, A: agents, U: action dim.
    B, T, A, U = fixture.arrays["inputs/rollout/actions"].shape
    bTaU_mean = torch.empty((B, T, A, U), dtype=fixture.tensor("checkpoints/actor/rollout/mean").dtype)
    bTaU_std = torch.empty_like(bTaU_mean)
    bTaU_mode = torch.empty_like(bTaU_mean)

    with torch.no_grad():
        for b, t in _iter_bt(B, T):
            graph = fixture_graph_data(fixture, index=(b, t))
            rnn_state = _policy_rnn_state(fixture, b, t, use_rnn=models.spec.use_rnn)
            dist, _ = models.policy.distribution(graph, rnn_state, A)
            bTaU_mean[b, t] = dist.mean
            bTaU_std[b, t] = dist.std
            bTaU_mode[b, t] = dist.mode()

    assert_parity_close(
        bTaU_mean,
        fixture.arrays["checkpoints/actor/rollout/mean"],
        stage="actor_distribution",
        tensor_name="bTaU_mean",
        atol=1e-5,
        rtol=1e-5,
    )
    assert_parity_close(
        bTaU_std,
        fixture.arrays["checkpoints/actor/rollout/std"],
        stage="actor_distribution",
        tensor_name="bTaU_std",
        atol=1e-5,
        rtol=1e-5,
    )
    assert_parity_close(
        bTaU_mode,
        fixture.arrays["checkpoints/actor/rollout/mode"],
        stage="actor_distribution",
        tensor_name="bTaU_mode",
        atol=1e-5,
        rtol=1e-5,
    )


def test_policy_log_prob_parity() -> None:
    fixture, models = _prepared_fixture_models()

    # B, T, A follow rollout shape; bTa_log_prob is [B, T, A].
    B, T, A, _ = fixture.arrays["inputs/rollout/actions"].shape
    bTa_log_prob = torch.empty((B, T, A), dtype=fixture.tensor("checkpoints/actor/rollout/log_prob").dtype)

    with torch.no_grad():
        for b, t in _iter_bt(B, T):
            graph = fixture_graph_data(fixture, index=(b, t))
            bTa_action = fixture.tensor("inputs/rollout/actions")[b, t]
            rnn_state = _policy_rnn_state(fixture, b, t, use_rnn=models.spec.use_rnn)
            log_prob, _, _ = models.policy.evaluate(graph, bTa_action, rnn_state, A)
            bTa_log_prob[b, t] = log_prob

    assert_parity_close(
        bTa_log_prob,
        fixture.arrays["checkpoints/actor/rollout/log_prob"],
        stage="policy_evaluate",
        tensor_name="bTa_log_prob",
        atol=1e-5,
        rtol=1e-5,
    )


def test_sampled_action_with_fixed_noise_parity() -> None:
    fixture, models = _prepared_fixture_models()

    # B, T, A, U: fixed noise and sampled action are both [B, T, A, U].
    B, T, A, U = fixture.arrays["checkpoints/actor/rollout/fixed_noise"].shape
    bTaU_action = torch.empty((B, T, A, U), dtype=fixture.tensor("checkpoints/actor/rollout/fixed_noise_action").dtype)

    with torch.no_grad():
        for b, t in _iter_bt(B, T):
            graph = fixture_graph_data(fixture, index=(b, t))
            bTaU_noise = fixture.tensor("checkpoints/actor/rollout/fixed_noise")[b, t]
            rnn_state = _policy_rnn_state(fixture, b, t, use_rnn=models.spec.use_rnn)
            dist, _ = models.policy.distribution(graph, rnn_state, A)
            bTaU_action[b, t] = dist.sample(noise=bTaU_noise)

    assert_parity_close(
        bTaU_action,
        fixture.arrays["checkpoints/actor/rollout/fixed_noise_action"],
        stage="actor_distribution",
        tensor_name="bTaU_fixed_noise_action",
        atol=1e-5,
        rtol=1e-5,
    )


def test_value_forward_parity() -> None:
    fixture, models = _prepared_fixture_models()

    # B: rollout envs, T: horizon, A: agents, NH: constraint heads.
    B, T = fixture.arrays["checkpoints/update/value/bT_Vl"].shape
    _, _, A, NH = fixture.arrays["checkpoints/update/value/bTah_Vh"].shape
    bT_Vl = torch.empty((B, T), dtype=fixture.tensor("checkpoints/update/value/bT_Vl").dtype)
    bTah_Vh = torch.empty((B, T, A, NH), dtype=fixture.tensor("checkpoints/update/value/bTah_Vh").dtype)

    with torch.no_grad():
        for b in range(B):
            Vl_rnn_state = models.Vl.rnn.initialize_carry(1) if models.Vl.rnn is not None else None
            for t in range(T):
                graph = fixture_graph_data(fixture, index=(b, t))

                Vl, Vl_rnn_state = models.Vl(graph, Vl_rnn_state, A)
                bT_Vl[b, t] = Vl.squeeze(0).squeeze(-1)

                Vh_rnn_state = _policy_rnn_state(fixture, b, t, use_rnn=models.spec.use_rnn)
                Vh, _ = models.Vh(graph, Vh_rnn_state, A)
                bTah_Vh[b, t] = Vh

    assert_parity_close(
        bT_Vl,
        fixture.arrays["checkpoints/update/value/bT_Vl"],
        stage="value_forward",
        tensor_name="bT_Vl",
        atol=1e-5,
        rtol=1e-5,
    )
    assert_parity_close(
        bTah_Vh,
        fixture.arrays["checkpoints/update/value/bTah_Vh"],
        stage="value_forward",
        tensor_name="bTah_Vh",
        atol=1e-5,
        rtol=1e-5,
    )


def test_full_loss_before_optimizer_step_parity() -> None:
    fixture, models = _prepared_fixture_models()
    cfg = fixture.metadata["config"]
    graph = fixture_graph_data(fixture)

    # B: rollout envs, T: horizon, A: agents.
    B, T, A, _ = fixture.arrays["inputs/rollout/actions"].shape
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
        stage="pre_update_loss/value_scan",
        tensor_name="bT_Vl",
        atol=1e-5,
        rtol=1e-5,
    )
    assert_parity_close(
        bTah_Vh,
        fixture.arrays["checkpoints/update/value/bTah_Vh"],
        stage="pre_update_loss/value_scan",
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
    advantages = compute_cbf_advantages(
        bT_Ql=bT_Ql,
        bT_Vl=bT_Vl,
        bTah_Vh=bTah_Vh,
        bTp1ah_Vh=bTp1ah_Vh,
        alpha=float(cfg["alpha"]),
        cbf_eps=float(cfg["cbf_eps"]),
        cbf_weight=float(cfg["cbf_weight"]),
        dt=0.03,
        cbf_scale=float(cfg["cbf_weight"]),
    )

    assert_parity_close(
        bT_Ql,
        fixture.arrays["checkpoints/update/gae/bT_Ql"],
        stage="pre_update_loss/gae",
        tensor_name="bT_Ql",
        atol=1e-5,
        rtol=1e-4,
    )
    assert_parity_close(
        advantages["bTa_A"],
        fixture.arrays["checkpoints/update/adv/bTa_A"],
        stage="pre_update_loss/advantage",
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
        chunk_ids=fixture.tensor("checkpoints/aux/rnn_chunk_ids").long(),
        clip_eps=float(cfg["clip_eps"]),
        entropy_scale=float(cfg["coef_ent"]),
        n_agents=A,
    )
    policy_comparisons = {
        "ratio": "checkpoints/update/policy/ratio",
        "loss_policy1": "checkpoints/update/policy/loss_policy1",
        "loss_policy2": "checkpoints/update/policy/loss_policy2",
        "loss_policy": "checkpoints/update/policy/loss_policy",
        "clip_frac": "checkpoints/update/policy/clip_frac",
    }
    for name, key in policy_comparisons.items():
        assert_parity_close(
            policy_info[name],
            fixture.arrays[key],
            stage="pre_update_loss/policy",
            tensor_name=name,
            atol=1e-5,
            rtol=1e-4,
        )
    entropy_scaled = compute_policy_loss_from_log_prob(
        log_prob=policy_info["log_prob"],
        old_logp=fixture.tensor("inputs/rollout/log_pis")[:, fixture.tensor("checkpoints/aux/rnn_chunk_ids").long()],
        advantages=fixture.tensor("checkpoints/update/adv/bTa_A")[
            :, fixture.tensor("checkpoints/aux/rnn_chunk_ids").long()
        ],
        entropy=torch.full_like(policy_info["entropy"], float(fixture.tensor("checkpoints/update/policy/entropy"))),
        clip_eps=float(cfg["clip_eps"]),
        entropy_scale=float(cfg["coef_ent"]),
    )
    assert_parity_close(
        entropy_scaled["loss_policy_total"],
        fixture.arrays["checkpoints/update/policy/policy_loss"],
        stage="pre_update_loss/policy",
        tensor_name="policy_loss",
        atol=1e-5,
        rtol=1e-4,
    )

    assert_parity_close(
        compute_value_l2_loss(bT_Vl, bT_Ql),
        fixture.arrays["checkpoints/update/loss/Vl_global"],
        stage="pre_update_loss/value",
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
        stage="pre_update_loss/value",
        tensor_name="Vh_det_global",
        atol=1e-5,
        rtol=1e-4,
    )


def test_value_gradient_norm_before_optimizer_step_parity() -> None:
    fixture, models = _prepared_fixture_models()
    cfg = fixture.metadata["config"]
    graph = fixture_graph_data(fixture)
    B, _T, A, _ = fixture.arrays["inputs/rollout/actions"].shape

    info = compute_rollout_vl_loss(
        Vl=models.Vl,
        graph=graph,
        targets=fixture.tensor("checkpoints/update/gae/bT_Ql"),
        chunk_ids=fixture.tensor("checkpoints/aux/rnn_chunk_ids").long(),
        A=A,
    )
    assert_parity_close(
        info["loss_vl"],
        fixture.arrays["checkpoints/update/metrics/Vl_loss"],
        stage="optimizer_pre_step/Vl",
        tensor_name="Vl_loss",
        atol=1e-5,
        rtol=1e-4,
    )

    params = (
        list(models.Vl.gnn.parameters())
        + list(models.Vl.head.parameters())
        + list(models.Vl.net.value_out.parameters())
    )
    if models.Vl.rnn is not None:
        params += list(models.Vl.rnn.parameters())
    for param in params:
        param.grad = None
    info["loss_vl"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 2.0)))

    assert B == int(cfg["n_env_train"])
    assert_parity_close(
        grad_norm,
        fixture.arrays["checkpoints/update/metrics/Vl_grad_norm"],
        stage="optimizer_pre_step/Vl",
        tensor_name="Vl_grad_norm",
        atol=1e-5,
        rtol=1e-4,
    )


def test_vh_gradient_norm_before_optimizer_step_parity() -> None:
    fixture, models = _prepared_fixture_models()
    cfg = fixture.metadata["config"]
    chunk_ids = fixture.tensor("checkpoints/aux/rnn_chunk_ids").long()
    targets = fixture.tensor("checkpoints/update/gae/bTah_Qh_det")
    B, _T, A, n_cost = targets.shape
    C, R = chunk_ids.shape
    values = targets.new_empty((B, C, R, A, n_cost))

    for b in range(B):
        for c in range(C):
            for r in range(R):
                t = int(chunk_ids[c, r].item())
                graph = fixture_graph_data(fixture, index=(b, t), prefix="inputs/det_rollout/graph")
                rnn_state = fixture.tensor("inputs/det_rollout/rnn_states")[b, t]
                value, _ = models.Vh(graph, rnn_state, A)
                values[b, c, r] = value

    loss_vh = compute_value_l2_loss(values, targets[:, chunk_ids])
    assert_parity_close(
        loss_vh,
        fixture.arrays["checkpoints/update/metrics/Vh_loss"],
        stage="optimizer_pre_step/Vh",
        tensor_name="Vh_loss",
        atol=1e-5,
        rtol=1e-4,
    )

    params = (
        list(models.Vh.gnn.parameters())
        + list(models.Vh.head.parameters())
        + list(models.Vh.net.value_out.parameters())
    )
    if models.Vh.rnn is not None:
        params += list(models.Vh.rnn.parameters())
    for param in params:
        param.grad = None
    loss_vh.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 2.0)))

    assert B == int(cfg["n_env_train"])
    assert_parity_close(
        grad_norm,
        fixture.arrays["checkpoints/update/metrics/Vh_grad_Vh_norm"],
        stage="optimizer_pre_step/Vh",
        tensor_name="Vh_grad_Vh_norm",
        atol=1e-5,
        rtol=5e-4,
    )


def test_optimizer_step_fixture_contract_keys_exist() -> None:
    fixture = load_update_fixture_for_num_envs(2)
    required = (
        "checkpoints/update/metrics/policy_grad_norm",
        "checkpoints/update/metrics/Vl_grad_norm",
        "checkpoints/update/metrics/Vh_grad_Vh_norm",
        "checkpoints/update/param_delta/policy",
        "checkpoints/update/param_delta/Vl",
        "checkpoints/update/param_delta/Vh",
        "inputs/optimizer_state_before_update/policy/notfinite_count",
        "inputs/optimizer_state_before_update/policy/last_finite",
        "inputs/optimizer_state_before_update/policy/total_notfinite",
        "inputs/optimizer_state_before_update/policy/inner_state/0/count",
        "inputs/optimizer_state_before_update/Vl/notfinite_count",
        "inputs/optimizer_state_before_update/Vl/last_finite",
        "inputs/optimizer_state_before_update/Vl/total_notfinite",
        "inputs/optimizer_state_before_update/Vl/inner_state/0/count",
        "inputs/optimizer_state_before_update/Vh/notfinite_count",
        "inputs/optimizer_state_before_update/Vh/last_finite",
        "inputs/optimizer_state_before_update/Vh/total_notfinite",
        "inputs/optimizer_state_before_update/Vh/inner_state/0/count",
    )
    missing = [key for key in required if key not in fixture.arrays]
    assert not missing, f"update fixture is missing optimizer-step keys: {missing}"

    for model_name in ("policy", "Vl", "Vh"):
        assert int(fixture.arrays[f"inputs/optimizer_state_before_update/{model_name}/notfinite_count"]) == 0
        assert bool(fixture.arrays[f"inputs/optimizer_state_before_update/{model_name}/last_finite"])
        assert int(fixture.arrays[f"inputs/optimizer_state_before_update/{model_name}/total_notfinite"]) == 0
        assert int(fixture.arrays[f"inputs/optimizer_state_before_update/{model_name}/inner_state/0/count"]) == 0

        mu_keys = [
            key
            for key in fixture.arrays
            if key.startswith(f"inputs/optimizer_state_before_update/{model_name}/inner_state/0/mu/")
        ]
        nu_keys = [
            key
            for key in fixture.arrays
            if key.startswith(f"inputs/optimizer_state_before_update/{model_name}/inner_state/0/nu/")
        ]
        assert mu_keys, f"{model_name}: missing Optax Adam first-moment leaves"
        assert len(mu_keys) == len(nu_keys), f"{model_name}: Adam mu/nu leaf count mismatch"

    optim_meta = fixture.metadata.get("optimizer", {})
    for model_name, lr_key in (("policy", "lr_actor"), ("Vl", "lr_Vl"), ("Vh", "lr_Vh")):
        meta = optim_meta.get(model_name, {})
        assert meta.get("type") == "optax.apply_if_finite(optax.adam)"
        assert meta.get("b1") == pytest.approx(0.9)
        assert meta.get("b2") == pytest.approx(0.999)
        assert meta.get("eps") == pytest.approx(1e-8)
        assert meta.get("eps_root") == pytest.approx(0.0)
        assert meta.get("learning_rate") == pytest.approx(float(fixture.metadata["config"][lr_key]))


def test_one_optimizer_step_parity() -> None:
    fixture, models = _prepared_fixture_models(2)
    cfg = fixture.metadata["config"]
    optimizer_state_keys = [key for key in fixture.arrays if key.startswith("inputs/optimizer_state_before_update/")]
    assert optimizer_state_keys, "update fixture is missing exported Optax optimizer state"

    max_grad_norm = float(cfg.get("max_grad_norm", 2.0))
    B, _T, A, _action_dim = fixture.arrays["inputs/rollout/actions"].shape
    chunk_ids = fixture.tensor("checkpoints/aux/rnn_chunk_ids").long()

    # The first-step replay uses the real PyTorch losses/gradients and
    # torch.optim.Adam with the exported Optax hyperparameters. Losses and
    # global grad norms are strict; post-step deltas are kept as an explicit
    # PyTorch/Optax drift contract because the JAX fixture does not export
    # per-leaf clipped gradients.
    graph = fixture_graph_data(fixture)
    vl_before = {key: value.detach().clone() for key, value in models.Vl.state_dict().items()}
    vl_loss = compute_rollout_vl_loss(
        Vl=models.Vl,
        graph=graph,
        targets=fixture.tensor("checkpoints/update/gae/bT_Ql"),
        chunk_ids=chunk_ids,
        A=A,
    )["loss_vl"]
    vl_grad_norm = _apply_first_torch_adam_step(
        parameters=_value_optimizer_parameters(models.Vl),
        loss=vl_loss,
        learning_rate=float(cfg["lr_Vl"]),
        max_grad_norm=max_grad_norm,
    )
    assert_parity_close(
        vl_loss,
        fixture.arrays["checkpoints/update/metrics/Vl_loss"],
        stage="optimizer_step/Vl",
        tensor_name="loss",
        atol=1e-5,
        rtol=1e-4,
    )
    assert_parity_close(
        vl_grad_norm,
        fixture.arrays["checkpoints/update/metrics/Vl_grad_norm"],
        stage="optimizer_step/Vl",
        tensor_name="grad_norm",
        atol=1e-5,
        rtol=1e-4,
    )
    _assert_delta_norm_contract(
        fixture=fixture,
        model_name="Vl",
        actual=_mapped_model_delta_norm("Vl", models.Vl, vl_before, fixture),
        max_rel_drift=0.05,
    )

    vh_before = {key: value.detach().clone() for key, value in models.Vh.state_dict().items()}
    targets = fixture.tensor("checkpoints/update/gae/bTah_Qh_det")
    _B, _T, _A, n_cost = targets.shape
    C, R = chunk_ids.shape
    vh_values = targets.new_empty((B, C, R, A, n_cost))
    for b in range(B):
        for c in range(C):
            for r in range(R):
                t = int(chunk_ids[c, r].item())
                det_graph = fixture_graph_data(fixture, index=(b, t), prefix="inputs/det_rollout/graph")
                value, _ = models.Vh(det_graph, fixture.tensor("inputs/det_rollout/rnn_states")[b, t], A)
                vh_values[b, c, r] = value
    vh_loss = compute_value_l2_loss(vh_values, targets[:, chunk_ids])
    vh_grad_norm = _apply_first_torch_adam_step(
        parameters=_value_optimizer_parameters(models.Vh),
        loss=vh_loss,
        learning_rate=float(cfg["lr_Vh"]),
        max_grad_norm=max_grad_norm,
    )
    assert_parity_close(
        vh_loss,
        fixture.arrays["checkpoints/update/metrics/Vh_loss"],
        stage="optimizer_step/Vh",
        tensor_name="loss",
        atol=1e-5,
        rtol=1e-4,
    )
    assert_parity_close(
        vh_grad_norm,
        fixture.arrays["checkpoints/update/metrics/Vh_grad_Vh_norm"],
        stage="optimizer_step/Vh",
        tensor_name="grad_norm",
        atol=1e-5,
        rtol=5e-4,
    )
    _assert_delta_norm_contract(
        fixture=fixture,
        model_name="Vh",
        actual=_mapped_model_delta_norm("Vh", models.Vh, vh_before, fixture),
        max_rel_drift=0.05,
    )

    # Policy post-step equality is intentionally represented as a drift
    # contract: JAX samples the TFP tanh entropy term through NumPy during the
    # update, and the fixture only exports the resulting scalar entropy. The
    # PyTorch replay therefore verifies the real policy surrogate, the scalar
    # entropy-adjusted loss value, and a bounded optimizer drift instead of
    # claiming strict parameter equality for an unexported stochastic gradient.
    policy_before = {key: value.detach().clone() for key, value in models.policy.state_dict().items()}
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
    policy_loss = policy_info["loss_policy"] - float(cfg["coef_ent"]) * fixture.tensor(
        "checkpoints/update/policy/entropy"
    )
    policy_grad_norm = _apply_first_torch_adam_step(
        parameters=list(models.policy.parameters()),
        loss=policy_loss,
        learning_rate=float(cfg["lr_actor"]),
        max_grad_norm=max_grad_norm,
    )
    assert_parity_close(
        policy_loss,
        fixture.arrays["checkpoints/update/metrics/policy_loss"],
        stage="optimizer_step/policy_drift_contract",
        tensor_name="loss_with_exported_entropy",
        atol=1e-5,
        rtol=1e-4,
    )
    assert_parity_close(
        policy_info["loss_policy"],
        fixture.arrays["checkpoints/update/policy/loss_policy"],
        stage="optimizer_step/policy_drift_contract",
        tensor_name="surrogate_loss",
        atol=1e-5,
        rtol=1e-4,
    )
    expected_policy_grad_norm = fixture.tensor("checkpoints/update/metrics/policy_grad_norm")
    grad_norm_rel_drift = torch.abs(policy_grad_norm - expected_policy_grad_norm) / expected_policy_grad_norm
    assert float(grad_norm_rel_drift) < 0.01

    _assert_delta_norm_contract(
        fixture=fixture,
        model_name="policy",
        actual=_mapped_model_delta_norm("policy", models.policy, policy_before, fixture),
        max_rel_drift=0.05,
    )


def test_deterministic_replay_rollout_loop_parity() -> None:
    num_envs = 2
    fixture, models = _prepared_fixture_models(num_envs)
    missing = [key for key in _det_rollout_required_keys() if key not in fixture.arrays]
    assert not missing, f"update fixture is missing deterministic replay keys: {missing}"

    B, T, A, action_dim = fixture.arrays["inputs/det_rollout/actions"].shape
    det_actions = torch.empty((B, T, A, action_dim), dtype=fixture.tensor("inputs/det_rollout/actions").dtype)
    det_rnn_states = torch.empty_like(fixture.tensor("inputs/det_rollout/rnn_states")) if models.spec.use_rnn else None
    with torch.no_grad():
        for b in range(B):
            rnn_state = models.policy.initialize_carry(A) if models.spec.use_rnn else None
            for t in range(T):
                graph = fixture_graph_data(fixture, index=(b, t), prefix="inputs/det_rollout/graph")
                action, _log_prob, _mode, rnn_state = models.policy.act(
                    graph,
                    rnn_state,
                    A,
                    deterministic=True,
                )
                det_actions[b, t] = action
                if det_rnn_states is not None:
                    det_rnn_states[b, t] = rnn_state

    assert_parity_close(
        det_actions,
        fixture.arrays["inputs/det_rollout/actions"],
        stage="num_envs=2/det_replay/policy",
        tensor_name="actions",
        atol=1e-5,
        rtol=1e-5,
    )
    if det_rnn_states is not None:
        assert_parity_close(
            det_rnn_states,
            fixture.arrays["inputs/det_rollout/rnn_states"],
            stage="num_envs=2/det_replay/policy",
            tensor_name="rnn_states",
            atol=1e-5,
            rtol=1e-5,
        )

    if "inputs/det_rollout/log_pis" in fixture.arrays:
        det_logp = torch.empty((B, T, A), dtype=fixture.tensor("inputs/det_rollout/log_pis").dtype)
        with torch.no_grad():
            for b, t in _iter_bt(B, T):
                graph = fixture_graph_data(fixture, index=(b, t), prefix="inputs/det_rollout/graph")
                rnn_state = _det_policy_rnn_state(fixture, b, t, use_rnn=models.spec.use_rnn)
                log_prob, _entropy, _next_rnn_state = models.policy.evaluate(
                    graph,
                    fixture.tensor("inputs/det_rollout/actions")[b, t],
                    rnn_state,
                    A,
                )
                det_logp[b, t] = log_prob
        assert_parity_close(
            det_logp,
            fixture.arrays["inputs/det_rollout/log_pis"],
            stage="num_envs=2/det_replay/policy",
            tensor_name="log_pis",
            atol=1e-5,
            rtol=1e-5,
        )

    _compute_det_vh_forward_and_targets(fixture, models, num_envs=num_envs)


@pytest.mark.parametrize("num_envs", _NUM_ENVS_WITH_UPDATE_FIXTURES)
def test_deterministic_vh_target_loss_parity(num_envs: int) -> None:
    fixture, models = _prepared_fixture_models(num_envs)
    missing = [key for key in _det_rollout_required_keys() if key not in fixture.arrays]
    assert not missing, f"update fixture is missing deterministic Vh parity keys: {missing}"

    _compute_det_vh_forward_and_targets(fixture, models, num_envs=num_envs)


def test_rnn_rollout_parity() -> None:
    fixture, models = _prepared_fixture_models()
    if not models.spec.use_rnn:
        pytest.skip("fixture does not use RNN")

    B, T, A, _ = fixture.arrays["inputs/rollout/actions"].shape
    rnn_states = scan_policy_rnn_states(
        policy=models.policy,
        graph=fixture_graph_data(fixture),
        B=B,
        T=T,
        A=A,
    )
    assert_parity_close(
        rnn_states,
        fixture.arrays["inputs/rollout/rnn_states"],
        stage="rnn_rollout",
        tensor_name="policy_rnn_states",
        atol=1e-5,
        rtol=1e-4,
    )
