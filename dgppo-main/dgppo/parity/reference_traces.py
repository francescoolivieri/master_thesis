import functools as ft
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from dgppo.algo import make_algo
from dgppo.algo.dgppo import DGPPO
from dgppo.algo.utils import compute_dec_ocp_gae
from dgppo.env import make_env
from dgppo.trainer.data import Rollout

from .determinism import DeterminismContract, apply_determinism_contract
from .io import flatten_tree, save_json, save_npz, save_pickle, to_numpy_tree
from .manifest import default_checkpoint_manifest

if not hasattr(jax, "tree_map"):
    # JAX removed the top-level alias in 0.6; the reference code still calls it in update paths.
    jax.tree_map = jtu.tree_map


@dataclass(frozen=True)
class UpdateFixtureConfig:
    seed: int = 0
    env_id: str = "LidarSpread"
    num_agents: int = 8
    obs: int = 2
    n_rays: int = 32
    full_observation: bool = False
    n_env_train: int = 16
    batch_size: int = 1024
    rnn_step: int = 16
    actor_gnn_layers: int = 2
    Vl_gnn_layers: int = 2
    Vh_gnn_layers: int = 1
    lr_actor: float = 3e-4
    lr_Vl: float = 1e-3
    lr_Vh: float = 1e-3
    clip_eps: float = 0.25
    coef_ent: float = 1e-2
    alpha: float = 10.0
    cbf_eps: float = 1e-2
    cbf_weight: float = 1.0
    cbf_schedule: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    cost_weight: float = 0.0
    cost_schedule: bool = False
    train_steps: int = 1000
    epoch_ppo: int = 1
    use_rnn: bool = True
    use_lstm: bool = False
    update_step: int = 0
    n_drift_updates: int = 4


def _param_norm(tree: Any) -> float:
    sq = 0.0
    for leaf in jtu.tree_leaves(tree):
        leaf_np = np.asarray(leaf)
        sq += float(np.square(leaf_np).sum())
    return float(np.sqrt(sq))


def _param_delta_norm(before: Any, after: Any) -> float:
    delta = jtu.tree_map(lambda a, b: b - a, before, after)
    return _param_norm(delta)


def _copy_params(tree: Any) -> Any:
    return jtu.tree_map(lambda x: jnp.array(x), tree)


def _copy_optimizer_state(train_state: Any) -> Any:
    return jtu.tree_map(lambda x: jnp.array(x), train_state.opt_state)


def _clean_rollout_env_state(rollout: Rollout) -> Rollout:
    graph_clean = rollout.graph._replace(env_states=None)
    next_graph_clean = rollout.next_graph._replace(env_states=None)
    return rollout._replace(graph=graph_clean, next_graph=next_graph_clean)


def _build_rollout_batching(rollout: Rollout, batch_size: int, rnn_step: int, seed: int) -> dict[str, np.ndarray]:
    b = rollout.dones.shape[0]
    T = rollout.dones.shape[1]
    if b * T < batch_size:
        raise ValueError(f"batch_size={batch_size} exceeds rollout items={b * T}.")
    if T % rnn_step != 0:
        raise ValueError(f"rnn_step={rnn_step} must divide horizon T={T}.")
    if batch_size % T != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by horizon T={T}.")
    envs_per_batch = batch_size // T
    if envs_per_batch <= 0:
        raise ValueError(f"Invalid envs_per_batch={envs_per_batch} from batch_size={batch_size}, T={T}.")
    if b % envs_per_batch != 0:
        raise ValueError(f"n_env={b} must be divisible by envs_per_batch={envs_per_batch}.")
    np.random.seed(seed)
    idx = np.arange(b)
    np.random.shuffle(idx)
    rnn_chunk_ids = jnp.arange(T)
    rnn_chunk_ids = jnp.array(jnp.array_split(rnn_chunk_ids, T // rnn_step))
    n_batch_split = idx.shape[0] // envs_per_batch
    batch_idx = jnp.array(jnp.array_split(idx, n_batch_split))
    return {"batch_idx": np.asarray(batch_idx), "rnn_chunk_ids": np.asarray(rnn_chunk_ids)}


def _compute_rollout_actor_checkpoints(
    algo: DGPPO,
    rollout: Rollout,
    fixed_noise: jax.Array,
) -> dict[str, Any]:
    def eval_one(graph, action, rnn_state, noise):
        dist, _ = algo.policy.dist.apply(
            algo.policy_train_state.params,
            graph,
            rnn_state,
            n_agents=algo.n_agents,
        )
        base_normal = dist.distribution.distribution
        mean = base_normal.loc
        std = base_normal.scale
        mode = dist.mode()
        log_prob = dist.log_prob(action)
        fixed_noise_action = jnp.tanh(mean + std * noise)
        fixed_noise_log_prob = dist.log_prob(fixed_noise_action)
        return {
            "mean": mean,
            "std": std,
            "mode": mode,
            "log_prob": log_prob,
            "fixed_noise": noise,
            "fixed_noise_action": fixed_noise_action,
            "fixed_noise_log_prob": fixed_noise_log_prob,
        }

    return jax.vmap(jax.vmap(eval_one))(rollout.graph, rollout.actions, rollout.rnn_states, fixed_noise)


def _compute_pre_update_checkpoints(
    algo: DGPPO,
    rollout: Rollout,
    batch_idx: jnp.ndarray,
    rnn_chunk_ids: jnp.ndarray,
    step: int,
) -> tuple[dict[str, Any], Rollout]:
    b, T, a, _ = rollout.actions.shape

    key_for_det, key_for_inner = jr.split(algo.key)
    b_key = jr.split(key_for_det, b)
    det_rollout = algo.det_rollout_fn(algo.params, b_key)

    rollout = _clean_rollout_env_state(rollout)
    det_rollout = _clean_rollout_env_state(det_rollout)

    bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
        ft.partial(algo.scan_Vl, init_Vl_rnn_state=algo.init_Vl_rnn_state, Vl_params=algo.Vl_train_state.params)
    )(rollout)

    def final_Vl_fn_(graph, rnn_state):
        from dgppo.utils.utils import tree_index

        Vl, _ = algo.Vl.get_value(algo.Vl_train_state.params, tree_index(graph, -1), rnn_state)
        return Vl.squeeze(0).squeeze(0)

    b_final_Vl = jax.vmap(final_Vl_fn_)(rollout.next_graph, final_Vl_rnn_states)
    bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)

    bTah_Vh = jax.vmap(jax.vmap(ft.partial(algo.get_Vh, params={"Vh": algo.Vh_train_state.params})))(
        rollout.graph, rollout.rnn_states
    )

    def final_Vh_fn_(graph, rnn_state):
        from dgppo.utils.utils import tree_index

        _, final_rnn_state = algo.act(tree_index(graph, -1), rnn_state[-1], {"policy": algo.policy_train_state.params})
        return algo.get_Vh(tree_index(graph, -1), final_rnn_state, {"Vh": algo.Vh_train_state.params})

    final_Vh = jax.vmap(final_Vh_fn_)(rollout.next_graph, rollout.rnn_states)
    bTp1ah_Vh = jnp.concatenate([bTah_Vh, final_Vh[:, None]], axis=1)

    bTah_Qh, bT_Ql = jax.vmap(ft.partial(compute_dec_ocp_gae, disc_gamma=algo.gamma, gae_lambda=algo.gae_lambda))(
        Tah_hs=rollout.costs,
        T_l=-rollout.rewards,
        Tp1ah_Vh=bTp1ah_Vh,
        Tp1_Vl=bTp1_Vl,
    )

    bTah_Vh_det = jax.vmap(jax.vmap(ft.partial(algo.get_Vh, params={"Vh": algo.Vh_train_state.params})))(
        det_rollout.graph, det_rollout.rnn_states
    )
    final_Vh_det = jax.vmap(final_Vh_fn_)(det_rollout.next_graph, det_rollout.rnn_states)
    bTp1ah_Vh_det = jnp.concatenate([bTah_Vh_det, final_Vh_det[:, None]], axis=1)
    bTah_Qh_det, _ = jax.vmap(ft.partial(compute_dec_ocp_gae, disc_gamma=algo.gamma, gae_lambda=algo.gae_lambda))(
        Tah_hs=det_rollout.costs,
        T_l=-det_rollout.rewards,
        Tp1ah_Vh=bTp1ah_Vh_det,
        Tp1_Vl=bTp1_Vl,
    )

    bT_Al_raw = bT_Ql - bT_Vl
    bT_Al_norm = (bT_Al_raw - bT_Al_raw.mean(axis=1, keepdims=True)) / (bT_Al_raw.std(axis=1, keepdims=True) + 1e-8)
    bTa_Al = bT_Al_norm[:, :, None].repeat(algo.n_agents, axis=-1)
    bTah_cbf_deriv = (bTp1ah_Vh[:, 1:] - bTah_Vh) / algo._env.dt + algo.alpha * bTah_Vh
    bTah_Acbf = jnp.maximum(bTah_cbf_deriv + algo.cbf_eps, 0)
    bTa_is_safe = (bTah_cbf_deriv <= 0).min(axis=-1)
    bTa_A = jnp.where(bTa_is_safe, bTa_Al, jnp.zeros_like(bTa_Al))
    if algo.cbf_schedule:
        cbf_scale = algo.cbf_schedule_fn(jnp.asarray(step))
    else:
        cbf_scale = algo.cbf_weight
    bTa_A += bTah_Acbf.max(axis=-1) * cbf_scale
    bTa_A = -bTa_A

    bcT_graph = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.graph)
    bcTa_action = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.actions)
    bcTa_log_pis_old = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.log_pis)
    bcTa_A = jax.tree_map(lambda x: x[:, rnn_chunk_ids], bTa_A)
    bc_rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])

    action_key = jr.fold_in(key_for_inner, algo.policy_train_state.step)
    action_keys = jr.split(action_key, b * T).reshape((b, T, 2))
    bcT_action_keys = jax.tree_map(lambda x: x[:, rnn_chunk_ids], action_keys)
    actor_fixed_noise = jr.normal(jr.fold_in(key_for_inner, 0xD650), rollout.actions.shape)
    actor_rollout = _compute_rollout_actor_checkpoints(algo, rollout, actor_fixed_noise)

    eval_action = ft.partial(algo.scan_eval_action, actor_params=algo.policy_train_state.params)
    bcTa_log_pis, bcTa_policy_entropy, _, _ = jax.vmap(jax.vmap(eval_action))(
        bcT_graph,
        bcTa_action,
        bc_rnn_state_inits,
        bcT_action_keys,
    )
    ratio = jnp.exp(bcTa_log_pis - bcTa_log_pis_old)
    loss_policy1 = -ratio * bcTa_A
    loss_policy2 = -jnp.clip(ratio, 1.0 - algo.clip_eps, 1.0 + algo.clip_eps) * bcTa_A
    loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
    clip_frac = jnp.mean(loss_policy2 > loss_policy1)
    entropy = bcTa_policy_entropy.mean()
    policy_loss = loss_policy - algo.coef_ent * entropy
    loss_Vl_global = optax.l2_loss(bT_Vl, bT_Ql).mean()
    loss_Vh_det_global = optax.l2_loss(bTah_Vh_det, bTah_Qh_det).mean()

    pre = {
        "actor/rollout/mean": actor_rollout["mean"],
        "actor/rollout/std": actor_rollout["std"],
        "actor/rollout/mode": actor_rollout["mode"],
        "actor/rollout/log_prob": actor_rollout["log_prob"],
        "actor/rollout/fixed_noise": actor_rollout["fixed_noise"],
        "actor/rollout/fixed_noise_action": actor_rollout["fixed_noise_action"],
        "actor/rollout/fixed_noise_log_prob": actor_rollout["fixed_noise_log_prob"],
        "update/value/bT_Vl": bT_Vl,
        "update/value/bTp1_Vl": bTp1_Vl,
        "update/value/bTah_Vh": bTah_Vh,
        "update/value/bTp1ah_Vh": bTp1ah_Vh,
        "update/value/bTah_Vh_det": bTah_Vh_det,
        "update/value/bTp1ah_Vh_det": bTp1ah_Vh_det,
        "update/gae/bTah_Qh": bTah_Qh,
        "update/gae/bTah_Qh_det": bTah_Qh_det,
        "update/gae/bT_Ql": bT_Ql,
        "update/adv/bT_Al_raw": bT_Al_raw,
        "update/adv/bT_Al_norm": bT_Al_norm,
        "update/adv/bTah_cbf_deriv": bTah_cbf_deriv,
        "update/adv/bTah_Acbf": bTah_Acbf,
        "update/adv/bTa_is_safe": bTa_is_safe,
        "update/adv/bTa_A": bTa_A,
        "update/adv/cbf_scale": cbf_scale,
        "update/policy/ratio": ratio,
        "update/policy/loss_policy1": loss_policy1,
        "update/policy/loss_policy2": loss_policy2,
        "update/policy/loss_policy": loss_policy,
        "update/policy/clip_frac": clip_frac,
        "update/policy/entropy": entropy,
        "update/policy/policy_loss": policy_loss,
        "update/loss/Vl_global": loss_Vl_global,
        "update/loss/Vh_det_global": loss_Vh_det_global,
        "aux/batch_idx": batch_idx,
        "aux/rnn_chunk_ids": rnn_chunk_ids,
    }
    return pre, det_rollout


def _build_algo(config: UpdateFixtureConfig) -> DGPPO:
    env = make_env(
        env_id=config.env_id,
        num_agents=config.num_agents,
        num_obs=config.obs,
        n_rays=config.n_rays,
        full_observation=config.full_observation,
    )
    algo = make_algo(
        algo="dgppo",
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.cost_weight,
        cbf_weight=config.cbf_weight,
        actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=config.Vh_gnn_layers,
        rnn_layers=1,
        lr_actor=config.lr_actor,
        lr_Vl=config.lr_Vl,
        lr_Vh=config.lr_Vh,
        max_grad_norm=2.0,
        alpha=config.alpha,
        cbf_eps=config.cbf_eps,
        seed=config.seed,
        batch_size=config.batch_size,
        use_rnn=config.use_rnn,
        use_lstm=config.use_lstm,
        coef_ent=config.coef_ent,
        rnn_step=config.rnn_step,
        gamma=config.gamma,
        clip_eps=config.clip_eps,
        lagr_init=0.5,
        lr_lagr=1e-7,
        train_steps=config.train_steps,
        cbf_schedule=config.cbf_schedule,
        cost_schedule=config.cost_schedule,
        epoch_ppo=config.epoch_ppo,
        gae_lambda=config.gae_lambda,
    )
    if not isinstance(algo, DGPPO):
        raise TypeError(f"Expected DGPPO, got {type(algo)}")
    return algo


def _collect_rollout(algo: DGPPO, n_env_train: int, trainer_key: jax.Array) -> tuple[Rollout, jax.Array]:
    key_x0, trainer_key = jr.split(trainer_key)
    b_key = jr.split(key_x0, n_env_train)
    rollout = algo.collect(algo.params, b_key)
    return rollout, trainer_key


def export_update_fixtures(
    output_dir: Path,
    config: UpdateFixtureConfig,
    contract: DeterminismContract,
) -> dict[str, Path]:
    apply_determinism_contract(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    algo = _build_algo(config)

    trainer_key = jr.PRNGKey(config.seed)
    rollout, trainer_key = _collect_rollout(algo, config.n_env_train, trainer_key)

    np_seed_for_update = config.seed + int(config.update_step)
    batching = _build_rollout_batching(
        rollout=rollout,
        batch_size=config.batch_size,
        rnn_step=config.rnn_step,
        seed=np_seed_for_update,
    )
    pre, det_rollout = _compute_pre_update_checkpoints(
        algo=algo,
        rollout=rollout,
        batch_idx=jnp.asarray(batching["batch_idx"]),
        rnn_chunk_ids=jnp.asarray(batching["rnn_chunk_ids"]),
        step=config.update_step,
    )

    params_before = {
        "policy": _copy_params(algo.policy_train_state.params),
        "Vl": _copy_params(algo.Vl_train_state.params),
        "Vh": _copy_params(algo.Vh_train_state.params),
    }
    optimizer_state_before = {
        "policy": _copy_optimizer_state(algo.policy_train_state),
        "Vl": _copy_optimizer_state(algo.Vl_train_state),
        "Vh": _copy_optimizer_state(algo.Vh_train_state),
    }
    algo_key_before = np.asarray(algo.key)
    if contract.reset_numpy_seed_before_update:
        np.random.seed(np_seed_for_update)
    update_info = algo.update(rollout, step=config.update_step)
    params_after = {
        "policy": _copy_params(algo.policy_train_state.params),
        "Vl": _copy_params(algo.Vl_train_state.params),
        "Vh": _copy_params(algo.Vh_train_state.params),
    }

    post_metrics = {
        "update/metrics/policy_loss": update_info.get("policy/loss"),
        "update/metrics/Vl_loss": update_info.get("Vl/loss"),
        "update/metrics/Vh_loss": update_info.get("Vh/loss_Vh"),
        "update/metrics/policy_grad_norm": update_info.get("policy/grad_norm"),
        "update/metrics/Vl_grad_norm": update_info.get("Vl/grad_norm"),
        "update/metrics/Vh_grad_Vh_norm": update_info.get("Vh/grad_Vh_norm"),
        "update/param_delta/policy": np.asarray(
            _param_delta_norm(params_before["policy"], params_after["policy"]),
            dtype=np.float32,
        ),
        "update/param_delta/Vl": np.asarray(
            _param_delta_norm(params_before["Vl"], params_after["Vl"]),
            dtype=np.float32,
        ),
        "update/param_delta/Vh": np.asarray(
            _param_delta_norm(params_before["Vh"], params_after["Vh"]),
            dtype=np.float32,
        ),
    }

    payload = {
        "metadata": {
            "fixture_type": "single_update",
            "config": asdict(config),
            "determinism_contract": contract.to_dict(),
            "manifest": [spec.to_dict() for spec in default_checkpoint_manifest()],
            "numpy_seed_for_update": np_seed_for_update,
            "optimizer": {
                "policy": {
                    "type": "optax.apply_if_finite(optax.adam)",
                    "learning_rate": config.lr_actor,
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-8,
                    "eps_root": 0.0,
                    "max_consecutive_errors": 1_000_000,
                    "gradient_clipping": "compute_norm_and_clip before TrainState.apply_gradients",
                },
                "Vl": {
                    "type": "optax.apply_if_finite(optax.adam)",
                    "learning_rate": config.lr_Vl,
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-8,
                    "eps_root": 0.0,
                    "max_consecutive_errors": 1_000_000,
                    "gradient_clipping": "compute_norm_and_clip before TrainState.apply_gradients",
                },
                "Vh": {
                    "type": "optax.apply_if_finite(optax.adam)",
                    "learning_rate": config.lr_Vh,
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-8,
                    "eps_root": 0.0,
                    "max_consecutive_errors": 1_000_000,
                    "gradient_clipping": "compute_norm_and_clip before TrainState.apply_gradients",
                },
            },
        },
        "inputs": {
            "rollout": to_numpy_tree(_clean_rollout_env_state(rollout)),
            "det_rollout": to_numpy_tree(det_rollout),
            "algo_key_before_update": algo_key_before,
            "trainer_key_after_collect": np.asarray(trainer_key),
            "batching": batching,
            "params_before_update": to_numpy_tree(params_before),
            "optimizer_state_before_update": to_numpy_tree(optimizer_state_before),
        },
        "checkpoints": to_numpy_tree(pre | post_metrics),
        "raw_update_info": to_numpy_tree(update_info),
        "params_after_update": to_numpy_tree(params_after),
    }

    payload_np = to_numpy_tree(payload)
    pickle_path = output_dir / "update_fixture.pkl"
    npz_path = output_dir / "update_fixture.npz"
    meta_path = output_dir / "update_fixture.metadata.json"
    save_pickle(pickle_path, payload_np)
    save_npz(npz_path, flatten_tree(payload_np))
    save_json(meta_path, payload_np["metadata"])
    return {"pickle": pickle_path, "npz": npz_path, "metadata": meta_path}


def export_drift_trace(
    output_dir: Path,
    config: UpdateFixtureConfig,
    contract: DeterminismContract,
) -> dict[str, Path]:
    apply_determinism_contract(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    algo = _build_algo(config)
    trainer_key = jr.PRNGKey(config.seed)

    trace = []
    for i in range(config.n_drift_updates):
        rollout, trainer_key = _collect_rollout(algo, config.n_env_train, trainer_key)
        np_seed_for_update = config.seed + i
        batching = _build_rollout_batching(
            rollout=rollout,
            batch_size=config.batch_size,
            rnn_step=config.rnn_step,
            seed=np_seed_for_update,
        )
        pre, det_rollout = _compute_pre_update_checkpoints(
            algo=algo,
            rollout=rollout,
            batch_idx=jnp.asarray(batching["batch_idx"]),
            rnn_chunk_ids=jnp.asarray(batching["rnn_chunk_ids"]),
            step=i,
        )
        params_before = {
            "policy": _copy_params(algo.policy_train_state.params),
            "Vl": _copy_params(algo.Vl_train_state.params),
            "Vh": _copy_params(algo.Vh_train_state.params),
        }
        optimizer_state_before = {
            "policy": _copy_optimizer_state(algo.policy_train_state),
            "Vl": _copy_optimizer_state(algo.Vl_train_state),
            "Vh": _copy_optimizer_state(algo.Vh_train_state),
        }
        algo_key_before = np.asarray(algo.key)
        if contract.reset_numpy_seed_before_update:
            np.random.seed(np_seed_for_update)
        info = algo.update(rollout, step=i)
        params_after = {
            "policy": _copy_params(algo.policy_train_state.params),
            "Vl": _copy_params(algo.Vl_train_state.params),
            "Vh": _copy_params(algo.Vh_train_state.params),
        }
        post_metrics = {
            "update/metrics/policy_loss": info.get("policy/loss"),
            "update/metrics/Vl_loss": info.get("Vl/loss"),
            "update/metrics/Vh_loss": info.get("Vh/loss_Vh"),
            "update/metrics/policy_grad_norm": info.get("policy/grad_norm"),
            "update/metrics/Vl_grad_norm": info.get("Vl/grad_norm"),
            "update/metrics/Vh_grad_Vh_norm": info.get("Vh/grad_Vh_norm"),
            "update/param_delta/policy": np.asarray(
                _param_delta_norm(params_before["policy"], params_after["policy"]),
                dtype=np.float32,
            ),
            "update/param_delta/Vl": np.asarray(
                _param_delta_norm(params_before["Vl"], params_after["Vl"]),
                dtype=np.float32,
            ),
            "update/param_delta/Vh": np.asarray(
                _param_delta_norm(params_before["Vh"], params_after["Vh"]),
                dtype=np.float32,
            ),
        }
        trace.append({
            "step": i,
            "numpy_seed_for_update": np_seed_for_update,
            "inputs": {
                "rollout": to_numpy_tree(_clean_rollout_env_state(rollout)),
                "det_rollout": to_numpy_tree(det_rollout),
                "algo_key_before_update": algo_key_before,
                "trainer_key_after_collect": np.asarray(trainer_key),
                "batching": batching,
                "params_before_update": to_numpy_tree(params_before),
                "optimizer_state_before_update": to_numpy_tree(optimizer_state_before),
            },
            "checkpoints": to_numpy_tree(pre | post_metrics),
            "update_info": to_numpy_tree(info),
            "param_delta_norms": {
                "policy": _param_delta_norm(params_before["policy"], params_after["policy"]),
                "Vl": _param_delta_norm(params_before["Vl"], params_after["Vl"]),
                "Vh": _param_delta_norm(params_before["Vh"], params_after["Vh"]),
            },
            "param_norms_after": {
                "policy": _param_norm(params_after["policy"]),
                "Vl": _param_norm(params_after["Vl"]),
                "Vh": _param_norm(params_after["Vh"]),
            },
            "params_after_update": to_numpy_tree(params_after),
        })

    payload = {
        "metadata": {
            "fixture_type": "drift_trace",
            "config": asdict(config),
            "determinism_contract": contract.to_dict(),
            "derived": {
                "num_envs_total": 2 * config.n_env_train,
                "n_updates": config.n_drift_updates,
            },
            "manifest": [spec.to_dict() for spec in default_checkpoint_manifest()],
        },
        "trace": trace,
    }
    payload_np = to_numpy_tree(payload)

    pickle_path = output_dir / "drift_trace.pkl"
    npz_path = output_dir / "drift_trace.npz"
    meta_path = output_dir / "drift_trace.metadata.json"
    save_pickle(pickle_path, payload_np)
    save_npz(npz_path, flatten_tree(payload_np))
    save_json(meta_path, payload_np["metadata"])
    return {"pickle": pickle_path, "npz": npz_path, "metadata": meta_path}
