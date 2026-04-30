from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from dgppo.algo.utils import compute_dec_ocp_gae
from dgppo.trainer.utils import compute_norm_and_clip

from .io import flatten_tree, save_json, save_npz, save_pickle, to_numpy_tree


@dataclass(frozen=True)
class KernelFixtureConfig:
    seed: int = 0
    T: int = 5
    n_agents: int = 4
    n_cost: int = 2
    disc_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.25
    max_grad_norm: float = 2.0


def build_kernel_fixtures(config: KernelFixtureConfig) -> dict:
    rng = np.random.default_rng(config.seed)

    Tah_hs = rng.normal(0.0, 1.0, size=(config.T, config.n_agents, config.n_cost)).astype(np.float32)
    T_l = rng.normal(0.0, 1.0, size=(config.T,)).astype(np.float32)
    Tp1ah_Vh = rng.normal(0.0, 1.0, size=(config.T + 1, config.n_agents, config.n_cost)).astype(np.float32)
    Tp1_Vl = rng.normal(0.0, 1.0, size=(config.T + 1,)).astype(np.float32)

    Qhs, Ql = compute_dec_ocp_gae(
        Tah_hs=jnp.asarray(Tah_hs),
        T_l=jnp.asarray(T_l),
        Tp1ah_Vh=jnp.asarray(Tp1ah_Vh),
        Tp1_Vl=jnp.asarray(Tp1_Vl),
        disc_gamma=config.disc_gamma,
        gae_lambda=config.gae_lambda,
    )

    ratio = rng.uniform(0.5, 1.5, size=(3, 2, config.T, config.n_agents)).astype(np.float32)
    advantage = rng.normal(0.0, 1.0, size=(3, 2, config.T, config.n_agents)).astype(np.float32)
    loss_policy1 = -ratio * advantage
    loss_policy2 = -np.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantage
    loss_policy = np.maximum(loss_policy1, loss_policy2).mean().astype(np.float32)
    clip_frac = (loss_policy2 > loss_policy1).mean().astype(np.float32)

    grad = {
        "w1": jnp.asarray(rng.normal(size=(8, 8)).astype(np.float32)),
        "b1": jnp.asarray(rng.normal(size=(8,)).astype(np.float32)),
        "w2": jnp.asarray(rng.normal(size=(8, 3)).astype(np.float32)),
    }
    clipped_grad, original_norm = compute_norm_and_clip(grad, max_norm=config.max_grad_norm)
    clipped_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in clipped_grad.values()))

    payload = {
        "metadata": {"kernel_fixture_config": asdict(config)},
        "inputs": {
            "gae": {
                "Tah_hs": Tah_hs,
                "T_l": T_l,
                "Tp1ah_Vh": Tp1ah_Vh,
                "Tp1_Vl": Tp1_Vl,
            },
            "ppo": {"ratio": ratio, "advantage": advantage, "clip_eps": np.asarray(config.clip_eps, dtype=np.float32)},
            "grad_clip": {"grad": to_numpy_tree(grad), "max_grad_norm": np.asarray(config.max_grad_norm, dtype=np.float32)},
        },
        "checkpoints": {
            "kernel/gae/Qhs": np.asarray(Qhs),
            "kernel/gae/Ql": np.asarray(Ql),
            "kernel/ppo/loss_policy1": loss_policy1,
            "kernel/ppo/loss_policy2": loss_policy2,
            "kernel/ppo/loss_policy": np.asarray(loss_policy),
            "kernel/ppo/clip_frac": np.asarray(clip_frac),
            "kernel/grad/original_norm": np.asarray(original_norm),
            "kernel/grad/clipped_norm": np.asarray(clipped_norm),
            "kernel/grad/clipped_grad": to_numpy_tree(clipped_grad),
        },
    }
    return payload


def export_kernel_fixtures(output_dir: Path, config: KernelFixtureConfig) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_kernel_fixtures(config)
    payload_np = to_numpy_tree(payload)

    pickle_path = output_dir / "kernel_fixtures.pkl"
    save_pickle(pickle_path, payload_np)

    flat = flatten_tree(payload_np)
    npz_path = output_dir / "kernel_fixtures.npz"
    save_npz(npz_path, flat)

    meta_path = output_dir / "kernel_fixtures.metadata.json"
    save_json(meta_path, payload_np["metadata"])
    return {"pickle": pickle_path, "npz": npz_path, "metadata": meta_path}
