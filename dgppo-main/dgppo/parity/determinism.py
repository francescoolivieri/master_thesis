import os
import random
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class DeterminismContract:
    seed: int = 0
    dtype: str = "float32"
    xla_preallocate: bool = False
    force_cpu: bool = False
    jax_disable_jit: bool = False
    # DG-PPO entropy uses numpy RNG inside JAX distribution code.
    # Resetting NumPy seed before each compared step keeps this stable.
    reset_numpy_seed_before_update: bool = True
    note_entropy_rng: str = (
        "Entropy in TanhTransformedDistribution samples with numpy.randint; "
        "compare entropy-sensitive metrics only when call order and NumPy seed resets match."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def apply_determinism_contract(contract: DeterminismContract) -> None:
    if contract.dtype != "float32":
        raise ValueError(f"Only float32 is supported for parity exports, got {contract.dtype}.")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if contract.xla_preallocate else "false"
    if contract.force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    if contract.jax_disable_jit:
        os.environ["JAX_DISABLE_JIT"] = "true"

    random.seed(contract.seed)
    np.random.seed(contract.seed)
