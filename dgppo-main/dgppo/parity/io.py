import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp


def to_numpy_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(_to_numpy_leaf, tree)


def _to_numpy_leaf(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (jnp.ndarray, jax.Array)):
        return np.asarray(value)
    try:
        return np.asarray(value)
    except Exception:
        return value


def flatten_tree(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    _flatten_impl(tree, prefix, flat)
    return flat


def _flatten_impl(tree: Any, prefix: str, output: dict[str, np.ndarray]) -> None:
    if tree is None:
        return
    if isinstance(tree, dict):
        for key in sorted(tree.keys()):
            _flatten_impl(tree[key], _join(prefix, str(key)), output)
        return
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        for key in tree._fields:
            _flatten_impl(getattr(tree, key), _join(prefix, key), output)
        return
    if isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            _flatten_impl(value, _join(prefix, str(idx)), output)
        return
    if isinstance(tree, str):
        return

    array = _to_numpy_leaf(tree)
    if isinstance(array, (int, float, bool)):
        array = np.asarray(array)
    if isinstance(array, np.ndarray):
        output[prefix] = array


def _join(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not arrays:
        raise ValueError(f"No arrays to save for {path}.")
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
