from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def importorskip(module_name: str) -> ModuleType:
    """Import a module or skip without relying on pytest>=8.2's exc_type kwarg."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"could not import {module_name!r}: {exc}", allow_module_level=True)


torch = importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = REPO_ROOT / "dgppo-main" / "parity_artifacts"
UPDATE_FIXTURE_PATHS = {
    2: ARTIFACT_ROOT / "update_small" / "update_fixture.npz",
    6: ARTIFACT_ROOT / "update_num_envs6" / "update_fixture.npz",
}
DRIFT_TRACE_PATHS = {
    6: ARTIFACT_ROOT / "drift_num_envs6" / "drift_trace.npz",
}


@dataclass(frozen=True)
class LoadedFixture:
    path: Path
    metadata: dict
    arrays: dict[str, np.ndarray]

    def tensor(self, key: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.as_tensor(self.arrays[key], device=device)


def fixture_graph_data(
    fixture: LoadedFixture,
    *,
    index: tuple[int, ...] | None = None,
    prefix: str = "inputs/rollout/graph",
    device: str | torch.device = "cpu",
):
    """Build production ``GraphData`` from raw JAX fixture graph leaves.

    The JAX artifacts store the exact rollout ``GraphsTuple`` leaves as:
    ``n_node, n_edge, nodes, edges, states, receivers, senders, node_types``.
    This adapter only converts dtypes and offsets sender/receiver indices when
    multiple leading graph dimensions are flattened into one PyTorch graph
    batch; it does not rebuild topology from simplified states.
    """

    from dgppo.utils import GraphData

    leaves = [fixture.arrays[f"{prefix}/{leaf_id}"] for leaf_id in range(8)]
    if index is not None:
        leaves = [leaf[index] for leaf in leaves]

    n_nodes_np, n_edges_np, nodes_np, edges_np, states_np, receivers_np, senders_np, node_types_np = leaves
    n_nodes = torch.as_tensor(n_nodes_np, device=device).long()
    n_edges = torch.as_tensor(n_edges_np, device=device).long()
    nodes = torch.as_tensor(nodes_np, device=device)
    edges = torch.as_tensor(edges_np, device=device)
    states = torch.as_tensor(states_np, device=device)
    receivers = torch.as_tensor(receivers_np, device=device).long()
    senders = torch.as_tensor(senders_np, device=device).long()
    node_types = torch.as_tensor(node_types_np, device=device).long()

    if n_nodes.ndim == 0:
        return GraphData(
            n_nodes=n_nodes,
            n_edges=n_edges,
            nodes=nodes,
            edges=edges,
            states=states,
            receivers=receivers,
            senders=senders,
            node_types=node_types,
        )

    # Leading dimensions follow the project convention: B, T, A, NH. Graph
    # leaves are [B, T, N, ...] / [B, T, E, ...]; flatten only for the
    # production GraphData representation and preserve each graph's local
    # sender/receiver topology via a fixed padded-node offset.
    leading_shape = n_nodes.shape
    n_graphs = int(np.prod(tuple(leading_shape)))
    padded_nodes_per_graph = nodes.shape[-2]
    padded_edges_per_graph = receivers.shape[-1]
    node_offsets = (
        torch.arange(n_graphs, device=device, dtype=torch.long).reshape(tuple(leading_shape) + (1,))
        * padded_nodes_per_graph
    )

    return GraphData(
        n_nodes=n_nodes,
        n_edges=n_edges,
        nodes=nodes.reshape(n_graphs * padded_nodes_per_graph, nodes.shape[-1]),
        edges=edges.reshape(n_graphs * padded_edges_per_graph, edges.shape[-1]),
        states=states.reshape(n_graphs * padded_nodes_per_graph, states.shape[-1]),
        receivers=(receivers + node_offsets).reshape(n_graphs * padded_edges_per_graph),
        senders=(senders + node_offsets).reshape(n_graphs * padded_edges_per_graph),
        node_types=node_types.reshape(n_graphs * padded_nodes_per_graph),
    )


def load_kernel_fixture() -> LoadedFixture:
    return _load_fixture(ARTIFACT_ROOT / "kernel" / "kernel_fixtures.npz")


def load_update_fixture_for_num_envs(num_envs: int) -> LoadedFixture:
    """Load a JAX update fixture for a PyTorch total env count."""
    if num_envs not in UPDATE_FIXTURE_PATHS:
        raise ValueError(f"Unsupported DG-PPO parity num_envs={num_envs}")

    path = UPDATE_FIXTURE_PATHS[num_envs]
    if not path.exists():
        pytest.xfail(
            f"JAX update fixture for num_envs={num_envs} is not exported yet. "
            f"Export with: {export_update_command(num_envs)}"
        )

    fixture = _load_fixture(path)
    n_env_train = int(fixture.metadata["config"]["n_env_train"])
    expected_total_envs = 2 * n_env_train
    if expected_total_envs != num_envs:
        pytest.fail(f"{path} metadata maps to num_envs={expected_total_envs}, expected {num_envs}")
    return fixture


def load_drift_trace_for_num_envs(num_envs: int) -> LoadedFixture:
    """Load a JAX multi-update drift trace for a PyTorch total env count."""
    if num_envs not in DRIFT_TRACE_PATHS:
        raise ValueError(f"Unsupported DG-PPO drift parity num_envs={num_envs}")

    path = DRIFT_TRACE_PATHS[num_envs]
    if not path.exists():
        pytest.xfail(
            f"JAX drift trace for num_envs={num_envs} is not exported yet. "
            f"Export with: {export_drift_command(num_envs)}"
        )

    fixture = _load_fixture(path)
    n_env_train = int(fixture.metadata["config"]["n_env_train"])
    expected_total_envs = 2 * n_env_train
    if expected_total_envs != num_envs:
        pytest.fail(f"{path} metadata maps to num_envs={expected_total_envs}, expected {num_envs}")
    return fixture


def assert_parity_close(
    actual: torch.Tensor | np.ndarray,
    expected: torch.Tensor | np.ndarray,
    *,
    stage: str,
    tensor_name: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    actual_np = _as_numpy(actual)
    expected_np = _as_numpy(expected)

    if actual_np.shape != expected_np.shape:
        pytest.fail(
            f"{stage}: {tensor_name} shape mismatch: "
            f"expected {expected_np.shape} got {actual_np.shape}; "
            f"expected dtype={expected_np.dtype} actual dtype={actual_np.dtype}"
        )

    if actual_np.dtype == np.bool_ or expected_np.dtype == np.bool_:
        close = actual_np == expected_np
    else:
        close = np.isclose(actual_np, expected_np, atol=atol, rtol=rtol)

    if np.all(close):
        return

    abs_diff = np.abs(actual_np.astype(np.float64) - expected_np.astype(np.float64))
    denom = np.maximum(np.abs(expected_np.astype(np.float64)), 1e-12)
    rel_diff = abs_diff / denom
    mismatch_idx = tuple(np.argwhere(~close)[0])

    pytest.fail(
        f"{stage}: {tensor_name} mismatch; "
        f"shape={actual_np.shape} expected_dtype={expected_np.dtype} actual_dtype={actual_np.dtype}; "
        f"max_abs={float(abs_diff.max())} max_rel={float(rel_diff.max())}; "
        f"first_mismatch_index={mismatch_idx}; "
        f"expected={expected_np[mismatch_idx]!r} actual={actual_np[mismatch_idx]!r}"
    )


def _load_fixture(path: Path) -> LoadedFixture:
    if not path.exists():
        pytest.skip(f"Missing JAX DG-PPO parity fixture: {path}")
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return LoadedFixture(path=path, metadata=metadata, arrays=arrays)


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def export_update_command(num_envs: int) -> str:
    n_env_train = num_envs // 2
    out_dir = f"dgppo-main/parity_artifacts/update_num_envs{num_envs}"
    return (
        "JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib "
        "conda run -n dgppo python dgppo-main/parity_checks.py export-update "
        f"--output-dir {out_dir} --n-env-train {n_env_train} "
        "--batch-size 128 --rnn-step 16 --force-cpu"
    )


def export_drift_command(num_envs: int, *, n_drift_updates: int = 4) -> str:
    n_env_train = num_envs // 2
    out_dir = f"dgppo-main/parity_artifacts/drift_num_envs{num_envs}"
    return (
        "JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib "
        "conda run -n dgppo python dgppo-main/parity_checks.py export-drift "
        f"--output-dir {out_dir} --n-env-train {n_env_train} "
        f"--batch-size 128 --rnn-step 16 --n-drift-updates {n_drift_updates} --force-cpu"
    )
