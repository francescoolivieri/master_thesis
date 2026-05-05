from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from dgppo.dgppo_models import DGPPOPolicy, DGPPOValueNet


PARAM_PREFIX = "inputs/params_before_update"


class ParameterMappingError(RuntimeError):
    """Raised when a JAX fixture parameter tree cannot match the PyTorch model."""


@dataclass(frozen=True)
class FixtureModelSpec:
    num_agents: int
    node_dim: int
    edge_dim: int
    action_dim: int
    n_cost: int
    use_rnn: bool
    use_lstm: bool
    rnn_step: int
    actor_gnn_layers: int
    Vl_gnn_layers: int
    Vh_gnn_layers: int
    policy_gnn_out_dim: int
    Vl_gnn_out_dim: int
    Vh_gnn_out_dim: int


@dataclass(frozen=True)
class MappingEntry:
    target_key: str
    source_keys: tuple[str, ...]
    transform: str


@dataclass(frozen=True)
class MappingReport:
    model_name: str
    entries: tuple[MappingEntry, ...]
    missing_target_keys: tuple[str, ...]
    unmapped_source_keys: tuple[str, ...]


@dataclass(frozen=True)
class FixtureModels:
    policy: DGPPOPolicy
    Vl: DGPPOValueNet
    Vh: DGPPOValueNet
    spec: FixtureModelSpec


def instantiate_fixture_models(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> FixtureModels:
    """Instantiate the production PyTorch DG-PPO models with the JAX fixture architecture."""

    spec = infer_fixture_model_spec(metadata, arrays)
    rnn_cell = "lstm" if spec.use_lstm else "gru"
    if spec.use_lstm:
        raise ParameterMappingError("LSTM parameter mapping pending GRU/LSTM gate-order parity")

    policy = DGPPOPolicy(
        node_dim=spec.node_dim,
        edge_dim=spec.edge_dim,
        action_dim=spec.action_dim,
        gnn_layers=spec.actor_gnn_layers,
        gnn_out_dim=spec.policy_gnn_out_dim,
        gnn_msg_dim=32,
        gnn_heads=3,
        mlp_hid=(64, 64),
        use_rnn=spec.use_rnn,
        rnn_cell=rnn_cell,
        rnn_hidden=64,
        rnn_layers=1,
    )
    Vl = DGPPOValueNet(
        node_dim=spec.node_dim,
        edge_dim=spec.edge_dim,
        gnn_layers=spec.Vl_gnn_layers,
        gnn_out_dim=spec.Vl_gnn_out_dim,
        gnn_msg_dim=32,
        gnn_heads=3,
        mlp_hid=(64, 64),
        use_rnn=spec.use_rnn,
        rnn_cell=rnn_cell,
        rnn_hidden=64,
        rnn_layers=1,
        n_out=1,
        decompose=False,
    )
    Vh = DGPPOValueNet(
        node_dim=spec.node_dim,
        edge_dim=spec.edge_dim,
        gnn_layers=spec.Vh_gnn_layers,
        gnn_out_dim=spec.Vh_gnn_out_dim,
        gnn_msg_dim=32,
        gnn_heads=3,
        mlp_hid=(64, 64),
        use_rnn=spec.use_rnn,
        rnn_cell=rnn_cell,
        rnn_hidden=64,
        rnn_layers=1,
        n_out=spec.n_cost,
        decompose=True,
    )
    return FixtureModels(policy=policy, Vl=Vl, Vh=Vh, spec=spec)


def infer_fixture_model_spec(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> FixtureModelSpec:
    cfg = metadata.get("config", {})
    if not isinstance(cfg, Mapping):
        cfg = {}

    num_agents = int(cfg.get("num_agents", arrays["inputs/rollout/actions"].shape[-2]))
    node_dim = int(arrays["inputs/rollout/graph/2"].shape[-1])
    edge_dim = int(arrays["inputs/rollout/graph/3"].shape[-1])
    action_dim = int(arrays["inputs/rollout/actions"].shape[-1])
    n_cost = int(arrays["inputs/rollout/costs"].shape[-1])
    actor_gnn_layers = int(cfg.get("actor_gnn_layers", 1))
    Vl_gnn_layers = int(cfg.get("Vl_gnn_layers", 1))
    Vh_gnn_layers = int(cfg.get("Vh_gnn_layers", 1))

    return FixtureModelSpec(
        num_agents=num_agents,
        node_dim=node_dim,
        edge_dim=edge_dim,
        action_dim=action_dim,
        n_cost=n_cost,
        use_rnn=bool(cfg.get("use_rnn", False)),
        use_lstm=bool(cfg.get("use_lstm", False)),
        rnn_step=int(cfg.get("rnn_step", 1)),
        actor_gnn_layers=actor_gnn_layers,
        Vl_gnn_layers=Vl_gnn_layers,
        Vh_gnn_layers=Vh_gnn_layers,
        policy_gnn_out_dim=_gnn_out_dim(arrays, "policy", actor_gnn_layers),
        Vl_gnn_out_dim=_gnn_out_dim(arrays, "Vl", Vl_gnn_layers),
        Vh_gnn_out_dim=_gnn_out_dim(arrays, "Vh", Vh_gnn_layers),
    )


def map_fixture_params_to_state_dict(
    model_name: str,
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, torch.Tensor], MappingReport]:
    """Map one fixture parameter subtree into a full PyTorch state_dict.

    Flax stores Dense kernels as ``[in_dim, out_dim]``; PyTorch Linear stores
    weights as ``[out_dim, in_dim]``. The mapper records every source key used
    so tests can fail clearly on stale fixture or model naming changes.
    """

    if model_name not in {"policy", "Vl", "Vh"}:
        raise ValueError(f"unknown DG-PPO fixture model {model_name!r}")

    state_template = model.state_dict()
    mapped_state = {key: value.detach().clone() for key, value in state_template.items()}
    builder = _MappingBuilder(model_name, state_template, mapped_state, arrays)

    if model_name == "policy":
        root = f"{PARAM_PREFIX}/policy/params"
        policy_root = f"{root}/PolicyNet_0"
        _map_gnn(builder, f"{policy_root}/GraphTransformerGNN_0", ["gnn"])
        _map_mlp(builder, f"{policy_root}/PolicyGNNHead", ["mlp"])
        _map_rnn(builder, f"{policy_root}/RNN_0", ["rnn"])
        _map_linear(builder, f"{root}/ScaleHid", ["scale_hid"])
        _map_linear(builder, f"{root}/OutputDenseMean", ["mean_head"])
        _map_linear(builder, f"{root}/OutputDenseStdTrans", ["std_head"])
    else:
        root = f"{PARAM_PREFIX}/{model_name}/params"
        _map_gnn(builder, f"{root}/GraphTransformerGNN_0", ["gnn", "net.gnn"])
        _map_mlp(builder, f"{root}/ValueGNNHead", ["head", "net.mlp"])
        _map_rnn(builder, f"{root}/RNN_0", ["rnn", "net.rnn"])
        _map_linear(builder, f"{root}/Dense_0", ["net.value_out"])

    report = builder.report()
    return mapped_state, report


def load_fixture_params_into_model(
    model_name: str,
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
) -> MappingReport:
    state_dict, report = map_fixture_params_to_state_dict(model_name, model, arrays)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ParameterMappingError(
            f"{model_name}: strict load failed; "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    return report


def expected_tensor_for_entry(entry: MappingEntry, arrays: Mapping[str, np.ndarray]) -> torch.Tensor:
    if entry.transform == "identity":
        return torch.as_tensor(_array(arrays, entry.source_keys[0]))
    if entry.transform == "dense_kernel":
        return torch.as_tensor(_array(arrays, entry.source_keys[0]).T)
    if entry.transform == "gru_weight_ih":
        return torch.cat([torch.as_tensor(_array(arrays, key).T) for key in entry.source_keys], dim=0)
    if entry.transform == "gru_weight_hh":
        return torch.cat([torch.as_tensor(_array(arrays, key).T) for key in entry.source_keys], dim=0)
    if entry.transform == "gru_bias_ih":
        return torch.cat([torch.as_tensor(_array(arrays, key)) for key in entry.source_keys], dim=0)
    if entry.transform == "gru_bias_hh":
        hn_bias = torch.as_tensor(_array(arrays, entry.source_keys[0]))
        zeros = torch.zeros_like(hn_bias)
        return torch.cat([zeros, zeros, hn_bias], dim=0)
    raise ValueError(f"unknown parameter transform {entry.transform!r}")


class _MappingBuilder:
    def __init__(
        self,
        model_name: str,
        state_template: Mapping[str, torch.Tensor],
        mapped_state: dict[str, torch.Tensor],
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        self.model_name = model_name
        self.state_template = state_template
        self.mapped_state = mapped_state
        self.arrays = arrays
        self.entries: list[MappingEntry] = []
        self.assigned: set[str] = set()
        self.consumed_sources: set[str] = set()

    def add(self, target_key: str, source_keys: str | tuple[str, ...], transform: str) -> None:
        if target_key not in self.state_template:
            raise ParameterMappingError(f"{self.model_name}: target state key does not exist: {target_key}")
        if isinstance(source_keys, str):
            source_keys = (source_keys,)
        for key in source_keys:
            _array(self.arrays, key)

        entry = MappingEntry(target_key=target_key, source_keys=source_keys, transform=transform)
        expected = expected_tensor_for_entry(entry, self.arrays)
        template = self.state_template[target_key]
        if tuple(expected.shape) != tuple(template.shape):
            raise ParameterMappingError(
                f"{self.model_name}: mapped shape mismatch for {target_key}; "
                f"source={source_keys} transform={transform}; "
                f"expected {tuple(template.shape)} got {tuple(expected.shape)}"
            )

        self.mapped_state[target_key] = expected.to(device=template.device, dtype=template.dtype)
        self.entries.append(entry)
        self.assigned.add(target_key)
        self.consumed_sources.update(source_keys)

    def existing_prefixes(self, prefixes: list[str]) -> list[str]:
        return [prefix for prefix in prefixes if any(key.startswith(f"{prefix}.") for key in self.state_template)]

    def report(self) -> MappingReport:
        root = f"{PARAM_PREFIX}/{self.model_name}/params"
        source_keys = {key for key in self.arrays if key.startswith(f"{root}/")}
        missing_targets = tuple(sorted(set(self.state_template) - self.assigned))
        unmapped_sources = tuple(sorted(source_keys - self.consumed_sources))
        return MappingReport(
            model_name=self.model_name,
            entries=tuple(self.entries),
            missing_target_keys=missing_targets,
            unmapped_source_keys=unmapped_sources,
        )


def _map_gnn(builder: _MappingBuilder, source_root: str, target_prefixes: list[str]) -> None:
    prefixes = builder.existing_prefixes(target_prefixes)
    if not prefixes:
        return
    layer_ids = sorted(
        {
            int(key.split("/GraphTransformer_")[1].split("/")[0])
            for key in builder.arrays
            if key.startswith(f"{source_root}/GraphTransformer_")
        }
    )
    dense_to_linear = {
        "Dense_0": "query",
        "Dense_1": "key",
        "Dense_2": "value",
        "Dense_3": "edge_feats",
        "Dense_4": "node_proj",
    }
    for layer_id in layer_ids:
        for prefix in prefixes:
            target_root = f"{prefix}.gnn_layers.{layer_id}"
            source_layer = f"{source_root}/GraphTransformer_{layer_id}"
            for dense_name, linear_name in dense_to_linear.items():
                _map_linear(
                    builder,
                    f"{source_layer}/{dense_name}",
                    [f"{target_root}.{linear_name}"],
                    bias=dense_name != "Dense_3",
                )


def _map_mlp(builder: _MappingBuilder, source_root: str, target_prefixes: list[str]) -> None:
    prefixes = builder.existing_prefixes(target_prefixes)
    if not prefixes:
        return
    dense_ids = sorted(
        {
            int(key.split("/Dense_")[1].split("/")[0])
            for key in builder.arrays
            if key.startswith(f"{source_root}/Dense_")
        }
    )
    norm_ids = sorted(
        {
            int(key.split("/LayerNorm_")[1].split("/")[0])
            for key in builder.arrays
            if key.startswith(f"{source_root}/LayerNorm_")
        }
    )
    for prefix in prefixes:
        for dense_id in dense_ids:
            _map_linear(builder, f"{source_root}/Dense_{dense_id}", [f"{prefix}.layers.{dense_id}"])
        for norm_id in norm_ids:
            builder.add(
                f"{prefix}.layer_norms.{norm_id}.weight",
                f"{source_root}/LayerNorm_{norm_id}/scale",
                "identity",
            )
            builder.add(
                f"{prefix}.layer_norms.{norm_id}.bias",
                f"{source_root}/LayerNorm_{norm_id}/bias",
                "identity",
            )


def _map_rnn(builder: _MappingBuilder, source_root: str, target_prefixes: list[str]) -> None:
    prefixes = builder.existing_prefixes(target_prefixes)
    if not prefixes:
        return
    if not any(key.startswith(f"{source_root}/") for key in builder.arrays):
        return

    cell_roots = sorted(
        {
            "/".join(key.split("/")[:-2])
            for key in builder.arrays
            if key.startswith(f"{source_root}/GRUCell_")
        },
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if not cell_roots and any(key.startswith(f"{source_root}/LSTMCell_") for key in builder.arrays):
        raise ParameterMappingError("RNN parameter mapping pending GRU/LSTM gate-order parity")

    for layer_id, cell_root in enumerate(cell_roots):
        ih_kernels = tuple(f"{cell_root}/{gate}/kernel" for gate in ("ir", "iz", "in"))
        hh_kernels = tuple(f"{cell_root}/{gate}/kernel" for gate in ("hr", "hz", "hn"))
        ih_biases = tuple(f"{cell_root}/{gate}/bias" for gate in ("ir", "iz", "in"))
        hn_bias = f"{cell_root}/hn/bias"
        for prefix in prefixes:
            target_root = f"{prefix}.cells.{layer_id}"
            builder.add(f"{target_root}.weight_ih", ih_kernels, "gru_weight_ih")
            builder.add(f"{target_root}.weight_hh", hh_kernels, "gru_weight_hh")
            builder.add(f"{target_root}.bias_ih", ih_biases, "gru_bias_ih")
            builder.add(f"{target_root}.bias_hh", hn_bias, "gru_bias_hh")


def _map_linear(
    builder: _MappingBuilder,
    source_root: str,
    target_prefixes: list[str],
    *,
    bias: bool = True,
) -> None:
    prefixes = builder.existing_prefixes(target_prefixes)
    if not prefixes:
        raise ParameterMappingError(f"{builder.model_name}: no target prefix exists for {target_prefixes}")
    for prefix in prefixes:
        builder.add(f"{prefix}.weight", f"{source_root}/kernel", "dense_kernel")
        if bias:
            builder.add(f"{prefix}.bias", f"{source_root}/bias", "identity")


def _gnn_out_dim(arrays: Mapping[str, np.ndarray], model_name: str, n_layers: int) -> int:
    if model_name == "policy":
        root = f"{PARAM_PREFIX}/policy/params/PolicyNet_0/GraphTransformerGNN_0"
    else:
        root = f"{PARAM_PREFIX}/{model_name}/params/GraphTransformerGNN_0"
    return int(arrays[f"{root}/GraphTransformer_{n_layers - 1}/Dense_4/bias"].shape[0])


def _array(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    try:
        return arrays[key]
    except KeyError as exc:
        raise ParameterMappingError(f"fixture parameter is missing: {key}") from exc
