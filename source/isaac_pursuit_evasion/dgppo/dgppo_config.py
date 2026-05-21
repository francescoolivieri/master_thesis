import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any, ClassVar

from skrl.agents.torch import AgentCfg
from skrl.agents.torch.base import ExperimentCfg


@dataclasses.dataclass(kw_only=True)
class DGPPOAgentCfg(AgentCfg):
    """skrl-compatible config wrapper for the custom DG-PPO agent."""

    alpha: float = 10.0
    cbf_eps: float = 1e-2
    cbf_weight: float = 1.0
    cbf_schedule: bool = True

    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    bootstrap_on_truncated: bool = False
    learning_starts: int = 0
    rollouts: int = 32
    rnn_step: int = 16
    learning_epochs: int = 8
    mini_batches: int = 8
    ratio_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    vl_loss_scale: float = 1.0
    vh_loss_scale: float = 1.0
    grad_norm_clip: float = 2.0
    rewards_shaper_scale: float = 1.0

    lr_policy: float = 3e-4
    lr_vl: float = 1e-3
    lr_vh: float = 1e-3

    obs_radius: float = 2.0
    use_rnn: bool = True
    rnn: dict[str, Any] = dataclasses.field(default_factory=lambda: {"cell": "gru", "hidden": 64, "layers": 1})
    gnn: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "policy_layers": 1,
            "vl_layers": 1,
            "vh_layers": 1,
            "policy_out_dim": 64,
            "critic_out_dim": 64,
            "msg_dim": 32,
            "n_heads": 3,
        }
    )
    model: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "policy_mlp_hid": [128, 64],
            "critic_mlp_hid": [128, 64],
            "scale_hid": 64,
            "scale_final": 0.01,
            "std_dev_init": 0.5,
            "std_dev_min": 1e-5,
        }
    )
    seed: int | None = None
    num_envs: int | None = None
    _raw: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    _ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "gae_lambda": ("lambda",),
        "rewards_shaper_scale": ("reward_scale",),
    }
    _DEFAULT_WARNING_EXEMPT_KEYS: ClassVar[frozenset[str]] = frozenset({"_raw", "seed", "num_envs"})
    _NESTED_DEFAULT_KEYS: ClassVar[frozenset[str]] = frozenset({"rnn", "gnn", "model"})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DGPPOAgentCfg":
        raw = dict(data or {})
        cfg = cls()
        cls._warn_defaulted_keys(raw, cfg)

        def read(name: str) -> Any:
            for key in (name, *cls._ALIASES.get(name, ())):
                if key in raw:
                    return raw[key]
            return getattr(cfg, name)

        cfg.experiment = cls._read_experiment(raw.get("experiment", cfg.experiment))
        cfg.alpha = float(read("alpha"))
        cfg.cbf_eps = float(read("cbf_eps"))
        cfg.cbf_weight = float(read("cbf_weight"))
        cfg.cbf_schedule = cls._as_bool(read("cbf_schedule"))
        cfg.discount_factor = float(read("discount_factor"))
        cfg.gae_lambda = float(read("gae_lambda"))
        cfg.bootstrap_on_truncated = cls._as_bool(read("bootstrap_on_truncated"))
        cfg.learning_starts = int(read("learning_starts"))
        cfg.rollouts = int(read("rollouts"))
        cfg.rnn_step = int(read("rnn_step"))
        cfg.learning_epochs = int(read("learning_epochs"))
        cfg.mini_batches = int(read("mini_batches"))
        cfg.ratio_clip = float(read("ratio_clip"))
        cfg.entropy_loss_scale = float(read("entropy_loss_scale"))
        cfg.vl_loss_scale = float(read("vl_loss_scale"))
        cfg.vh_loss_scale = float(read("vh_loss_scale"))
        cfg.grad_norm_clip = float(read("grad_norm_clip"))
        cfg.rewards_shaper_scale = float(read("rewards_shaper_scale"))
        cfg.lr_policy = float(read("lr_policy"))
        cfg.lr_vl = float(read("lr_vl"))
        cfg.lr_vh = float(read("lr_vh"))
        cfg.obs_radius = float(read("obs_radius"))
        cfg.use_rnn = cls._as_bool(read("use_rnn"))
        cfg.rnn = cls._merge_dict(cfg.rnn, raw.get("rnn"), "rnn")
        cfg.gnn = cls._read_gnn(cfg.gnn, raw.get("gnn"))
        cfg.model = cls._merge_dict(cfg.model, raw.get("model"), "model")
        cfg.seed = None if raw.get("seed") is None else int(raw["seed"])
        cfg.num_envs = None if raw.get("num_envs") is None else int(raw["num_envs"])
        cfg._raw = raw
        return cfg

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self._raw.get(key, default)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _read_experiment(value: Any) -> ExperimentCfg:
        if isinstance(value, ExperimentCfg):
            return value
        if value is None:
            return ExperimentCfg()
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected experiment to be a mapping, got {type(value).__name__}")
        return ExperimentCfg(**dict(value))

    @classmethod
    def _merge_dict(cls, defaults: Mapping[str, Any], value: Any, name: str) -> dict[str, Any]:
        if value is None:
            return dict(defaults)
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected {name} to be a mapping or null, got {type(value).__name__}")
        return {**dict(defaults), **dict(value)}

    @classmethod
    def _read_gnn(cls, defaults: Mapping[str, Any], value: Any) -> dict[str, Any]:
        gnn = cls._merge_dict(defaults, value, "gnn")
        if not isinstance(value, Mapping) or "critic_layers" not in value:
            return gnn

        critic_layers = int(value["critic_layers"])
        if "vl_layers" not in value:
            gnn["vl_layers"] = critic_layers
        if "vh_layers" not in value:
            gnn["vh_layers"] = critic_layers
        gnn.pop("critic_layers", None)
        warnings.warn(
            "DGPPOAgentCfg.gnn.critic_layers is deprecated and ambiguous; "
            "use gnn.vl_layers and gnn.vh_layers instead.",
            stacklevel=3,
        )
        return gnn

    @classmethod
    def _warn_defaulted_keys(cls, raw: Mapping[str, Any], defaults: "DGPPOAgentCfg") -> None:
        missing = cls._missing_top_level_keys(raw, defaults)
        missing.extend(cls._missing_nested_keys(raw, defaults))
        if not missing:
            return

        if not raw:
            message = "DGPPOAgentCfg received an empty config mapping; all DG-PPO parameters are using defaults."
        else:
            listed = ", ".join(missing)
            message = f"DGPPOAgentCfg is using defaults for missing config keys: {listed}."
        warnings.warn(message, stacklevel=3)

    @classmethod
    def _missing_top_level_keys(cls, raw: Mapping[str, Any], defaults: "DGPPOAgentCfg") -> list[str]:
        missing = []
        for field in dataclasses.fields(defaults):
            name = field.name
            if not field.init or name in cls._DEFAULT_WARNING_EXEMPT_KEYS:
                continue
            keys = (name, *cls._ALIASES.get(name, ()))
            if all(key not in raw for key in keys):
                missing.append(name)
        return missing

    @classmethod
    def _missing_nested_keys(cls, raw: Mapping[str, Any], defaults: "DGPPOAgentCfg") -> list[str]:
        missing = []
        for name in sorted(cls._NESTED_DEFAULT_KEYS):
            value = raw.get(name)
            if value is None:
                if name in raw:
                    missing.append(name)
                continue
            if not isinstance(value, Mapping):
                continue
            for key in getattr(defaults, name):
                if key in value:
                    continue
                if name == "gnn" and key in {"vl_layers", "vh_layers"} and "critic_layers" in value:
                    continue
                missing.append(f"{name}.{key}")
        return missing
