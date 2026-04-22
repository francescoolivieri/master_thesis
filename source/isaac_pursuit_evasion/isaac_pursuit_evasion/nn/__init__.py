"""Neural network and algorithm utilities for pursuit-evasion research."""

from .dgppo_losses import compute_cbf_advantages, compute_dec_ocp_gae, compute_policy_surrogate
from .dgppo_models import DecStateFn, MLP, RNN, RStateFn, ValueNet
from .dgppo_parity_torch import run_update_fixture_parity
from .gnn import EdgeBlock, GraphData, GraphTransformer, GraphTransformerGNN

__all__ = [
    "EdgeBlock",
    "GraphData",
    "GraphTransformer",
    "GraphTransformerGNN",
    "MLP",
    "RNN",
    "DecStateFn",
    "RStateFn",
    "ValueNet",
    "compute_dec_ocp_gae",
    "compute_cbf_advantages",
    "compute_policy_surrogate",
    "run_update_fixture_parity",
]

