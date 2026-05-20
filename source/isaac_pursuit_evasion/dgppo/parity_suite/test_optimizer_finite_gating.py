from __future__ import annotations

from .parity_test_utils import importorskip

torch = importorskip("torch")

from dgppo.update_helpers import apply_value_update


def test_optimizer_step_is_skipped_for_nonfinite_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)

    grad_norm = apply_value_update(
        optimizer=optimizer,
        loss=(parameter * torch.tensor(float("nan"))).sum(),
        parameters=[parameter],
        grad_clip=2.0,
    )

    assert not bool(torch.isfinite(grad_norm).item())
    assert parameter.detach().equal(torch.tensor([1.0]))
    assert parameter.grad is None
    assert optimizer.state == {}


def test_optimizer_step_runs_for_finite_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)

    grad_norm = apply_value_update(
        optimizer=optimizer,
        loss=(parameter.square()).sum(),
        parameters=[parameter],
        grad_clip=2.0,
    )

    assert bool(torch.isfinite(grad_norm).item())
    assert parameter.detach().item() != 1.0
    assert optimizer.state
