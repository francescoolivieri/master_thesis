"""CLI entrypoint for DGPPO torch parity checks."""

from __future__ import annotations

import argparse

try:
    from .dgppo_parity_torch import run_update_fixture_parity
except ImportError:
    from dgppo_parity_torch import run_update_fixture_parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=str,
        default="source/isaac_pursuit_evasion/nn/update_fixture.npz",
        help="Path to update_fixture.npz",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--cbf-dt",
        type=float,
        default=0.03,
        help="Time-step used for the CBF finite-difference term.",
    )
    args = parser.parse_args()

    ok, _ = run_update_fixture_parity(
        args.fixture,
        rtol=args.rtol,
        atol=args.atol,
        cbf_dt=args.cbf_dt,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
