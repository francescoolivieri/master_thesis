from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_npz, save_json
from .manifest import CheckpointSpec


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    expected_shape: tuple[int, ...] | None
    actual_shape: tuple[int, ...] | None
    max_abs_error: float | None
    max_rel_error: float | None
    atol: float
    rtol: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_npz(
    expected_npz: Path,
    actual_npz: Path,
    manifest: list[CheckpointSpec],
    output_json: Path | None = None,
) -> dict[str, Any]:
    expected = load_npz(expected_npz)
    actual = load_npz(actual_npz)

    results: list[CheckResult] = []
    for spec in manifest:
        name = spec.name
        if name not in expected:
            status = "missing_expected"
            results.append(CheckResult(name, status, None, None, None, None, spec.atol, spec.rtol, "Missing in expected npz"))
            continue
        if name not in actual:
            status = "missing_actual"
            exp_shape = tuple(expected[name].shape)
            results.append(CheckResult(name, status, exp_shape, None, None, None, spec.atol, spec.rtol, "Missing in actual npz"))
            continue

        exp = expected[name]
        got = actual[name]
        exp_shape = tuple(exp.shape)
        got_shape = tuple(got.shape)
        if exp_shape != got_shape:
            results.append(
                CheckResult(
                    name=name,
                    status="shape_mismatch",
                    expected_shape=exp_shape,
                    actual_shape=got_shape,
                    max_abs_error=None,
                    max_rel_error=None,
                    atol=spec.atol,
                    rtol=spec.rtol,
                    note="Shapes differ",
                )
            )
            continue

        abs_error = np.abs(got - exp)
        denom = np.maximum(np.abs(exp), 1e-12)
        rel_error = abs_error / denom
        max_abs = float(abs_error.max()) if abs_error.size else 0.0
        max_rel = float(rel_error.max()) if rel_error.size else 0.0
        ok = np.allclose(got, exp, atol=spec.atol, rtol=spec.rtol)
        results.append(
            CheckResult(
                name=name,
                status="pass" if ok else "fail",
                expected_shape=exp_shape,
                actual_shape=got_shape,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                atol=spec.atol,
                rtol=spec.rtol,
            )
        )

    failed = [r for r in results if r.status not in {"pass"}]
    summary = {
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "failed_names": [r.name for r in failed],
    }
    report = {"summary": summary, "results": [r.to_dict() for r in results]}
    if output_json is not None:
        save_json(output_json, report)
    return report


def format_report_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "DG-PPO parity report",
        f"total={summary['total']} passed={summary['passed']} failed={summary['failed']}",
    ]
    for result in report["results"]:
        status = result["status"]
        name = result["name"]
        if status == "pass":
            continue
        lines.append(
            f"- {status}: {name} "
            f"(shape exp={result['expected_shape']} got={result['actual_shape']}, "
            f"max_abs={result['max_abs_error']}, max_rel={result['max_rel_error']})"
        )
    return "\n".join(lines)
