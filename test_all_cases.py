#!/usr/bin/env python3
"""Smoke-test every synthetic simulation case through the public runner.

This file intentionally follows the same command-building pattern as
Run_Example/run_one_case.py. The example wrapper runs one paper-sized Case 1a
experiment; this test loops over Case 1a, Case 1b, Case 2a, and Case 2b with
small MCMC settings so the full runner path can be checked quickly.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Run_Example.run_one_case import (
    KERNEL_TYPE,
    LAYERS,
    REPO_ROOT,
    RUN_SIMULATION,
    W_VARIANTS,
    extend_values,
)


@dataclass(frozen=True)
class SyntheticCase:
    label: str
    data_case: str
    data_dimension: int


SMOKE_CASES = (
    SyntheticCase("case1a", "case1", 1),
    SyntheticCase("case1b", "case1", 2),
    SyntheticCase("case2a", "case2", 1),
    SyntheticCase("case2b", "case2", 2),
)

# Small settings keep this script in smoke-test territory. Increase these if
# you want a stronger statistical run; use Run_Example/run_one_case.py for the
# paper-sized replication settings.
SAMPLE_SIZE = 24
N_CHAINS = 2
N_ITERATIONS = 2
BURN_IN = 0
THIN = 1
OUTPUT_ROOT = REPO_ROOT / "test_outputs" / "all_cases_smoke"

EXPECTED_METRIC_COLUMNS = (
    "rmspe_mean",
    "nsme_mean",
    "crps_mean",
    "score_mean",
    "bic_mean",
    "mlppd_mean",
    "cp_mean",
    "alci_mean",
)


def format_command(command: Iterable[str]) -> str:
    """Return a shell-readable command string for logs."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def build_case_command(case: SyntheticCase) -> list[str]:
    """Build one smoke-test command using the run_one_case.py CLI shape."""
    output_dir = OUTPUT_ROOT / case.label
    variant_dimensions = [
        "full=1,2,3",
        "No_W_Selective=1,2,3",
        f"W_known={case.data_dimension}",
    ]

    command = [sys.executable, str(RUN_SIMULATION)]
    command.extend(["--sample-size", str(SAMPLE_SIZE)])
    command.extend(["--n-chains", str(N_CHAINS)])
    extend_values(command, "--data-cases", [case.data_case])
    extend_values(command, "--data-dimensions", [case.data_dimension])
    extend_values(command, "--layers", LAYERS)
    command.extend(["--kernel-type", KERNEL_TYPE])
    extend_values(command, "--w-variants", W_VARIANTS)
    extend_values(command, "--variant-dimensions", variant_dimensions)
    command.extend(["--n-iterations", str(N_ITERATIONS)])
    command.extend(["--burn-in", str(BURN_IN)])
    command.extend(["--thin", str(THIN)])
    command.extend(["--output-dir", str(output_dir)])
    command.extend(["--no-plots", "--no-save-samples"])
    return command


def tail(text: str, max_lines: int = 80) -> str:
    """Keep subprocess failure messages readable."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def expected_row_count() -> int:
    """Return rows expected for one data case command."""
    full_dims = 3
    no_w_selective_dims = 3
    w_known_dims = 1
    variants_per_layer = full_dims + no_w_selective_dims + w_known_dims
    return variants_per_layer * len(LAYERS)


def validate_summary(case: SyntheticCase) -> None:
    """Assert that the runner wrote successful rows and metrics."""
    summary_path = OUTPUT_ROOT / case.label / "simulation_summary.csv"
    if not summary_path.exists():
        raise AssertionError(f"{case.label}: missing {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_rows = expected_row_count()
    if len(rows) != expected_rows:
        raise AssertionError(f"{case.label}: expected {expected_rows} summary rows, got {len(rows)}")

    bad_rows = [row for row in rows if row.get("status") != "ok"]
    if bad_rows:
        errors = "; ".join(
            f"D={row.get('posterior_D')} L={row.get('layer')} "
            f"variant={row.get('variant')}: {row.get('error')}"
            for row in bad_rows[:5]
        )
        raise AssertionError(f"{case.label}: non-ok summary rows: {errors}")

    missing_metrics = [
        (
            row.get("posterior_D"),
            row.get("layer"),
            row.get("variant"),
            column,
        )
        for row in rows
        for column in EXPECTED_METRIC_COLUMNS
        if row.get(column, "") == ""
    ]
    if missing_metrics:
        posterior_D, layer, variant, column = missing_metrics[0]
        raise AssertionError(
            f"{case.label}: missing {column} for D={posterior_D}, "
            f"layer={layer}, variant={variant}"
        )


def run_case(case: SyntheticCase) -> bool:
    """Run one synthetic case and validate its aggregate summary."""
    command = build_case_command(case)
    print(f"\n[{case.label}] {format_command(command)}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(f"[{case.label}] FAILED with exit code {completed.returncode}")
        if completed.stdout:
            print("\n--- stdout tail ---")
            print(tail(completed.stdout))
        if completed.stderr:
            print("\n--- stderr tail ---")
            print(tail(completed.stderr))
        return False

    try:
        validate_summary(case)
    except Exception as exc:
        print(f"[{case.label}] FAILED validation: {exc}")
        if completed.stdout:
            print("\n--- stdout tail ---")
            print(tail(completed.stdout))
        if completed.stderr:
            print("\n--- stderr tail ---")
            print(tail(completed.stderr))
        return False

    print(f"[{case.label}] PASS")
    return True


def main() -> int:
    """Run all synthetic case smoke tests."""
    print("=" * 72)
    print("Synthetic all-case smoke test")
    print("=" * 72)
    print(f"Runner: {RUN_SIMULATION}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Settings: n={SAMPLE_SIZE}, chains={N_CHAINS}, iterations={N_ITERATIONS}")
    print(f"Layers: {', '.join(str(layer) for layer in LAYERS)}")
    print(f"Variants: {', '.join(W_VARIANTS)}")

    results = {case.label: run_case(case) for case in SMOKE_CASES}
    passed = sum(results.values())
    total = len(results)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for label, ok in results.items():
        print(f"{label:<8} {'PASS' if ok else 'FAIL'}")
    print(f"\nResults: {passed}/{total} cases passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
