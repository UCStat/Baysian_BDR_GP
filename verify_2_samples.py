#!/usr/bin/env python3
"""Verify that the public simulation runner saves exactly two posterior draws.

This follows Run_Example/run_one_case.py instead of constructing sampler
classes directly. The command uses the Case 1a wrapper configuration with tiny
MCMC settings, saves mcmc_samples.pkl files, and checks every sampled parameter
array in every run folder.
"""

from __future__ import annotations

import csv
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from Run_Example.run_one_case import (
    DATA_CASES,
    DATA_DIMENSIONS,
    KERNEL_TYPE,
    LAYERS,
    REPO_ROOT,
    RUN_SIMULATION,
    VARIANT_DIMENSIONS,
    W_VARIANTS,
    extend_values,
)


SAMPLE_SIZE = 24
N_CHAINS = 2
N_ITERATIONS = 2
BURN_IN = 0
THIN = 1
EXPECTED_SAVED_SAMPLES = (N_ITERATIONS - BURN_IN) // THIN
OUTPUT_DIR = REPO_ROOT / "test_outputs" / "verify_2_samples"
SCALAR_METRIC_SAMPLE_KEYS = {
    "rmspe_samples",
    "nsme_samples",
    "crps_samples",
    "score_samples",
    "bic_samples",
    "mlppd_samples",
    "cp_samples",
    "alci_samples",
}
STATIC_CHAIN_KEYS = {
    "column_indices",
    "X_selected",
    "W_fixed",
    "Z",
}


def format_command(command: Iterable[str]) -> str:
    """Return a shell-readable command string for logs."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def build_command() -> list[str]:
    """Build the verification command using run_one_case.py constants."""
    command = [sys.executable, str(RUN_SIMULATION)]
    command.extend(["--sample-size", str(SAMPLE_SIZE)])
    command.extend(["--n-chains", str(N_CHAINS)])
    extend_values(command, "--data-cases", DATA_CASES)
    extend_values(command, "--data-dimensions", DATA_DIMENSIONS)
    extend_values(command, "--layers", LAYERS)
    command.extend(["--kernel-type", KERNEL_TYPE])
    extend_values(command, "--w-variants", W_VARIANTS)
    extend_values(command, "--variant-dimensions", VARIANT_DIMENSIONS)
    command.extend(["--n-iterations", str(N_ITERATIONS)])
    command.extend(["--burn-in", str(BURN_IN)])
    command.extend(["--thin", str(THIN)])
    command.extend(["--output-dir", str(OUTPUT_DIR)])
    command.append("--no-plots")
    return command


def tail(text: str, max_lines: int = 80) -> str:
    """Keep subprocess failure messages readable."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def variant_dimension_count() -> int:
    """Return selected posterior-D count per layer from run_one_case constants."""
    total = 0
    for entry in VARIANT_DIMENSIONS:
        _, dims_text = entry.split("=", 1)
        total += len([dim for dim in dims_text.split(",") if dim.strip()])
    return total


def expected_row_count() -> int:
    """Return the number of run folders expected for the Case 1a command."""
    return len(DATA_CASES) * len(DATA_DIMENSIONS) * len(LAYERS) * variant_dimension_count()


def sample_count(value: Any) -> int:
    """Return the saved MCMC draw count for one sampled parameter value."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if hasattr(value, "__len__"):
        return len(value)
    return 1


def sample_metric_keys(metrics: Mapping[str, Any]) -> list[str]:
    """Return per-sample metric keys whose leading dimension should be checked."""
    keys = []
    for key, value in metrics.items():
        lower_key = key.lower()
        if lower_key in SCALAR_METRIC_SAMPLE_KEYS:
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                keys.append(key)
    return keys


def load_summary_rows() -> list[dict[str, str]]:
    """Read the aggregate summary written by run_simulation.py."""
    summary_path = OUTPUT_DIR / "simulation_summary.csv"
    if not summary_path.exists():
        raise AssertionError(f"Missing simulation summary: {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_rows = expected_row_count()
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} summary rows, got {len(rows)}")

    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        details = "; ".join(
            f"D={row.get('posterior_D')} L={row.get('layer')} "
            f"variant={row.get('variant')}: {row.get('error')}"
            for row in failures[:5]
        )
        raise AssertionError(f"Runner produced non-ok rows: {details}")

    return rows


def verify_run_samples(row: Mapping[str, str]) -> None:
    """Verify parameter and per-sample metric draw counts for one run folder."""
    run_dir = Path(row["output_dir"])
    samples_path = run_dir / "mcmc_samples.pkl"
    if not samples_path.exists():
        raise AssertionError(f"Missing saved samples: {samples_path}")

    with samples_path.open("rb") as handle:
        results = pickle.load(handle)

    chains_samples = results.get("chains_samples", [])
    if len(chains_samples) != N_CHAINS:
        raise AssertionError(f"{run_dir.name}: expected {N_CHAINS} chains, got {len(chains_samples)}")

    for chain_idx, samples in enumerate(chains_samples, start=1):
        if not samples:
            raise AssertionError(f"{run_dir.name}: chain {chain_idx} has no sampled parameters")
        for parameter, values in samples.items():
            if parameter in STATIC_CHAIN_KEYS:
                continue
            count = sample_count(values)
            if count != EXPECTED_SAVED_SAMPLES:
                raise AssertionError(
                    f"{run_dir.name}: chain {chain_idx} parameter {parameter} "
                    f"has {count} samples, expected {EXPECTED_SAVED_SAMPLES}"
                )

    chains_metrics = results.get("chains_metrics", [])
    if len(chains_metrics) != N_CHAINS:
        raise AssertionError(f"{run_dir.name}: expected {N_CHAINS} metric chains, got {len(chains_metrics)}")

    for chain_idx, metrics in enumerate(chains_metrics, start=1):
        for metric in sample_metric_keys(metrics):
            count = sample_count(metrics[metric])
            if count != EXPECTED_SAVED_SAMPLES:
                raise AssertionError(
                    f"{run_dir.name}: chain {chain_idx} metric {metric} "
                    f"has {count} samples, expected {EXPECTED_SAVED_SAMPLES}"
                )


def run_verification_command() -> None:
    """Run the small simulation command and fail with useful output on errors."""
    command = build_command()
    print("Running two-sample verification command:")
    print(format_command(command))

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print("\n--- stdout tail ---")
            print(tail(completed.stdout))
        if completed.stderr:
            print("\n--- stderr tail ---")
            print(tail(completed.stderr))
        raise RuntimeError(f"Simulation command failed with exit code {completed.returncode}")


def main() -> int:
    """Run the verifier and return a process exit code."""
    print("=" * 72)
    print("Verification: exactly 2 saved posterior samples")
    print("=" * 72)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Expected saved samples per parameter: {EXPECTED_SAVED_SAMPLES}")

    try:
        run_verification_command()
        rows = load_summary_rows()
        for row in rows:
            verify_run_samples(row)
            print(
                "PASS "
                f"D={row.get('posterior_D')} "
                f"L={row.get('layer')} "
                f"variant={row.get('variant')}"
            )
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        return 1

    print("\nAll saved parameter and per-sample metric arrays have exactly 2 samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
