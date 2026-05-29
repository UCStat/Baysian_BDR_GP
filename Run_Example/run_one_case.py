#!/usr/bin/env python3
# Case 1a
#
# This example runs one synthetic Case 1a, data-dimension-1 experiment through
# Scripts/run_simulation.py.
#
# Syntax-check this example without running the simulation:
# python3 -m py_compile Run_Example/run_one_case.py
#
# Run the Case 1a example:
# python3 Run_Example/run_one_case.py
#
# Expected Case 1a output:
# - Creates ./simulation_outputs.
# - Runs Case 1 with DATA_DIMENSIONS = [1] and SAMPLE_SIZE = 350.
# - Produces per-run folders, posterior summary tables, time-complexity tables,
#   diagnostic plots, metric comparison tables, and metric_boxplots_by_layer/.
# - To reproduce the larger Case 1a setting, run again with SAMPLE_SIZE = 600.
#
# Similar plots and tables can be generated for the other synthetic cases by
# editing the constants below:
# - Case 1b: DATA_CASES = ["case1"], DATA_DIMENSIONS = [2], and run once with
#   SAMPLE_SIZE = 350 and again with SAMPLE_SIZE = 600.
# - Case 2a: DATA_CASES = ["case2"], DATA_DIMENSIONS = [1], and run once with
#   SAMPLE_SIZE = 300 and again with SAMPLE_SIZE = 500.
# - Case 2b: DATA_CASES = ["case2"], DATA_DIMENSIONS = [2], and run once with
#   SAMPLE_SIZE = 300 and again with SAMPLE_SIZE = 500.

from __future__ import annotations  # Allow modern type annotations consistently.

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]  # Repository root: parent of Run_Example/.
RUN_SIMULATION = REPO_ROOT / "Scripts" / "run_simulation.py"  # Simulation runner entry point.

SAMPLE_SIZE = 350  # Total generated samples before the train/test split.
DATA_CASES = ["case1"]  # Synthetic data generator case(s) to run.
DATA_DIMENSIONS = [1]  # True data-generation dimension(s).
LAYERS = [1, 2, 3]  # Model depths to run.
W_VARIANTS = ["full", "No_W_Selective", "W_Known"]  # Model variants to compare.
VARIANT_DIMENSIONS = [  # Per-variant posterior dimensions.
    "full=1,2,3",  # Run full BDR with posterior D = 1, 2, and 3.
    "No_W_Selective=1,2,3",  # Run W/o selective BDR with posterior D = 1, 2, and 3.
    "W_known=1",  # Run Oracle/W_Known only at D = 1 for Case 1a.
]  # End of per-variant posterior dimension list.
N_ITERATIONS = 2000  # MCMC iterations per chain.
BURN_IN = 500  # Initial MCMC iterations discarded before summaries.
THIN = 3  # Keep every third post-burn-in MCMC sample.
OUTPUT_DIR = REPO_ROOT / "simulation_outputs"  # Folder where all outputs are written.
KERNEL_TYPE = "isotropic_squared_exponential"  # Covariance kernel used for this example.


def extend_values(command: list[str], option: str, values: list[object]) -> None:
    """Append an argparse option followed by one or more values."""
    command.append(option)
    command.extend(str(value) for value in values)


def build_command() -> list[str]:
    """Build the exact command list passed to subprocess.run."""
    command = [sys.executable, str(RUN_SIMULATION)]
    command.extend(["--sample-size", str(SAMPLE_SIZE)])
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
    return command


def main() -> int:
    """Run the Case 1a command and return the child process exit code."""
    command = build_command()
    print("Running Case 1a command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":  # Only execute when this file is run directly.
    raise SystemExit(main())  # Exit with the same status code as the simulation runner.
