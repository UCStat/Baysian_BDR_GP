#!/usr/bin/env python3
"""Verify imports for the current runner-based BDR workflow.

This checker follows the same module path assumptions as Run_Example/run_one_case.py
and Scripts/run_simulation.py. It imports the public runner modules, validates that
their key helpers are available, and exits nonzero if any required import fails.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
RUN_SIMULATION_PATH = REPO_ROOT / "Scripts" / "run_simulation.py"

MODULE_FOLDERS = (
    REPO_ROOT,
    REPO_ROOT / "Data Generation",
    REPO_ROOT / "Multichain",
    REPO_ROOT / "Gibbs Sampling",
    REPO_ROOT / "Parameter Sampler",
    REPO_ROOT / "BDR Metrics and Plot",
    REPO_ROOT / "Covariance Functions",
)


def configure_environment() -> None:
    """Add repo module folders and set writable caches for plotting imports."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bdr_mpl_config"))

    for folder in MODULE_FOLDERS:
        folder_str = str(folder)
        if folder_str not in sys.path:
            sys.path.insert(0, folder_str)


def load_module_from_path(module_name: str, path: Path):
    """Import a Python file by path without requiring its folder to be a package."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_dependencies() -> None:
    """Verify third-party packages imported by the runner stack."""
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import pyDOE  # noqa: F401
    import matplotlib  # noqa: F401
    import seaborn  # noqa: F401
    import pandas  # noqa: F401


def check_run_one_case() -> None:
    """Verify the example wrapper imports and builds the expected CLI command."""
    from Run_Example import run_one_case

    command = run_one_case.build_command()
    command_text = " ".join(command)

    if command[0] != sys.executable:
        raise AssertionError("run_one_case command should start with sys.executable")
    if str(run_one_case.RUN_SIMULATION) not in command:
        raise AssertionError("run_one_case command does not include Scripts/run_simulation.py")
    for option in (
        "--sample-size",
        "--data-cases",
        "--data-dimensions",
        "--layers",
        "--kernel-type",
        "--w-variants",
        "--variant-dimensions",
        "--n-iterations",
        "--burn-in",
        "--thin",
        "--output-dir",
    ):
        if option not in command:
            raise AssertionError(f"run_one_case command is missing {option}")
    if "W_known=1" not in command_text:
        raise AssertionError("run_one_case command should include W_known=1 for Case 1a")


def check_data_generation() -> None:
    """Verify synthetic data generators import and return required keys."""
    from Data_generation import (
        generate_case1_1d,
        generate_case1_2d,
        generate_case2_1d,
        generate_case2_2d,
    )

    required_keys = {"X_train", "X_test", "y_train", "y_test", "W", "n_train", "n_test"}
    for name, generator in (
        ("case1_1d", generate_case1_1d),
        ("case1_2d", generate_case1_2d),
        ("case2_1d", generate_case2_1d),
        ("case2_2d", generate_case2_2d),
    ):
        data = generator(n=12, seed=42)
        missing = required_keys - set(data)
        if missing:
            raise AssertionError(f"{name} missing keys: {sorted(missing)}")
        if data["X_train"].shape[0] != data["n_train"]:
            raise AssertionError(f"{name} X_train rows do not match n_train")
        if data["X_test"].shape[0] != data["n_test"]:
            raise AssertionError(f"{name} X_test rows do not match n_test")


def check_core_modules() -> None:
    """Verify direct module imports used by run_multichains.py."""
    from covariance_kernel_functions_and_gradients_W import (  # noqa: F401
        IsotropicSquaredExponentialKernel,
        SeparableSquaredExponentialKernel,
    )
    from BDR_metrics import compute_BIC, compute_RMSPE  # noqa: F401
    from BDR_plot import plot_density, plot_trace  # noqa: F401
    from BDR_summaries import write_run_summary_tables  # noqa: F401
    from parameter_sampler_D1 import rmf_matrix, rmf_matrixN, sample_tau2  # noqa: F401
    from parameter_sampler_Dgeneral import rmf_matrix as rmf_matrix_Dgeneral  # noqa: F401
    from gibbs_sampler_layers_D1 import GibbsSampler1Layer  # noqa: F401
    from gibbs_sampler_layers_Dgeneral import GibbsSampler1Layer as GibbsSampler1LayerDgeneral  # noqa: F401
    from multichain_sampler_D1 import MultiChainSampler as MultiChainSamplerD1  # noqa: F401
    from multichain_sampler_Dgeneral import MultiChainSampler as MultiChainSamplerDgeneral  # noqa: F401
    from multichain_sampler_L1_variants import MultiChainSampler_L1_Variants  # noqa: F401
    from multichain_sampler_L2_variants import MultiChainSampler_L2_Variants  # noqa: F401
    from multichain_sampler_L3_variants import MultiChainSampler_L3_Variants  # noqa: F401


def check_run_multichains_api() -> None:
    """Verify the current public API from run_multichains.py."""
    from run_multichains import (
        CONFIG_FUNCTIONS,
        create_config_D1_L1,
        create_config_D2_L2,
        create_config_L1_W_Known,
        get_config_for,
        initialize_M_Lambda_V_W_D1,
        initialize_M_Lambda_V_W_Dgeneral,
        run_multichain_analysis,
    )

    expected_configs = {
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2), (3, 3),
        (5, 1), (5, 2), (5, 3),
    }
    missing_configs = expected_configs - set(CONFIG_FUNCTIONS)
    if missing_configs:
        raise AssertionError(f"Missing CONFIG_FUNCTIONS entries: {sorted(missing_configs)}")

    init_d1 = initialize_M_Lambda_V_W_D1(p=10, D=1, seed=42)
    init_d2 = initialize_M_Lambda_V_W_Dgeneral(p=10, D=2, seed=42)
    if init_d1["W_init"].shape != (10, 1):
        raise AssertionError("D=1 W_init shape mismatch")
    if init_d2["W_init"].shape != (10, 2):
        raise AssertionError("D=2 W_init shape mismatch")

    config_d1 = create_config_D1_L1(p=10, seed=42, n_chains=2, n_iterations=2, burn_in=0, thin=1)
    if config_d1["D"] != 1 or config_d1["layer"] != 1:
        raise AssertionError("create_config_D1_L1 returned wrong D/layer")
    if config_d1["W_init"].shape != (10, 1):
        raise AssertionError("create_config_D1_L1 did not attach W_init")

    config_d2 = create_config_D2_L2(
        p=10,
        seed=42,
        kernel_type="separable_squared_exponential",
        n_chains=2,
        n_iterations=2,
        burn_in=0,
        thin=1,
    )
    if config_d2["D"] != 2 or config_d2["layer"] != 2:
        raise AssertionError("create_config_D2_L2 returned wrong D/layer")
    if np.asarray(config_d2["theta_y_init"]).shape != (2,):
        raise AssertionError("D=2 theta_y_init should be length 2")

    config_get = get_config_for(D=1, layer=1, p=10, seed=42)
    if config_get["D"] != 1 or config_get["layer"] != 1:
        raise AssertionError("get_config_for returned wrong D/layer")

    variant_config = create_config_L1_W_Known(W_fixed=np.ones((10, 1)))
    if variant_config["variant"] != "W_Known":
        raise AssertionError("create_config_L1_W_Known returned wrong variant")

    if not callable(run_multichain_analysis):
        raise AssertionError("run_multichain_analysis is not callable")


def check_run_simulation_import() -> None:
    """Verify Scripts/run_simulation.py imports and exposes runner helpers."""
    run_simulation = load_module_from_path("verify_run_simulation", RUN_SIMULATION_PATH)

    for data_case in ("case1", "case2"):
        if data_case not in run_simulation.DATA_GENERATORS:
            raise AssertionError(f"DATA_GENERATORS missing {data_case}")
        for data_dim in (1, 2):
            if data_dim not in run_simulation.DATA_GENERATORS[data_case]:
                raise AssertionError(f"DATA_GENERATORS missing {data_case} dimension {data_dim}")

    old_argv = sys.argv[:]
    try:
        sys.argv = ["verify_imports.py"]
        args = run_simulation.parse_args()
    finally:
        sys.argv = old_argv
    args.sample_size = 12
    args.n_chains = 2
    args.n_iterations = 2
    args.burn_in = 0
    args.thin = 1
    args.data_cases = ["case1"]
    args.data_dimensions = [1]
    args.posterior_dimensions = [1]
    args.layers = [1]
    args.w_variants = ["full", "W_Known"]
    args.include_w_variants = False
    args.variant_dimensions = ["full=1", "W_known=1"]
    run_simulation.validate_args(args)

    selected = run_simulation.selected_w_variants(args)
    pairs = run_simulation.variant_dimension_pairs(args, selected)
    if pairs != [(None, 1), ("W_Known", 1)]:
        raise AssertionError(f"Unexpected variant/D pairs: {pairs}")

    run_name = run_simulation.run_name_for("case1", 1, 1, 1, 12, None)
    if run_name != "case1_dataD1_postD1_L1_n12":
        raise AssertionError(f"Unexpected run name: {run_name}")


def run_check(name: str, check: Callable[[], None]) -> bool:
    """Run one check, print a compact status, and return success."""
    print(f"\n[{name}]")
    try:
        check()
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        return False
    print("PASS")
    return True


def main() -> int:
    """Run all import checks and return a shell-friendly status code."""
    configure_environment()

    print("=" * 72)
    print("VERIFYING IMPORTS FOR RUNNER-BASED BDR WORKFLOW")
    print("=" * 72)
    print(f"Repo root: {REPO_ROOT}")

    checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("third-party dependencies", check_dependencies),
        ("Run_Example/run_one_case.py", check_run_one_case),
        ("Data_generation.py", check_data_generation),
        ("core sampler/metric modules", check_core_modules),
        ("run_multichains.py API", check_run_multichains_api),
        ("Scripts/run_simulation.py", check_run_simulation_import),
    )

    results = [run_check(name, check) for name, check in checks]
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 72)
    print(f"IMPORT VERIFICATION COMPLETE: {passed}/{total} checks passed")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
