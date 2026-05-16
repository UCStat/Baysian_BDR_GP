#!/usr/bin/env python3
"""
Run small end-to-end BDR simulations across data dimensions, posterior
dimensions, and layers.

This script generates Case 1 or Case 2 simulation data, initializes priors and
starting values through the main configuration helpers, runs three-chain
posterior sampling, and writes diagnostics, posterior samples, and metric plots.

Implementation map:
1. Parse CLI options in `parse_args`.
2. Generate data with `DATA_GENERATORS[data_case][data_dim]`.
3. Build posterior-D SVD/matrix-Langevin initial values with `build_initial_values`.
4. Build either a full-model config or a W-variant config with `build_config`.
5. Call `run_multichain_analysis`, which dispatches to the correct sampler.
6. Save configuration, posterior samples, summaries, diagnostics, and plots.

To add a new data scenario, update `DATA_GENERATORS`. To add a new posterior
model option, update `build_config` rather than duplicating the main run loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# The repository uses directories with spaces and plain Python modules rather
# than an installed package. Add each module folder to sys.path so this script
# can be launched directly from the command line.
for folder in [
    REPO_ROOT,
    REPO_ROOT / "Data Generation",
    REPO_ROOT / "Multichain",
    REPO_ROOT / "Gibbs Sampling",
    REPO_ROOT / "Parameter Sampler",
    REPO_ROOT / "BDR Metrics and Plot",
    REPO_ROOT / "Covariance Functions",
]:
    folder_str = str(folder)
    if folder_str not in sys.path:
        sys.path.insert(0, folder_str)

from Data_generation import (  # type: ignore[import]
    generate_case1_1d,
    generate_case1_2d,
    generate_case2_1d,
    generate_case2_2d,
)
from BDR_plot import (  # type: ignore[import]
    plot_actual_vs_predicted,
    plot_autocorrelation,
    plot_density,
    plot_metrics_boxplot,
    plot_metrics_comparison_table,
    plot_trace,
)
from run_multichains import (  # type: ignore[import]
    CONFIG_FUNCTIONS,
    get_config_for,
    get_default_config,
    initialize_M_Lambda_V_W_D1,
    initialize_M_Lambda_V_W_Dgeneral,
    run_multichain_analysis,
)


# Kernel used for every run. Change this value to switch covariance functions.
# Supported options from the covariance/sampler modules are:
# - "isotropic_squared_exponential"
# - "separable_squared_exponential"
# - "isotropic_matern32"
# - "separable_matern32"
KERNEL_TYPE = "isotropic_squared_exponential"

# Prior and initialization values requested for every run. These constants are
# passed into both full BDR models and W-not-sampled variants so the comparison
# only changes the treatment of W, not the prior setup.
LAMBDA_GAMMA_SHAPE = 5.0 / 2.0
LAMBDA_GAMMA_RATE = 10.0 / 3.0
TAU2_ALPHA1 = 0.001
TAU2_ALPHA2 = 0.001
GAMMA_SHAPE = 3.0 / 2.0
GAMMA_RATE_G_AND_THETA_Y = 3.9
GAMMA_RATE_THETA_Q = 3.9 / 3.0
GAMMA_RATE_THETA_R = 3.9 / 6.0
SLIDING_WINDOW_L = 1.0
SLIDING_WINDOW_U = 2.0
TAU2_INIT = 0.005
G_INIT = 9e-5
THETA_INIT = 1.0

# W variants can be selected explicitly with --w-variants. The legacy
# --include-w-variants flag remains a shortcut for full + all three variants.
W_VARIANTS = ("W_Known", "No_W", "No_W_Selective")
W_VARIANT_CHOICES = ("full",) + W_VARIANTS

# Parameter aliases differ slightly between full samplers and variant samplers.
# This table lets the runner create diagnostics for either result format.
DIAGNOSTIC_SAMPLE_KEYS = (
    ("tau2_y", ("tau2_y", "tau2")),
    ("g_y", ("g_y", "g")),
    ("theta_y", ("theta_D_y", "theta_D", "theta_y")),
    ("tau2_q", ("tau2_q",)),
    ("g_q", ("g_q",)),
    ("theta_q", ("theta_q",)),
    ("tau2_r", ("tau2_r",)),
    ("g_r", ("g_r",)),
    ("theta_r", ("theta_r",)),
)

DATA_GENERATORS = {
    # Add new data scenarios here if you want the runner to support more cases.
    # Import the generator above, then add it under a case name and dimension.
    # The selected generator must return y_train, y_test, X_train, X_test,
    # n_train, and n_test keys.
    "case1": {
        1: generate_case1_1d,
        2: generate_case1_2d,
    },
    "case2": {
        1: generate_case2_1d,
        2: generate_case2_2d,
    },
}

# Parse command-line options for data size, MCMC settings, outputs, and variants.
def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Defaults intentionally form a small smoke-test run. For a real experiment,
    increase --sample-size, --n-iterations, and --burn-in.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate Case 1 or Case 2 data and run three-chain BDR simulations. "
            "Use --data-cases for the simulator case, "
            "--data-dimensions for the simulator's true D, and "
            "--posterior-dimensions for the BDR sampler's reduced D."
        )
    )
    # Main experiment controls:
    # --sample-size sets n for Data_generation.py before the train/test split.
    # --data-cases selects the Case 1 or Case 2 generator family.
    # --data-dimensions selects the true generator dimension, currently 1 or 2.
    # --posterior-dimensions selects the BDR posterior sampler's reduced D.
    # --layers selects the model depth: 1-layer, 2-layer, or 3-layer.
    parser.add_argument("--sample-size", type=int, default=24, help="Total generated samples before train/test split.")
    parser.add_argument("--n-chains", type=int, default=3, help="Number of MCMC chains.")
    parser.add_argument("--n-iterations", type=int, default=6, help="MCMC iterations per chain.")
    parser.add_argument("--burn-in", type=int, default=2, help="Burn-in iterations.")
    parser.add_argument("--thin", type=int, default=3, help="Thinning interval.")
    parser.add_argument(
        "--data-cases",
        type=str,
        nargs="+",
        default=["case1"],
        choices=sorted(DATA_GENERATORS),
        help="Data-generator case(s) to run. Available: case1 and case2.",
    )
    parser.add_argument(
        "--data-dimensions",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=sorted({dimension for generators in DATA_GENERATORS.values() for dimension in generators}),
        help="Data-generator dimension(s). Current Case 1 and Case 2 generators support 1 and 2.",
    )
    parser.add_argument(
        "--posterior-dimensions",
        "--dimensions",
        dest="posterior_dimensions",
        type=int,
        nargs="+",
        default=[1, 2],
        help=(
            "Posterior/model reduced dimension(s) D used by the BDR sampler. "
            "--dimensions is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Model depth(s) to run: 1, 2, and/or 3.",
    )
    parser.add_argument(
        "--mv-sampler",
        choices=["python", "rstiefel"],
        default="python",
        help=(
            "Backend for posterior M and V matrix-von-Mises-Fisher Gibbs updates. "
            "Use 'rstiefel' to call R rstiefel::rmf.matrix.gibbs via rpy2."
        ),
    )
    parser.add_argument(
        "--rstiefel-rscol",
        type=int,
        default=None,
        help="Optional rscol argument passed to rstiefel::rmf.matrix.gibbs when --mv-sampler rstiefel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "simulation_outputs",
        help="Directory for all simulation outputs.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic and metric plot generation.")
    parser.add_argument("--no-save-samples", action="store_true", help="Do not save posterior samples pickle files.")
    parser.add_argument(
        "--include-w-variants",
        action="store_true",
        help="Also run W_Known, No_W, and No_W_Selective variants for each data-case/data-D/posterior-D/layer run.",
    )
    parser.add_argument(
        "--w-variants",
        nargs="+",
        choices=W_VARIANT_CHOICES,
        default=None,
        help=(
            "Specific W/model variants to run. Choose from full, W_Known, "
            "No_W, and No_W_Selective. If omitted, only full runs unless "
            "--include-w-variants is used."
        ),
    )
    parser.add_argument(
        "--parameter-diagnostics",
        action="store_true",
        help="Also attempt R/coda parameter diagnostics. Requires rpy2, R, and coda.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining D/layer runs if one run fails.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print sampler progress.")
    parser.set_defaults(seed=42)
    return parser.parse_args()


# Validate that the requested run can produce at least one saved posterior sample.
def validate_args(args: argparse.Namespace) -> None:
    """Fail early for settings that cannot produce usable posterior samples."""
    if args.sample_size < 8:
        raise ValueError("--sample-size must be at least 8 so train and test splits are non-empty.")
    if args.n_chains < 2:
        raise ValueError("--n-chains must be at least 2 for multichain diagnostics.")
    if args.n_iterations <= 1:
        raise ValueError("--n-iterations must be greater than 1.")
    if args.burn_in < 0 or args.burn_in >= args.n_iterations:
        raise ValueError("--burn-in must be non-negative and smaller than --n-iterations.")
    if args.thin < 1:
        raise ValueError("--thin must be at least 1.")
    if (args.n_iterations - args.burn_in) // args.thin < 1:
        raise ValueError("--n-iterations, --burn-in, and --thin must leave at least one saved sample.")
    unsupported_data_cases = sorted(set(args.data_cases) - set(DATA_GENERATORS))
    if unsupported_data_cases:
        raise ValueError(f"Unsupported --data-cases values: {unsupported_data_cases}.")
    for data_case in args.data_cases:
        unsupported_data_dims = sorted(set(args.data_dimensions) - set(DATA_GENERATORS[data_case]))
        if unsupported_data_dims:
            raise ValueError(
                f"Unsupported --data-dimensions values for {data_case}: {unsupported_data_dims}."
            )
    if any(D < 1 for D in args.posterior_dimensions):
        raise ValueError("--posterior-dimensions values must be positive integers.")
    if args.rstiefel_rscol is not None and args.rstiefel_rscol < 1:
        raise ValueError("--rstiefel-rscol must be a positive integer when provided.")
    if args.include_w_variants and args.w_variants is not None:
        raise ValueError("Use either --include-w-variants or --w-variants, not both.")


# Convert CLI W variant choices into internal variant values used by run_one.
def selected_w_variants(args: argparse.Namespace) -> List[Optional[str]]:
    """Return requested variants, using None internally for the full model."""
    if args.w_variants is not None:
        variant_names = list(args.w_variants)
    elif args.include_w_variants:
        variant_names = ["full", *W_VARIANTS]
    else:
        variant_names = ["full"]

    unique_names: List[str] = []
    for name in variant_names:
        if name not in unique_names:
            unique_names.append(name)

    return [None if name == "full" else name for name in unique_names]


# Convert numpy arrays, numpy scalars, paths, and nested containers to JSON-safe values.
def to_jsonable(value: Any) -> Any:
    """Convert numpy/path objects into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


# Save a mapping as formatted JSON, creating the output folder first.
def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file and create its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


# Run a plotting function defensively so diagnostics do not stop completed sampling.
def safe_plot(description: str, plotter, *args, **kwargs) -> None:
    """Run one plotting function without letting plot failures hide samples."""
    try:
        plotter(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - plotting should not stop sampling outputs
        print(f"Warning: skipped {description} ({exc})", file=sys.stderr)


# Return the first available value when full and variant samplers use different keys.
def first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    """Return the first value found among several possible key names."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


# Collect one sampled parameter from every chain and reshape it for plotting.
def sample_chains_for_key(results: Mapping[str, Any], aliases: Sequence[str]) -> Optional[List[np.ndarray]]:
    """Extract one parameter across chains, flattening matrices for plots."""
    chains = []
    for samples in results.get("chains_samples", []):
        value = first_present(samples, aliases)
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        chains.append(arr)
    return chains or None


# Create trace, density, ACF, and prediction plots for W-variant sampler results.
def create_runner_diagnostics(results: Mapping[str, Any], y_test: np.ndarray, output_dir: Path) -> None:
    """Create diagnostics for variant outputs using shared BDR_plot functions.

    Full-model samplers already create their own diagnostics through
    run_multichain_analysis. Variants use different metric keys, so the runner
    creates their trace, density, ACF, and prediction plots here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for param_name, aliases in DIAGNOSTIC_SAMPLE_KEYS:
        chains = sample_chains_for_key(results, aliases)
        if chains is None:
            continue
        safe_name = param_name.replace("/", "_")
        safe_plot(
            f"{param_name} trace",
            plot_trace,
            chains,
            param_name,
            save_path=str(output_dir / f"trace_{safe_name}.pdf"),
        )
        safe_plot(
            f"{param_name} density",
            plot_density,
            chains,
            param_name,
            save_path=str(output_dir / f"density_{safe_name}.pdf"),
        )
        safe_plot(
            f"{param_name} autocorrelation",
            plot_autocorrelation,
            chains[0],
            f"{param_name} chain 1",
            save_path=str(output_dir / f"acf_{safe_name}_chain1.pdf"),
        )

    metrics = results.get("chains_metrics", [])
    if metrics:
        first_metrics = metrics[0]
        y_pred = first_present(first_metrics, ("y_pred_mean", "pred_mean"))
        ci_bounds = first_present(first_metrics, ("y_pred_quantiles", "pred_quantiles"))
        if y_pred is not None and ci_bounds is not None:
            safe_plot(
                "actual vs predicted",
                plot_actual_vs_predicted,
                y_test,
                np.asarray(y_pred),
                np.asarray(ci_bounds),
                save_path=str(output_dir / "actual_vs_predicted.pdf"),
            )


# Save the generated SVD/matrix-Langevin initial values and priors for reproducibility.
def save_initial_values_and_priors(
    config: Mapping[str, Any],
    output_dir: Path,
    init_values: Mapping[str, Any],
) -> None:
    """Persist SVD/matrix-Langevin initial values used to start the run."""
    array_keys = [
    "W_init",
    "W_fixed",
    "M_init",
        "V_init",
        "Lambda_init",
        "prior_M",
        "prior_V",
        "prior_Lambda",
    ]
    arrays = {
        key: np.asarray(init_values.get(key, config.get(key)))
        for key in array_keys
        if init_values.get(key, config.get(key)) is not None
    }
    if arrays:
        np.savez(output_dir / "initial_values_and_priors.npz", **arrays)


# Build the compact JSON summary for one completed data-case/data-D/posterior-D/layer/variant run.
def summarize_results(
    results: Mapping[str, Any],
    *,
    data_case: str,
    data_dim: int,
    posterior_D: int,
    layer: int,
    seed: int,
    run_dir: Path,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the compact per-run summary saved as results_summary.json."""
    return {
        "data_case": data_case,
        "data_dim": data_dim,
        "posterior_D": posterior_D,
        "D": posterior_D,
        "data_generator": f"{data_case}_{data_dim}d",
        "layer": layer,
        "variant": config.get("variant", "full") or "full",
        "W_fixed_source": "data_generation_true_W" if config.get("variant") == "W_Known" else None,
        "mv_sampler": config.get("mv_sampler", "python"),
        "rstiefel_rscol": config.get("rstiefel_rscol"),
        "kernel_type": config["kernel_type"],
        "seed": seed,
        "n_chains": config["n_chains"],
        "n_iterations": config["n_iterations"],
        "burn_in": config["burn_in"],
        "thin": config["thin"],
        "prior_hyperparameters": {
            "tau2_alpha1": TAU2_ALPHA1,
            "tau2_alpha2": TAU2_ALPHA2,
            "gamma_shape": GAMMA_SHAPE,
            "g_rate": GAMMA_RATE_G_AND_THETA_Y,
            "theta_y_rate": GAMMA_RATE_G_AND_THETA_Y,
            "theta_q_rate": GAMMA_RATE_THETA_Q,
            "theta_r_rate": GAMMA_RATE_THETA_R,
            "lambda_gamma_shape": LAMBDA_GAMMA_SHAPE,
            "lambda_gamma_rate": LAMBDA_GAMMA_RATE,
            "sliding_window_l": SLIDING_WINDOW_L,
            "sliding_window_u": SLIDING_WINDOW_U,
        },
        "initial_values": {
            "tau2": TAU2_INIT,
            "g": G_INIT,
            "theta": THETA_INIT,
        },
        "n_train": int(data["n_train"]),
        "n_test": int(data["n_test"]),
        "X_train_shape": list(data["X_train"].shape),
        "X_test_shape": list(data["X_test"].shape),
        "metrics_summary": results.get("metrics_summary", {}),
        "convergence": results.get("convergence", {}),
        "computation_times": results.get("computation_times", []),
        "parameter_diagnostics_error": results.get("parameter_diagnostics_error"),
        "output_dir": str(run_dir),
    }


# Write the top-level CSV that compares all requested data-case/data-D/posterior-D/layer/variant runs.
def write_aggregate_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write the top-level CSV that compares data-case/data-D/posterior-D/layer/variant runs."""
    fieldnames = [
        "data_case",
        "data_dim",
        "posterior_D",
        "layer",
        "variant",
        "mv_sampler",
        "rstiefel_rscol",
        "status",
        "rmspe_mean",
        "nsme_mean",
        "crps_mean",
        "bic_mean",
        "mlppd_mean",
        "cp_mean",
        "alci_mean",
        "total_seconds",
        "output_dir",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# Convert one run summary into the normalized row format used in simulation_summary.csv.
def aggregate_row(summary: Mapping[str, Any], status: str, error: str = "") -> Dict[str, Any]:
    """Convert a per-run summary into one CSV row."""
    metrics = summary.get("metrics_summary", {})

    # Extract a metric mean while accepting lowercase full-model keys and uppercase variant keys.
    def metric_mean(name: str) -> Any:
        metric = metrics.get(name, metrics.get(name.upper(), {}))
        if isinstance(metric, Mapping):
            return metric.get("mean", "")
        return ""

    return {
        "data_case": summary.get("data_case"),
        "data_dim": summary.get("data_dim"),
        "posterior_D": summary.get("posterior_D", summary.get("D")),
        "layer": summary.get("layer"),
        "variant": summary.get("variant", "full"),
        "mv_sampler": summary.get("mv_sampler", "python"),
        "rstiefel_rscol": summary.get("rstiefel_rscol"),
        "status": status,
        "rmspe_mean": metric_mean("rmspe"),
        "nsme_mean": metric_mean("nsme"),
        "crps_mean": metric_mean("crps"),
        "bic_mean": metric_mean("bic"),
        "mlppd_mean": metric_mean("mlppd"),
        "cp_mean": metric_mean("cp"),
        "alci_mean": metric_mean("alci"),
        "total_seconds": float(np.sum(summary.get("computation_times", []))),
        "output_dir": summary.get("output_dir"),
        "error": error,
    }


# Generate posterior-D SVD/matrix-Langevin initial W, M, Lambda, V, prior_M, and prior_V.
def build_initial_values(posterior_D: int, p: int, seed: int) -> Dict[str, Any]:
    """Generate M, Lambda, V, W, prior_M, and prior_V from the SVD strategy."""
    if posterior_D == 1:
        return initialize_M_Lambda_V_W_D1(p=p, D=1, seed=seed)
    return initialize_M_Lambda_V_W_Dgeneral(p=p, D=posterior_D, seed=seed)


# Return scalar theta for posterior D=1 and a length-D vector for full D>1 samplers.
def theta_init_for_dimension(posterior_D: int) -> Any:
    """Full-model D>1 samplers expect vector theta initials."""
    return THETA_INIT if posterior_D == 1 else np.full(posterior_D, THETA_INIT)


# Assemble the exact keyword arguments sent into run_multichain_analysis.
def build_config(
    args: argparse.Namespace,
    posterior_D: int,
    layer: int,
    p: int,
    n_train: int,
    output_dir: Path,
    init_values: Mapping[str, Any],
    true_W: Optional[np.ndarray] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the exact keyword arguments passed to run_multichain_analysis.

    Full models use the preset builders from run_multichains.py and receive
    W/M/V/Lambda initial values because W is sampled.

    W variants do not sample W:
    - W_Known fixes W at the true W returned by the data generator.
    - No_W uses X directly, so no W argument is passed.
    - No_W_Selective uses the first D columns of X.

    The prior and initial hyperparameter constants are kept the same for every
    model family so the output folders are comparable.
    """
    if variant is not None:
        config: Dict[str, Any] = {
            "D": posterior_D,
            "layer": layer,
            "variant": variant,
            "n_chains": args.n_chains,
            "n_iterations": args.n_iterations,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "kernel_type": KERNEL_TYPE,
            "mv_sampler": args.mv_sampler,
            "rstiefel_rscol": args.rstiefel_rscol,
            "use_mle_all": False,
            "use_mle_tau2": False,
            "use_mle_g": False,
            "use_mle_theta": False,
            "use_mle_g_y": False,
            "use_mle_theta_y": False,
            "use_tf_gradients": False,
            "alpha1_tau2": TAU2_ALPHA1,
            "alpha2_tau2": TAU2_ALPHA2,
            "beta1_g": GAMMA_SHAPE,
            "beta2_g": GAMMA_RATE_G_AND_THETA_Y,
            "gamma1_theta": GAMMA_SHAPE,
            "gamma2_theta": GAMMA_RATE_G_AND_THETA_Y,
            "gamma2_theta_y": GAMMA_RATE_G_AND_THETA_Y,
            "gamma2_theta_q": GAMMA_RATE_THETA_Q,
            "gamma2_theta_r": GAMMA_RATE_THETA_R,
            "l_g": SLIDING_WINDOW_L,
            "u_g": SLIDING_WINDOW_U,
            "l_theta": SLIDING_WINDOW_L,
            "u_theta": SLIDING_WINDOW_U,
            "tau2_y_init": TAU2_INIT,
            "tau2_q_init": TAU2_INIT,
            "tau2_r_init": TAU2_INIT,
            "g_y_init": G_INIT,
            "g_q_init": G_INIT,
            "g_r_init": G_INIT,
            "theta_y_init": THETA_INIT,
            "theta_q_init": THETA_INIT,
            "theta_r_init": THETA_INIT,
            "output_dir": str(output_dir),
            "save_samples": not args.no_save_samples,
            # Variant diagnostic functions have different prediction keys; the
            # runner creates variant plots after sampling using the shared plot API.
            "save_plots": False,
            "compute_parameter_diagnostics": args.parameter_diagnostics,
            "verbose": args.verbose,
        }
        if variant == "W_Known":
            if true_W is None:
                raise ValueError("W_Known requires the true W returned by the data generator.")
            true_W = np.asarray(true_W)
            if true_W.shape != (p, posterior_D):
                raise ValueError(
                    f"W_Known requires true W shape ({p}, {posterior_D}), got {true_W.shape}. "
                    "Use matching --data-dimensions and --posterior-dimensions for W_Known."
                )
            # Fix W at the true projection matrix from Data_generation.py.
            # There will be no posterior W samples because W is known/fixed.
            config["W_fixed"] = true_W
        elif variant == "No_W_Selective":
            # Use the first posterior_D input columns. Change this line if you
            # want a different subset, for example np.array([0, 3]) for D=2.
            config["column_indices"] = np.arange(posterior_D)
        return config

    # Full BDR model: W is sampled, so pass the SVD/matrix-Langevin initial
    # values and priors into the existing full-model configuration builder.
    config_overrides = {
        "n_chains": args.n_chains,
        "n_iterations": args.n_iterations,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "kernel_type": KERNEL_TYPE,
        "mv_sampler": args.mv_sampler,
        "rstiefel_rscol": args.rstiefel_rscol,
        "W_init": init_values["W_init"],
        "M_init": init_values["M_init"],
        "V_init": init_values["V_init"],
        "Lambda_init": init_values["Lambda_init"],
        "prior_M": init_values["prior_M"],
        "prior_V": init_values["prior_V"],
        "use_mle_all": False,
        "use_mle_tau2": False,
        "use_mle_g": False,
        "use_mle_theta": False,
        "use_mle_g_y": False,
        "use_mle_theta_y": False,
        "use_tf_gradients": False,
        "alpha1_tau2": TAU2_ALPHA1,
        "alpha2_tau2": TAU2_ALPHA2,
        "beta1_g": GAMMA_SHAPE,
        "beta2_g": GAMMA_RATE_G_AND_THETA_Y,
        "gamma1_theta": GAMMA_SHAPE,
        "gamma2_theta": GAMMA_RATE_G_AND_THETA_Y,
        "gamma2_theta_y": GAMMA_RATE_G_AND_THETA_Y,
        "gamma2_theta_q": GAMMA_RATE_THETA_Q,
        "gamma2_theta_r": GAMMA_RATE_THETA_R,
        "l_g": SLIDING_WINDOW_L,
        "u_g": SLIDING_WINDOW_U,
        "l_theta": SLIDING_WINDOW_L,
        "u_theta": SLIDING_WINDOW_U,
        "tau2_y_init": TAU2_INIT,
        "tau2_q_init": TAU2_INIT,
        "tau2_r_init": TAU2_INIT,
        "g_y_init": G_INIT,
        "g_q_init": G_INIT,
        "g_r_init": G_INIT,
        "theta_y_init": theta_init_for_dimension(posterior_D),
        "theta_q_init": theta_init_for_dimension(posterior_D),
        "theta_r_init": theta_init_for_dimension(posterior_D),
        "output_dir": str(output_dir),
        "save_samples": not args.no_save_samples,
        "save_plots": not args.no_plots,
        "compute_parameter_diagnostics": args.parameter_diagnostics,
        "verbose": args.verbose,
    }
    if (posterior_D, layer) in CONFIG_FUNCTIONS:
        config = get_config_for(D=posterior_D, layer=layer, **config_overrides)
    else:
        # get_config_for falls back to generic p=10/n_train=100 values, so use
        # get_default_config directly when there is no preset for this D.
        config = get_default_config(D=posterior_D, layer=layer, n_train=n_train, p=p)
        config.update(config_overrides)
    return config


# Create a stable output directory name for a full or W-variant run.
def run_name_for(
    data_case: str,
    data_dim: int,
    posterior_D: int,
    layer: int,
    sample_size: int,
    variant: Optional[str] = None,
) -> str:
    """Create stable output folder names for full and variant runs."""
    variant_part = f"_{variant}" if variant else ""
    return f"{data_case}_dataD{data_dim}_postD{posterior_D}_L{layer}{variant_part}_n{sample_size}"


# Execute one complete simulation: generate data, initialize, sample, summarize, and plot.
def run_one(
    args: argparse.Namespace,
    data_case: str,
    data_dim: int,
    posterior_D: int,
    layer: int,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one data-case/data-D/posterior-D/layer/model-variant combination end to end."""
    run_name = run_name_for(data_case, data_dim, posterior_D, layer, args.sample_size, variant)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data generation is repeated per run so every output folder is
    # self-contained. The fixed seed keeps the train/test split identical.
    data = DATA_GENERATORS[data_case][data_dim](n=args.sample_size, seed=args.seed)
    p = int(data["X_train"].shape[1])
    if posterior_D > p:
        raise ValueError(
            f"posterior_D={posterior_D} cannot exceed the input dimension p={p} "
            f"from the data_dim={data_dim} generator."
        )

    # These initial values are saved even for W variants. W_Known uses the true
    # W from the data generator as W_fixed; W_init documents the comparable setup.
    init_values = build_initial_values(posterior_D=posterior_D, p=p, seed=args.seed)
    config = build_config(
        args,
        posterior_D=posterior_D,
        layer=layer,
        p=p,
        n_train=int(data["n_train"]),
        output_dir=run_dir,
        init_values=init_values,
        true_W=np.asarray(data.get("W")) if "W" in data else None,
        variant=variant,
    )
    config_log = {
        **config,
        "seed": args.seed,
        "data_case": data_case,
        "data_dim": data_dim,
        "posterior_D": posterior_D,
        "W_fixed_source": "data_generation_true_W" if variant == "W_Known" else None,
        "lambda_gamma_shape": LAMBDA_GAMMA_SHAPE,
        "lambda_gamma_rate": LAMBDA_GAMMA_RATE,
        "initialization": (
            "F_ij iid N(0,1); SVD F = M_full L V^T; "
            "M_init=M_full[:,:D], Lambda_init=diag(L), V_init=V; "
            "W_init~ML(M_init Lambda_init V_init^T), "
            "prior_M~ML(M_init), prior_V~ML(V_init)"
        ),
    }
    write_json(run_dir / "config_used.json", config_log)
    save_initial_values_and_priors(config, run_dir, init_values)

    print(f"\nRunning {run_name}")
    print(f"  data: X_train={data['X_train'].shape}, X_test={data['X_test'].shape}")
    print(f"  dimensions: data_case={data_case}, data_dim={data_dim}, posterior_D={posterior_D}, layer={layer}")
    if variant:
        print(f"  variant: {variant}")
    print(f"  kernel: {KERNEL_TYPE}, chains={args.n_chains}, iterations={args.n_iterations}")
    print(f"  M/V sampler: {args.mv_sampler}" + (f" (rscol={args.rstiefel_rscol})" if args.rstiefel_rscol else ""))

    results = run_multichain_analysis(
        Y_train=data["y_train"],
        X_train=data["X_train"],
        Y_test=data["y_test"],
        X_test=data["X_test"],
        **config,
    )

    summary = summarize_results(
        results,
        data_case=data_case,
        data_dim=data_dim,
        posterior_D=posterior_D,
        layer=layer,
        seed=args.seed,
        run_dir=run_dir,
        config=config,
        data=data,
    )
    write_json(run_dir / "results_summary.json", summary)

    if not args.no_plots:
        if variant is not None:
            # Variant samplers use pred_mean/pred_quantiles keys. This helper
            # normalizes those keys and writes diagnostics in the run folder.
            create_runner_diagnostics(results, data["y_test"], run_dir)
        plot_metrics_boxplot(results["chains_metrics"], save_path=str(run_dir / "metrics_boxplot.pdf"))
        plot_metrics_comparison_table(results["metrics_summary"], save_path=str(run_dir / "metrics_summary_table.pdf"))

    return summary


# Command-line entry point that loops over all requested data cases, data dimensions, posterior dimensions, layers, and variants.
def main() -> int:
    """CLI entry point."""
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    requested_variants = selected_w_variants(args)

    for data_case in args.data_cases:
        for data_dim in args.data_dimensions:
            for posterior_D in args.posterior_dimensions:
                for layer in args.layers:
                    for variant in requested_variants:
                        partial_summary = {
                            "data_case": data_case,
                            "data_dim": data_dim,
                            "posterior_D": posterior_D,
                            "D": posterior_D,
                            "layer": layer,
                            "variant": variant or "full",
                            "mv_sampler": args.mv_sampler,
                            "rstiefel_rscol": args.rstiefel_rscol,
                            "output_dir": str(
                                args.output_dir
                                / run_name_for(data_case, data_dim, posterior_D, layer, args.sample_size, variant)
                            ),
                            "computation_times": [],
                            "metrics_summary": {},
                        }
                        if variant == "W_Known" and posterior_D != data_dim:
                            error = (
                                "Skipped W_Known because the true W from the data generator has "
                                f"{data_dim} column(s), but posterior_D={posterior_D}. "
                                "Use matching --data-dimensions and --posterior-dimensions for W_Known."
                            )
                            rows.append(aggregate_row(partial_summary, status="skipped", error=error))
                            write_json(
                                Path(partial_summary["output_dir"]) / "skipped.json",
                                {"status": "skipped", "reason": error},
                            )
                            print(
                                (
                                    f"\nSkipped data_case={data_case}, data_dim={data_dim}, "
                                    f"posterior_D={posterior_D}, layer={layer}, variant=W_Known: {error}"
                                ),
                                file=sys.stderr,
                            )
                            continue
                        try:
                            summary = run_one(
                                args,
                                data_case=data_case,
                                data_dim=data_dim,
                                posterior_D=posterior_D,
                                layer=layer,
                                variant=variant,
                            )
                            rows.append(aggregate_row(summary, status="ok"))
                        except Exception as exc:
                            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                            rows.append(aggregate_row(partial_summary, status="failed", error=error))
                            write_json(
                                Path(partial_summary["output_dir"]) / "error.json",
                                {"error": error, "traceback": traceback.format_exc()},
                            )
                            print(
                                (
                                    f"\nFailed data_case={data_case}, data_dim={data_dim}, "
                                    f"posterior_D={posterior_D}, layer={layer}, "
                                    f"variant={variant or 'full'}: {error}"
                                ),
                                file=sys.stderr,
                            )
                            if not args.continue_on_error:
                                write_aggregate_csv(args.output_dir / "simulation_summary.csv", rows)
                                raise

    write_aggregate_csv(args.output_dir / "simulation_summary.csv", rows)
    print(f"\nSimulation summary written to: {args.output_dir / 'simulation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
