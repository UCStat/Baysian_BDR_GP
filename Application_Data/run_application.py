#!/usr/bin/env python3
"""
Run BDR posterior sampling on application datasets.

This script mirrors the workflow in `Data Generation/run_simulation.py`, but it
loads real application data instead of generating synthetic Case 1/Case 2 data.

Application datasets:
- Elliptical_PDE: loads X_1/Y_1 and/or X_2/Y_2 from NumPy files and uses the
  separable Matern-3/2 kernel.
- Onera M6: loads onera_m6.csv, uses x_* columns as inputs, lift/drag as
  selectable targets, and uses the separable squared exponential kernel.

Implementation map:
1. Parse CLI options in `parse_args`.
2. Load selected application datasets with `load_application_datasets`.
3. Create a reproducible 80/20 train/test split with seed 42.
4. Build posterior-D SVD/matrix-Langevin initial values.
5. Build a full-model, No_W, or No_W_Selective config.
6. Call `run_multichain_analysis`, which dispatches to the correct sampler.
7. Save configuration, posterior samples, summaries, diagnostics, and plots.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# The repository uses plain Python modules inside directories with spaces.
# Add those folders to sys.path so this script can be launched directly.
for folder in [
    REPO_ROOT,
    REPO_ROOT / "Application_Data",
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

from BDR_plot import (  # type: ignore[import]
    plot_metrics_boxplot,
    plot_metrics_comparison_table,
)
from run_multichains import (  # type: ignore[import]
    CONFIG_FUNCTIONS,
    get_config_for,
    get_default_config,
    run_multichain_analysis,
)
from run_simulation import (  # type: ignore[import]
    GAMMA_RATE_G_AND_THETA_Y,
    GAMMA_RATE_THETA_Q,
    GAMMA_RATE_THETA_R,
    GAMMA_SHAPE,
    G_INIT,
    LAMBDA_GAMMA_RATE,
    LAMBDA_GAMMA_SHAPE,
    SLIDING_WINDOW_L,
    SLIDING_WINDOW_U,
    TAU2_ALPHA1,
    TAU2_ALPHA2,
    TAU2_INIT,
    THETA_INIT,
    build_initial_values,
    create_runner_diagnostics,
    save_initial_values_and_priors,
    theta_init_for_dimension,
    to_jsonable,
    write_json,
)


# Each application dataset has a fixed kernel requested for the real-data run.
# These names must match the covariance module options.
APPLICATION_KERNELS = {
    "elliptical_pde": "separable_matern32",
    "onera": "separable_squared_exponential",
}

# Real application data do not provide a known true W. The application runner
# therefore supports the full W-sampled model and the two W-not-sampled variants.
APPLICATION_VARIANTS = ("full", "No_W", "No_W_Selective")


@dataclass(frozen=True)
class ApplicationDataset:
    """Container for one application target before train/test splitting."""

    application: str
    target: str
    X: np.ndarray
    y: np.ndarray
    kernel_type: str
    source_path: str


# Parse command-line options for application data selection, MCMC settings, outputs, and variants.
def parse_args() -> argparse.Namespace:
    """Read command-line options for the application-data runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run BDR posterior sampling on Elliptical_PDE and Onera M6 data "
            "with an 80/20 train/test split."
        )
    )
    parser.add_argument(
        "--applications",
        nargs="+",
        choices=sorted(APPLICATION_KERNELS),
        default=["elliptical_pde", "onera"],
        help="Application dataset(s) to run.",
    )
    parser.add_argument(
        "--elliptical-outputs",
        type=int,
        nargs="+",
        default=[1],
        choices=[1, 2],
        help="Elliptical_PDE output pairs to run: X_1/Y_1 and/or X_2/Y_2.",
    )
    parser.add_argument(
        "--onera-targets",
        nargs="+",
        default=["lift"],
        choices=["lift", "drag"],
        help="Onera M6 response column(s) to model.",
    )
    parser.add_argument(
        "--posterior-dimensions",
        "--dimensions",
        dest="posterior_dimensions",
        type=int,
        nargs="+",
        default=[1],
        help="Posterior/model reduced dimension(s) D used by the BDR sampler.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1],
        choices=[1, 2, 3],
        help="Model depth(s) to run: 1, 2, and/or 3.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(APPLICATION_VARIANTS),
        choices=list(APPLICATION_VARIANTS),
        help="Model variants to run. Use 'full' for W-sampled BDR.",
    )
    parser.add_argument("--n-chains", type=int, default=3, help="Number of MCMC chains.")
    parser.add_argument("--n-iterations", type=int, default=6, help="MCMC iterations per chain.")
    parser.add_argument("--burn-in", type=int, default=2, help="Burn-in iterations.")
    parser.add_argument("--thin", type=int, default=3, help="Thinning interval.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and initialization.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of rows used for training. Default is 0.8.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap used before splitting; useful for quick smoke tests.",
    )
    parser.add_argument(
        "--mv-sampler",
        choices=["python", "rstiefel"],
        default="python",
        help="Backend for posterior M and V updates in full W-sampled models.",
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
        default=SCRIPT_DIR / "application_outputs",
        help="Directory for all application outputs.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic and metric plot generation.")
    parser.add_argument("--no-save-samples", action="store_true", help="Do not save posterior samples pickle files.")
    parser.add_argument(
        "--parameter-diagnostics",
        action="store_true",
        help="Also attempt R/coda parameter diagnostics. Requires rpy2, R, and coda.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs if one application/model combination fails.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print sampler progress.")
    return parser.parse_args()


# Validate command-line settings before any sampler is launched.
def validate_args(args: argparse.Namespace) -> None:
    """Fail early for invalid dimensions, split settings, or MCMC settings."""
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
    if any(D < 1 for D in args.posterior_dimensions):
        raise ValueError("--posterior-dimensions values must be positive integers.")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    if args.max_rows is not None and args.max_rows < 8:
        raise ValueError("--max-rows must be at least 8 when provided.")
    if args.rstiefel_rscol is not None and args.rstiefel_rscol < 1:
        raise ValueError("--rstiefel-rscol must be a positive integer when provided.")


# Load one Elliptical_PDE X_k/Y_k pair from NumPy files.
def load_elliptical_pde(output_id: int) -> ApplicationDataset:
    """Load Elliptical_PDE X_k/Y_k arrays for one requested output id."""
    base_dir = SCRIPT_DIR / "Elliptical_PDE"
    x_path = base_dir / f"X_{output_id}.npy"
    y_path = base_dir / f"Y_{output_id}.npy"
    X = np.asarray(np.load(x_path), dtype=float)
    y = np.asarray(np.load(y_path), dtype=float).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{x_path.name} and {y_path.name} row counts do not match.")
    return ApplicationDataset(
        application="elliptical_pde",
        target=f"Y_{output_id}",
        X=X,
        y=y,
        kernel_type=APPLICATION_KERNELS["elliptical_pde"],
        source_path=f"{x_path}; {y_path}",
    )


# Load Onera M6 CSV, using x_* columns as inputs and one scalar target.
def load_onera(target: str) -> ApplicationDataset:
    """Load the Onera M6 CSV for one target column, either lift or drag."""
    csv_path = SCRIPT_DIR / "Onera M6" / "onera_m6.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    x_columns = sorted(
        [name for name in rows[0] if name.startswith("x_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )
    if target not in rows[0]:
        raise ValueError(f"Target column {target!r} not found in {csv_path}.")

    X_values: List[List[float]] = []
    y_values: List[float] = []
    for row in rows:
        try:
            X_values.append([float(row[column]) for column in x_columns])
            y_values.append(float(row[target]))
        except (TypeError, ValueError):
            continue

    X = np.asarray(X_values, dtype=float)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Onera X and y row counts do not match after parsing.")
    return ApplicationDataset(
        application="onera",
        target=target,
        X=X,
        y=y,
        kernel_type=APPLICATION_KERNELS["onera"],
        source_path=str(csv_path),
    )


# Build the selected list of application datasets from CLI choices.
def load_application_datasets(args: argparse.Namespace) -> List[ApplicationDataset]:
    """Load all requested application datasets and targets."""
    datasets: List[ApplicationDataset] = []
    if "elliptical_pde" in args.applications:
        for output_id in args.elliptical_outputs:
            datasets.append(load_elliptical_pde(output_id))
    if "onera" in args.applications:
        for target in args.onera_targets:
            datasets.append(load_onera(target))
    return datasets


# Create a reproducible optional row cap plus 80/20 train/test split.
def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
    max_rows: Optional[int],
) -> Dict[str, Any]:
    """Shuffle rows reproducibly, optionally cap rows, then split train/test."""
    rng = np.random.default_rng(seed)
    n_total = int(X.shape[0])
    order = rng.permutation(n_total)
    if max_rows is not None:
        order = order[: min(max_rows, n_total)]
    X_shuffled = X[order]
    y_shuffled = y[order]

    n_used = int(X_shuffled.shape[0])
    n_train = int(np.floor(train_fraction * n_used))
    n_train = min(max(n_train, 1), n_used - 1)

    return {
        "X_train": X_shuffled[:n_train],
        "y_train": y_shuffled[:n_train],
        "X_test": X_shuffled[n_train:],
        "y_test": y_shuffled[n_train:],
        "n_total": n_total,
        "n_used": n_used,
        "n_train": n_train,
        "n_test": n_used - n_train,
        "row_indices_used": order.tolist(),
    }


# Build a JSON-safe summary for one loaded application dataset.
def data_summary(dataset: ApplicationDataset, split: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize application data shape, target, kernel, and split details."""
    return {
        "application": dataset.application,
        "target": dataset.target,
        "source_path": dataset.source_path,
        "kernel_type": dataset.kernel_type,
        "n_total": split["n_total"],
        "n_used": split["n_used"],
        "n_train": split["n_train"],
        "n_test": split["n_test"],
        "p": int(dataset.X.shape[1]),
        "X_train_shape": list(split["X_train"].shape),
        "X_test_shape": list(split["X_test"].shape),
        "y_train_shape": list(split["y_train"].shape),
        "y_test_shape": list(split["y_test"].shape),
    }


# Return scalar theta for D=1 and length-D vectors for separable D>1 kernels.
def application_theta_init(posterior_D: int) -> Any:
    """Return the theta initial value expected by the selected sampler."""
    return theta_init_for_dimension(posterior_D)


# Assemble keyword arguments sent into run_multichain_analysis for application data.
def build_application_config(
    args: argparse.Namespace,
    *,
    posterior_D: int,
    layer: int,
    p: int,
    n_train: int,
    output_dir: Path,
    init_values: Mapping[str, Any],
    kernel_type: str,
    variant: str,
) -> Dict[str, Any]:
    """Build full, No_W, or No_W_Selective configs for application data.

    Application data do not include a true W, so W_Known is intentionally not
    supported. Full models sample W, M, V, Lambda, and kernel hyperparameters.
    No_W and No_W_Selective do not sample W but still posterior-sample the
    available model hyperparameters.
    """
    common = {
        "D": posterior_D,
        "layer": layer,
        "n_chains": args.n_chains,
        "n_iterations": args.n_iterations,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "kernel_type": kernel_type,
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
        "theta_y_init": application_theta_init(posterior_D),
        "theta_q_init": application_theta_init(posterior_D),
        "theta_r_init": application_theta_init(posterior_D),
        "output_dir": str(output_dir),
        "save_samples": not args.no_save_samples,
        "compute_parameter_diagnostics": args.parameter_diagnostics,
        "verbose": args.verbose,
    }

    if variant != "full":
        config = {
            **common,
            "variant": variant,
            "mv_sampler": args.mv_sampler,
            "rstiefel_rscol": args.rstiefel_rscol,
            "theta_y_init": THETA_INIT,
            "theta_q_init": THETA_INIT,
            "theta_r_init": THETA_INIT,
            # Variant plot keys differ from the full model. The application
            # runner creates variant diagnostics after sampling.
            "save_plots": False,
        }
        if variant == "No_W_Selective":
            config["column_indices"] = np.arange(posterior_D)
        return config

    config_overrides = {
        "mv_sampler": args.mv_sampler,
        "rstiefel_rscol": args.rstiefel_rscol,
        "W_init": init_values["W_init"],
        "M_init": init_values["M_init"],
        "V_init": init_values["V_init"],
        "Lambda_init": init_values["Lambda_init"],
        "prior_M": init_values["prior_M"],
        "prior_V": init_values["prior_V"],
        "save_plots": not args.no_plots,
    }
    config_overrides.update({key: value for key, value in common.items() if key not in {"D", "layer"}})
    if (posterior_D, layer) in CONFIG_FUNCTIONS:
        return get_config_for(D=posterior_D, layer=layer, **config_overrides)

    config = get_default_config(D=posterior_D, layer=layer, n_train=n_train, p=p)
    config.update(config_overrides)
    return config


# Create a stable output directory name for one application run.
def run_name_for(
    application: str,
    target: str,
    posterior_D: int,
    layer: int,
    variant: str,
) -> str:
    """Create stable output folder names for application runs."""
    safe_target = target.replace("/", "_").replace(" ", "_")
    variant_part = "" if variant == "full" else f"_{variant}"
    return f"{application}_{safe_target}_postD{posterior_D}_L{layer}{variant_part}"


# Extract a metric mean while accepting full-model and variant metric key styles.
def metric_mean(metrics_summary: Mapping[str, Any], name: str) -> Any:
    """Read one metric mean from lowercase or uppercase metric summaries."""
    metric = metrics_summary.get(name, metrics_summary.get(name.upper(), {}))
    if isinstance(metric, Mapping):
        return metric.get("mean", "")
    return ""


# Build the compact JSON summary for one completed application run.
def summarize_application_results(
    results: Mapping[str, Any],
    *,
    dataset: ApplicationDataset,
    split: Mapping[str, Any],
    posterior_D: int,
    layer: int,
    variant: str,
    seed: int,
    train_fraction: float,
    run_dir: Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the per-run summary saved as results_summary.json."""
    return {
        **data_summary(dataset, split),
        "posterior_D": posterior_D,
        "D": posterior_D,
        "layer": layer,
        "variant": variant,
        "mv_sampler": config.get("mv_sampler", "python"),
        "rstiefel_rscol": config.get("rstiefel_rscol"),
        "seed": seed,
        "train_fraction": train_fraction,
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
        "metrics_summary": results.get("metrics_summary", {}),
        "convergence": results.get("convergence", {}),
        "computation_times": results.get("computation_times", []),
        "parameter_diagnostics_error": results.get("parameter_diagnostics_error"),
        "output_dir": str(run_dir),
    }


# Convert one application summary into a normalized aggregate CSV row.
def aggregate_row(summary: Mapping[str, Any], status: str, error: str = "") -> Dict[str, Any]:
    """Convert a per-run application summary into one CSV row."""
    metrics = summary.get("metrics_summary", {})
    return {
        "application": summary.get("application"),
        "target": summary.get("target"),
        "kernel_type": summary.get("kernel_type"),
        "posterior_D": summary.get("posterior_D", summary.get("D")),
        "layer": summary.get("layer"),
        "variant": summary.get("variant", "full"),
        "mv_sampler": summary.get("mv_sampler", "python"),
        "rstiefel_rscol": summary.get("rstiefel_rscol"),
        "status": status,
        "n_used": summary.get("n_used"),
        "n_train": summary.get("n_train"),
        "n_test": summary.get("n_test"),
        "p": summary.get("p"),
        "rmspe_mean": metric_mean(metrics, "rmspe"),
        "nsme_mean": metric_mean(metrics, "nsme"),
        "crps_mean": metric_mean(metrics, "crps"),
        "bic_mean": metric_mean(metrics, "bic"),
        "mlppd_mean": metric_mean(metrics, "mlppd"),
        "cp_mean": metric_mean(metrics, "cp"),
        "alci_mean": metric_mean(metrics, "alci"),
        "total_seconds": float(np.sum(summary.get("computation_times", []))),
        "output_dir": summary.get("output_dir"),
        "error": error,
    }


# Write the top-level CSV that compares all requested application runs.
def write_application_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write application_summary.csv with one row per run."""
    fieldnames = [
        "application",
        "target",
        "kernel_type",
        "posterior_D",
        "layer",
        "variant",
        "mv_sampler",
        "rstiefel_rscol",
        "status",
        "n_used",
        "n_train",
        "n_test",
        "p",
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


# Execute one complete application run: split data, initialize, sample, summarize, and plot.
def run_one(
    args: argparse.Namespace,
    dataset: ApplicationDataset,
    posterior_D: int,
    layer: int,
    variant: str,
) -> Dict[str, Any]:
    """Run one application/target/posterior-D/layer/variant combination end to end."""
    split = train_test_split(
        dataset.X,
        dataset.y,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    p = int(split["X_train"].shape[1])
    if posterior_D > p:
        raise ValueError(f"posterior_D={posterior_D} cannot exceed input dimension p={p}.")

    run_name = run_name_for(dataset.application, dataset.target, posterior_D, layer, variant)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    init_values = build_initial_values(posterior_D=posterior_D, p=p, seed=args.seed)
    config = build_application_config(
        args,
        posterior_D=posterior_D,
        layer=layer,
        p=p,
        n_train=int(split["n_train"]),
        output_dir=run_dir,
        init_values=init_values,
        kernel_type=dataset.kernel_type,
        variant=variant,
    )
    config_log = {
        **config,
        "seed": args.seed,
        "application": dataset.application,
        "target": dataset.target,
        "source_path": dataset.source_path,
        "posterior_D": posterior_D,
        "train_fraction": args.train_fraction,
        "max_rows": args.max_rows,
        "row_indices_used": split["row_indices_used"],
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
    print(
        f"  data: X_train={split['X_train'].shape}, X_test={split['X_test'].shape}, "
        f"target={dataset.target}"
    )
    print(f"  application={dataset.application}, kernel={dataset.kernel_type}")
    print(f"  posterior_D={posterior_D}, layer={layer}, variant={variant}")
    print(f"  chains={args.n_chains}, iterations={args.n_iterations}, thin={args.thin}")
    print(f"  M/V sampler: {args.mv_sampler}" + (f" (rscol={args.rstiefel_rscol})" if args.rstiefel_rscol else ""))

    results = run_multichain_analysis(
        Y_train=split["y_train"],
        X_train=split["X_train"],
        Y_test=split["y_test"],
        X_test=split["X_test"],
        **config,
    )

    summary = summarize_application_results(
        results,
        dataset=dataset,
        split=split,
        posterior_D=posterior_D,
        layer=layer,
        variant=variant,
        seed=args.seed,
        train_fraction=args.train_fraction,
        run_dir=run_dir,
        config=config,
    )
    write_json(run_dir / "results_summary.json", summary)

    if not args.no_plots:
        if variant != "full":
            create_runner_diagnostics(results, split["y_test"], run_dir)
        plot_metrics_boxplot(results["chains_metrics"], save_path=str(run_dir / "metrics_boxplot.pdf"))
        plot_metrics_comparison_table(results["metrics_summary"], save_path=str(run_dir / "metrics_summary_table.pdf"))

    return summary


# Command-line entry point that loops over requested application targets, dimensions, layers, and variants.
def main() -> int:
    """CLI entry point."""
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_application_datasets(args)
    rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        for posterior_D in args.posterior_dimensions:
            for layer in args.layers:
                for variant in args.variants:
                    partial_summary = {
                        "application": dataset.application,
                        "target": dataset.target,
                        "kernel_type": dataset.kernel_type,
                        "posterior_D": posterior_D,
                        "D": posterior_D,
                        "layer": layer,
                        "variant": variant,
                        "mv_sampler": args.mv_sampler,
                        "rstiefel_rscol": args.rstiefel_rscol,
                        "output_dir": str(
                            args.output_dir / run_name_for(dataset.application, dataset.target, posterior_D, layer, variant)
                        ),
                        "computation_times": [],
                        "metrics_summary": {},
                    }
                    try:
                        summary = run_one(
                            args,
                            dataset=dataset,
                            posterior_D=posterior_D,
                            layer=layer,
                            variant=variant,
                        )
                        rows.append(aggregate_row(summary, status="ok"))
                    except Exception as exc:
                        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                        rows.append(aggregate_row(partial_summary, status="failed", error=error))
                        error_dir = Path(partial_summary["output_dir"])
                        write_json(error_dir / "error.json", {"error": error, "traceback": traceback.format_exc()})
                        print(
                            (
                                f"\nFailed application={dataset.application}, target={dataset.target}, "
                                f"posterior_D={posterior_D}, layer={layer}, variant={variant}: {error}"
                            ),
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            write_application_csv(args.output_dir / "application_summary.csv", rows)
                            raise

    write_application_csv(args.output_dir / "application_summary.csv", rows)
    print(f"\nApplication summary written to: {args.output_dir / 'application_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
