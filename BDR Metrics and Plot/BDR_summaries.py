"""
Summary-table helpers for BDR runs.

These helpers write two run-level reports:
- posterior_parameter_summary.csv/.pdf
- time_complexity_summary.csv/.pdf
"""

from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover - scipy is expected, but keep reports robust.
    ks_2samp = None


PARAMETER_SUMMARY_COLUMNS = [
    "parameter",
    "component",
    "chain",
    "mean",
    "median",
    "sd",
    "2.5%",
    "97.5%",
    "ESS",
    "Rhat",
    "stest",
    "pvalue",
]

COMPLEXITY_COLUMNS = [
    "scope",
    "parameter",
    "sampling_step",
    "per_sample_complexity",
    "per_iteration_complexity",
    "overall_time_complexity",
    "notes",
]

METRIC_COMPARISON_COLUMNS = [
    "model",
    "data_case",
    "data_dim",
    "application",
    "target",
    "posterior_D",
    "layer",
    "variant",
    "RMSPE",
    "NSME",
    "CRPS",
    "BIC",
    "MLPPD",
    "CP",
    "ALCI",
    "Score",
]

PARAMETER_ORDER = [
    "tau2_y",
    "tau2",
    "tau2_q",
    "tau2_r",
    "g_y",
    "g",
    "g_q",
    "g_r",
    "theta_D_y",
    "theta_D",
    "theta_y",
    "theta_q",
    "theta_r",
    "W",
    "WWT",
    "Lambda",
    "M",
    "V",
    "Q",
    "R",
]

METRIC_MEAN_FIELDS = [
    ("RMSPE", "rmspe_mean"),
    ("NSME", "nsme_mean"),
    ("CRPS", "crps_mean"),
    ("BIC", "bic_mean"),
    ("MLPPD", "mlppd_mean"),
    ("CP", "cp_mean"),
    ("ALCI", "alci_mean"),
    ("Score", "score_mean"),
]

METRIC_BOXPLOT_COLORS = {
    "RMSPE": "skyblue",
    "NSME": "lightgreen",
    "CRPS": "salmon",
    "BIC": "plum",
    "MLPPD": "orange",
    "CP": "lightgray",
    "ALCI": "khaki",
    "Score": "lightcoral",
}

SYMLOG_METRICS = {"ALCI", "Score", "BIC", "MLPPD"}


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _padded_axis_limits(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    finite = _finite_values(np.asarray(values, dtype=float))
    if finite.size == 0:
        return None

    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    if span <= 0:
        scale = max(abs(lower), abs(upper))
        padding = scale * 0.1 if scale > 0 else 0.1
    else:
        padding = span * 0.08

    return lower - padding, upper + padding


def _format_float(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{float(value):.8g}"
    return value


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _to_numeric_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim < 1 or arr.shape[0] == 0:
        return None
    if not np.any(np.isfinite(arr)):
        return None
    return arr


def _component_names(parameter: str, tail_shape: Tuple[int, ...]) -> List[str]:
    if not tail_shape:
        return [parameter]

    names: List[str] = []
    for index in np.ndindex(tail_shape):
        one_based = ",".join(str(i + 1) for i in index)
        names.append(f"{parameter}[{one_based}]")
    return names


def _extract_parameter_matrix(chain: Mapping[str, Any], parameter: str) -> Tuple[np.ndarray, List[str]]:
    if parameter == "WWT":
        if "W" not in chain:
            raise KeyError("WWT requires W samples.")
        W = _to_numeric_array(chain["W"])
        if W is None or W.ndim != 3:
            raise ValueError("WWT requires W samples with shape (draws, p, D).")
        projections = np.asarray([draw @ draw.T for draw in W], dtype=float)
        n_draws = projections.shape[0]
        matrix = projections.reshape(n_draws, -1)
        return matrix, _component_names("WWT", projections.shape[1:])

    if parameter not in chain:
        raise KeyError(f"Parameter '{parameter}' not found.")

    arr = _to_numeric_array(chain[parameter])
    if arr is None:
        raise ValueError(f"Parameter '{parameter}' is not a numeric sample array.")

    n_draws = arr.shape[0]
    matrix = arr.reshape(n_draws, -1)
    return matrix, _component_names(parameter, tuple(arr.shape[1:]))


def _available_parameters(chains: Sequence[Mapping[str, Any]]) -> List[str]:
    if not chains:
        return []

    first_chain = chains[0]
    keys = []
    for key, value in first_chain.items():
        if all(key in chain for chain in chains) and _to_numeric_array(value) is not None:
            keys.append(key)

    if "W" in keys:
        keys.append("WWT")

    ordered = [key for key in PARAMETER_ORDER if key in keys]
    ordered.extend(sorted(key for key in keys if key not in ordered))
    return ordered


def _effective_sample_size(values: np.ndarray) -> float:
    x = _finite_values(values)
    n = x.size
    if n <= 1:
        return float(n)

    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float(n)

    positive_rhos = []
    for lag in range(1, n):
        rho = float(np.dot(x[:-lag], x[lag:]) / denom)
        if not np.isfinite(rho) or rho <= 0:
            break
        positive_rhos.append(rho)

    tau = 1.0 + 2.0 * float(np.sum(positive_rhos))
    if tau <= 0:
        return float(n)
    return float(min(n, max(1.0, n / tau)))


def _rhat_by_component(component_matrices: Sequence[np.ndarray]) -> np.ndarray:
    if len(component_matrices) < 2:
        return np.full(component_matrices[0].shape[1], np.nan)

    n_components = min(matrix.shape[1] for matrix in component_matrices)
    rhat = np.full(n_components, np.nan, dtype=float)

    for component_idx in range(n_components):
        component_chains = [_finite_values(matrix[:, component_idx]) for matrix in component_matrices]
        n = min(chain.size for chain in component_chains)
        if n < 2:
            continue

        truncated = np.vstack([chain[:n] for chain in component_chains])
        within_vars = np.var(truncated, axis=1, ddof=1)
        W_within = float(np.mean(within_vars))
        if W_within <= 0 or not np.isfinite(W_within):
            continue

        chain_means = np.mean(truncated, axis=1)
        B_between = n * float(np.var(chain_means, ddof=1))
        var_hat = ((n - 1) / n) * W_within + (B_between / n)
        if var_hat >= 0:
            rhat[component_idx] = math.sqrt(var_hat / W_within)

    return rhat


def _stationarity_test(values: np.ndarray, alpha: float = 0.05) -> Tuple[str, float]:
    x = _finite_values(values)
    if x.size < 8:
        return "", np.nan
    if np.allclose(x, x[0]):
        return "passed", 1.0
    if ks_2samp is None:
        return "", np.nan

    half = x.size // 2
    if half < 4 or x.size - half < 4:
        return "", np.nan

    result = ks_2samp(x[:half], x[half:])
    pvalue = float(result.pvalue)
    return ("passed" if pvalue > alpha else "failed"), pvalue


def _coda_lookup(results: Mapping[str, Any]) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Dict[Tuple[str, str], float]]:
    diagnostics = results.get("parameter_diagnostics")
    heidel_lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    rhat_lookup: Dict[Tuple[str, str], float] = {}
    if not isinstance(diagnostics, Mapping):
        return heidel_lookup, rhat_lookup

    for parameter, diagnostic in diagnostics.items():
        if not isinstance(diagnostic, Mapping):
            continue

        heidel = diagnostic.get("heidel")
        if hasattr(heidel, "to_dict"):
            for row in heidel.to_dict("records"):
                component = str(row.get("parameter", parameter))
                chain = int(row.get("chain", 0))
                heidel_lookup[(str(parameter), component, chain)] = row

        rhat = diagnostic.get("rhat")
        if hasattr(rhat, "to_dict"):
            for row in rhat.to_dict("records"):
                component = str(row.get("parameter", parameter))
                value = row.get("rhat")
                try:
                    rhat_lookup[(str(parameter), component)] = float(value)
                except (TypeError, ValueError):
                    continue

    return heidel_lookup, rhat_lookup


def _complexity_entry(
    parameter: str,
    *,
    n_chains: int,
    n_iterations: int,
) -> Dict[str, str]:
    """Return complexity text for a sampled parameter."""
    if parameter in {"tau2_y", "tau2", "tau2_q", "tau2_r"}:
        per_sample = "O(n^3 + n*p*D); latent-layer updates omit n*p*D"
        return {
            "sampling_step": "inverse-gamma conditional",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*(n^3 + n*p*D))",
            "notes": "One GP covariance factorization/solve dominates each sampled noise variance.",
        }

    if parameter in {"g_y", "g", "g_q", "g_r"}:
        per_sample = "O(2*n^3 + n*p*D); latent-layer updates omit n*p*D"
        return {
            "sampling_step": "Metropolis-Hastings update",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*(2*n^3 + n*p*D))",
            "notes": "Current and proposed GP likelihood evaluations dominate.",
        }

    if parameter in {"theta_D_y", "theta_D", "theta_y", "theta_q", "theta_r"}:
        per_sample = "O(2*n^3 + n*p*D); separable D-dimensional updates can be O(2*D*n^3)"
        return {
            "sampling_step": "Metropolis-Hastings update",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*(2*n^3 + n*p*D))",
            "notes": "Current and proposed GP likelihood evaluations dominate.",
        }

    if parameter == "W":
        per_sample = "O(A_hmc*((2*T_hmc+2)*(L*n^3 + n^2*p*D) + T_hmc*(p^2*D + p*D^2)))"
        return {
            "sampling_step": "HMC on the Stiefel manifold",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*A_hmc*((2*T_hmc+2)*(L*n^3 + n^2*p*D) + T_hmc*(p^2*D + p*D^2)))",
            "notes": "A_hmc is the proposal-attempt count needed for accepted W samples.",
        }

    if parameter == "WWT":
        return {
            "sampling_step": "derived projection from sampled W",
            "per_sample_complexity": "O(p^2*D) per saved W draw",
            "per_iteration_complexity": "not part of the MCMC update",
            "overall_time_complexity": "O(total_saved_W_draws*p^2*D)",
            "notes": "Derived after sampling; not an MCMC sampling step.",
        }

    if parameter == "Lambda":
        per_sample = "O(E_lambda*(p*D^2 + D^3 + D^2))"
        return {
            "sampling_step": "slice sampling",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*E_lambda*(p*D^2 + D^3 + D^2))",
            "notes": "E_lambda is the slice/angle trial count, capped by max_iter_lambda.",
        }

    if parameter == "M":
        per_sample = "O(p*D^2) plus backend/block-update overhead"
        return {
            "sampling_step": "matrix-von-Mises-Fisher Gibbs update",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*p*D^2)",
            "notes": "",
        }

    if parameter == "V":
        per_sample = "O(D^3)"
        return {
            "sampling_step": "matrix-von-Mises-Fisher Gibbs update",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*D^3)",
            "notes": "",
        }

    if parameter == "Q":
        per_sample = "D=1: O((E_Q+1)*n^3); D>1: O(D*(E_Q+1)*(n^3 + n^2*D))"
        return {
            "sampling_step": "elliptical slice sampling",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*D*(E_Q+1)*(n^3 + n^2*D))",
            "notes": "E_Q is the accepted slice trial count for latent Q.",
        }

    if parameter == "R":
        per_sample = "D=1: O((E_R+1)*n^3); D>1: O(D*(E_R+1)*(n^3 + n^2*D))"
        return {
            "sampling_step": "elliptical slice sampling",
            "per_sample_complexity": per_sample,
            "per_iteration_complexity": per_sample,
            "overall_time_complexity": f"O({n_chains}*{n_iterations}*D*(E_R+1)*(n^3 + n^2*D))",
            "notes": "E_R is the accepted slice trial count for latent R.",
        }

    return {
        "sampling_step": "saved numeric sample array",
        "per_sample_complexity": "not classified",
        "per_iteration_complexity": "not classified",
        "overall_time_complexity": "not classified",
        "notes": "No sampler-specific complexity rule is registered for this key.",
    }


def _parameter_complexities(
    parameters: Sequence[str],
    *,
    n_chains: int,
    n_iterations: int,
) -> Dict[str, Dict[str, str]]:
    return {
        parameter: _complexity_entry(
            parameter,
            n_chains=n_chains,
            n_iterations=n_iterations,
        )
        for parameter in parameters
    }


def build_posterior_parameter_summary(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build per-chain posterior summaries for every sampled numeric parameter."""
    chains = results.get("chains_samples", [])
    if not isinstance(chains, Sequence) or not chains:
        return []

    parameters = _available_parameters(chains)
    heidel_lookup, coda_rhat_lookup = _coda_lookup(results)
    rows: List[Dict[str, Any]] = []

    for parameter in parameters:
        try:
            extracted = [_extract_parameter_matrix(chain, parameter) for chain in chains]
        except (KeyError, ValueError):
            continue

        matrices = [matrix for matrix, _ in extracted]
        names = extracted[0][1]
        n_components = min(matrix.shape[1] for matrix in matrices)
        if n_components == 0:
            continue

        rhat_values = _rhat_by_component(matrices)

        for chain_idx, matrix in enumerate(matrices, start=1):
            for component_idx in range(n_components):
                component = names[component_idx]
                values = _finite_values(matrix[:, component_idx])
                if values.size == 0:
                    continue

                coda_row = heidel_lookup.get((parameter, component, chain_idx), {})
                coda_rhat = coda_rhat_lookup.get((parameter, component))
                stest, pvalue = _stationarity_test(values)
                ess = _effective_sample_size(values)

                if coda_row:
                    ess = coda_row.get("ess", ess)
                    stest = coda_row.get("stest", stest)
                    pvalue = coda_row.get("pvalue", pvalue)

                rows.append({
                    "parameter": parameter,
                    "component": component,
                    "chain": chain_idx,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "2.5%": float(np.quantile(values, 0.025)),
                    "97.5%": float(np.quantile(values, 0.975)),
                    "ESS": _safe_float(ess),
                    "Rhat": _safe_float(coda_rhat) if coda_rhat is not None else (
                        float(rhat_values[component_idx])
                        if component_idx < rhat_values.size and np.isfinite(rhat_values[component_idx])
                        else np.nan
                    ),
                    "stest": stest,
                    "pvalue": _safe_float(pvalue),
                })

    return rows


def _complexity_rows_for_parameter_summary(
    complexity_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Represent complexity notes as rows in the posterior summary table."""
    rows: List[Dict[str, Any]] = []
    for row in complexity_rows:
        scope = row.get("scope")
        if scope == "overall":
            parameter = "overall_time_complexity"
        else:
            parameter_name = str(row.get("parameter", "")).replace(" ", "_")
            parameter = f"time_complexity_{parameter_name}"

        component_parts = [
            str(row.get("sampling_step", "")),
            f"per_sample={row.get('per_sample_complexity', '')}",
            f"per_iteration={row.get('per_iteration_complexity', '')}",
            f"overall={row.get('overall_time_complexity', '')}",
        ]
        notes = row.get("notes")
        if notes:
            component_parts.append(f"notes={notes}")

        summary_row = {column: "" for column in PARAMETER_SUMMARY_COLUMNS}
        summary_row.update({
            "parameter": parameter,
            "component": " | ".join(part for part in component_parts if part),
            "chain": "all",
        })
        rows.append(summary_row)

    return rows


def build_time_complexity_summary(
    results: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    n_train: int,
    p: int,
    layer: int,
    variant: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a compact theoretical complexity report for selected parameters."""
    chains = results.get("chains_samples", [])
    parameters = _available_parameters(chains) if isinstance(chains, Sequence) and chains else []

    D = int(config.get("D", 1))
    n_chains = int(config.get("n_chains", len(chains) if chains else 1))
    n_iterations = int(config.get("n_iterations", 1))
    model_variant = variant or config.get("variant", "full") or "full"
    total_seconds = float(np.sum(results.get("computation_times", [])))
    complexity_by_parameter = _parameter_complexities(
        parameters,
        n_chains=n_chains,
        n_iterations=n_iterations,
    )

    run_scale = f"O({n_chains} chains * {n_iterations} iterations * per-iteration cost)"
    context = (
        f"selected_parameters={len(parameters)}, variant={model_variant}, "
        f"n={n_train}, p={p}, D={D}, L={layer}, measured_total_seconds={total_seconds:.4g}"
    )

    rows: List[Dict[str, Any]] = [{
        "scope": "overall",
        "parameter": "all active sampled parameters",
        "sampling_step": "complete MCMC sweep",
        "per_sample_complexity": "",
        "per_iteration_complexity": "sum of active rows below",
        "overall_time_complexity": run_scale,
        "notes": context,
    }]

    for parameter in parameters:
        complexity = complexity_by_parameter[parameter]
        rows.append({
            "scope": "parameter",
            "parameter": parameter,
            "sampling_step": complexity["sampling_step"],
            "per_sample_complexity": complexity["per_sample_complexity"],
            "per_iteration_complexity": complexity["per_iteration_complexity"],
            "overall_time_complexity": complexity["overall_time_complexity"],
            "notes": complexity["notes"],
        })

    return rows


def write_csv_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_float(row.get(column, "")) for column in columns})


def _pdf_cell(value: Any, width: int = 18) -> str:
    value = _format_float(value)
    text = "" if value is None else str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def write_pdf_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    path: Path,
    *,
    title: str,
    rows_per_page: int = 28,
    wrap_width: int = 18,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = [rows[i:i + rows_per_page] for i in range(0, len(rows), rows_per_page)] or [[]]

    with PdfPages(path) as pdf:
        for page_idx, page_rows in enumerate(pages, start=1):
            fig, ax = plt.subplots(figsize=(17, 11))
            ax.axis("off")
            page_title = f"{title} ({page_idx}/{len(pages)})"
            ax.set_title(page_title, fontsize=13, fontweight="bold", pad=16)

            table_data = [
                [_pdf_cell(row.get(column, ""), width=wrap_width) for column in columns]
                for row in page_rows
            ]
            if not table_data:
                table_data = [["" for _ in columns]]

            table = ax.table(
                cellText=table_data,
                colLabels=list(columns),
                cellLoc="center",
                colLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(5.5 if len(columns) > 10 else 7)
            table.scale(1.0, 1.6 if len(columns) > 10 else 2.0)

            for (row_idx, _), cell in table.get_celld().items():
                if row_idx == 0:
                    cell.set_facecolor("#4CAF50")
                    cell.set_text_props(weight="bold", color="white")
                elif row_idx % 2 == 0:
                    cell.set_facecolor("#f0f0f0")

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _as_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def model_label_for_run(row: Mapping[str, Any]) -> str:
    """Return the display label used in metrics comparison tables."""
    posterior_D = _as_int_or_none(row.get("posterior_D", row.get("D")))
    layer = _as_int_or_none(row.get("layer"))
    variant = str(row.get("variant", "full") or "full")
    variant_key = variant.lower()

    if variant_key == "full" and posterior_D is not None:
        if layer == 1:
            return f"GP ({posterior_D}) BDR"
        if layer == 2:
            return f"DGP 2-layer ({posterior_D}) BDR"
        if layer == 3:
            return f"DGP 3-layer ({posterior_D}) BDR"

    if variant_key == "w_known" and posterior_D is not None:
        if layer == 1:
            return f"GP ({posterior_D}) Oracle"
        if layer == 2:
            return f"DGP 2-layer ({posterior_D}) Oracle"
        if layer == 3:
            return f"DGP 3-layer ({posterior_D}) Oracle"

    if variant_key == "no_w_selective" and posterior_D is not None:
        if layer == 1:
            return f"GP ({posterior_D}) W/o"
        if layer == 2:
            return f"DGP 2-layer ({posterior_D}) W/o"
        if layer == 3:
            return f"DGP 3-layer ({posterior_D}) W/o"

    dimension_part = f"({posterior_D})" if posterior_D is not None else ""
    if variant_key == "full":
        return f"Layer {layer} GP {dimension_part} BDR" if layer is not None else f"GP {dimension_part} BDR"
    return f"{variant} L{layer} D{posterior_D}"


def compact_model_label_for_run(row: Mapping[str, Any]) -> str:
    """Return compact labels used on aggregate metric boxplot x-axes."""
    posterior_D = _as_int_or_none(row.get("posterior_D", row.get("D")))
    variant = str(row.get("variant", "full") or "full")
    variant_key = variant.lower()
    dimension_part = str(posterior_D) if posterior_D is not None else "D"

    if variant_key == "full":
        return f"{dimension_part}-BDR"
    if variant_key == "w_known":
        return f"{dimension_part}-Oracle"
    if variant_key == "no_w_selective":
        return f"{dimension_part}-W/o"
    if variant_key == "no_w":
        return "No_W"
    return f"{dimension_part}-{variant}"


def build_metrics_comparison_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build aggregate metric comparison rows from simulation/application CSV rows."""
    comparison_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("status") not in (None, "", "ok"):
            continue

        comparison_row: Dict[str, Any] = {
            "model": model_label_for_run(row),
            "data_case": row.get("data_case", ""),
            "data_dim": row.get("data_dim", ""),
            "application": row.get("application", ""),
            "target": row.get("target", ""),
            "posterior_D": row.get("posterior_D", row.get("D", "")),
            "layer": row.get("layer", ""),
            "variant": row.get("variant", ""),
        }
        for label, field in METRIC_MEAN_FIELDS:
            comparison_row[label] = row.get(field, "")
        comparison_rows.append(comparison_row)

    return comparison_rows


def write_layer_metric_boxplots(
    rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    write_plots: bool = True,
) -> Dict[str, str]:
    """Write aggregate per-layer model boxplots for each metric."""
    if not write_plots:
        return {}

    from BDR_plot import (  # type: ignore[import]
        plot_grouped_boxplot_by_dimension,
        plot_single_layer_by_dimension,
    )

    ok_rows = [row for row in rows if row.get("status") in (None, "", "ok")]
    output_dir = Path(output_dir)
    plot_dir = output_dir / "metric_boxplots_by_layer"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    for metric_label, field in METRIC_MEAN_FIELDS:
        score_map: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
        metric_values: List[float] = []

        for row in ok_rows:
            layer = _as_int_or_none(row.get("layer"))
            if layer is None:
                continue
            value = _safe_float(row.get(field))
            if not np.isfinite(value):
                continue

            model_label = compact_model_label_for_run(row)
            score_map.setdefault(model_label, {}).setdefault("all", {}).setdefault(layer, []).append(value)
            metric_values.append(value)

        layers = sorted(
            {
                layer
                for sample_scores in score_map.values()
                for layer_scores in sample_scores.values()
                for layer in layer_scores
            }
        )
        y_limits = _padded_axis_limits(metric_values)
        yscale = "symlog" if metric_label in SYMLOG_METRICS else "linear"
        layer_names = {layer: f"Layer {layer}" for layer in layers}

        if layers:
            grouped_filename = f"{metric_label.lower()}_layers_grouped_model_boxplot.png"
            grouped_save_path = plot_dir / grouped_filename
            plot_grouped_boxplot_by_dimension(
                score_map,
                sample_size="all",
                save_path=str(grouped_save_path),
                model_names=layer_names,
                xlabel="Model",
                ylabel=metric_label,
                ylim=y_limits,
                yscale=yscale,
            )
            paths[f"{metric_label.lower()}_layers_grouped_boxplot_png"] = str(grouped_save_path)
            paths[f"{metric_label.lower()}_layers_grouped_boxplot_pdf"] = str(grouped_save_path.with_suffix(".pdf"))

            if metric_label == "MLPPD":
                grouped_linear_filename = f"{metric_label.lower()}_layers_grouped_model_boxplot_linear.png"
                grouped_linear_save_path = plot_dir / grouped_linear_filename
                plot_grouped_boxplot_by_dimension(
                    score_map,
                    sample_size="all",
                    save_path=str(grouped_linear_save_path),
                    model_names=layer_names,
                    xlabel="Model",
                    ylabel=metric_label,
                    ylim=y_limits,
                    yscale="linear",
                )
                paths[f"{metric_label.lower()}_layers_grouped_linear_boxplot_png"] = str(grouped_linear_save_path)
                paths[f"{metric_label.lower()}_layers_grouped_linear_boxplot_pdf"] = str(
                    grouped_linear_save_path.with_suffix(".pdf")
                )

        for layer in layers:
            filename = f"{metric_label.lower()}_layer{layer}_model_boxplot.png"
            save_path = plot_dir / filename
            plot_single_layer_by_dimension(
                score_map,
                sample_size="all",
                layer=layer,
                save_path=str(save_path),
                xlabel="Model",
                ylabel=metric_label,
                facecolor=METRIC_BOXPLOT_COLORS.get(metric_label, "lightblue"),
                ylim=y_limits,
                yscale=yscale,
            )
            paths[f"{metric_label.lower()}_layer{layer}_boxplot_png"] = str(save_path)
            paths[f"{metric_label.lower()}_layer{layer}_boxplot_pdf"] = str(save_path.with_suffix(".pdf"))

            if metric_label == "MLPPD":
                linear_filename = f"{metric_label.lower()}_layer{layer}_model_boxplot_linear.png"
                linear_save_path = plot_dir / linear_filename
                plot_single_layer_by_dimension(
                    score_map,
                    sample_size="all",
                    layer=layer,
                    save_path=str(linear_save_path),
                    xlabel="Model",
                    ylabel=metric_label,
                    facecolor=METRIC_BOXPLOT_COLORS.get(metric_label, "lightblue"),
                    ylim=y_limits,
                    yscale="linear",
                )
                paths[f"{metric_label.lower()}_layer{layer}_linear_boxplot_png"] = str(linear_save_path)
                paths[f"{metric_label.lower()}_layer{layer}_linear_boxplot_pdf"] = str(
                    linear_save_path.with_suffix(".pdf")
                )

    return paths


def write_metrics_comparison_tables(
    rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    write_pdf: bool = True,
) -> Dict[str, str]:
    """Write top-level metric comparison CSV/PDF using publication labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = build_metrics_comparison_rows(rows)

    paths: Dict[str, str] = {}
    csv_path = output_dir / "metrics_comparison_table.csv"
    write_csv_table(comparison_rows, METRIC_COMPARISON_COLUMNS, csv_path)
    paths["metrics_comparison_table_csv"] = str(csv_path)

    if write_pdf:
        pdf_columns = ["model", "RMSPE", "NSME", "CRPS", "BIC", "MLPPD", "CP", "ALCI", "Score"]
        pdf_path = output_dir / "metrics_comparison_table.pdf"
        write_pdf_table(
            comparison_rows,
            pdf_columns,
            pdf_path,
            title="Metrics Comparison Table",
            rows_per_page=24,
            wrap_width=18,
        )
        paths["metrics_comparison_table_pdf"] = str(pdf_path)

    return paths


def write_run_summary_tables(
    results: Mapping[str, Any],
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    n_train: int,
    p: int,
    layer: int,
    variant: Optional[str] = None,
    write_pdf: bool = True,
) -> Dict[str, str]:
    """Write posterior parameter and complexity summary tables for a run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    complexity_rows = build_time_complexity_summary(
        results,
        config,
        n_train=n_train,
        p=p,
        layer=layer,
        variant=variant,
    )

    parameter_rows = build_posterior_parameter_summary(results)
    parameter_rows.extend(_complexity_rows_for_parameter_summary(complexity_rows))
    parameter_csv = output_dir / "posterior_parameter_summary.csv"
    write_csv_table(parameter_rows, PARAMETER_SUMMARY_COLUMNS, parameter_csv)
    paths["posterior_parameter_summary_csv"] = str(parameter_csv)
    if write_pdf:
        parameter_pdf = output_dir / "posterior_parameter_summary.pdf"
        write_pdf_table(
            parameter_rows,
            PARAMETER_SUMMARY_COLUMNS,
            parameter_pdf,
            title="Posterior Parameter Summary",
            rows_per_page=30,
            wrap_width=14,
        )
        paths["posterior_parameter_summary_pdf"] = str(parameter_pdf)

    complexity_csv = output_dir / "time_complexity_summary.csv"
    write_csv_table(complexity_rows, COMPLEXITY_COLUMNS, complexity_csv)
    paths["time_complexity_summary_csv"] = str(complexity_csv)
    if write_pdf:
        complexity_pdf = output_dir / "time_complexity_summary.pdf"
        write_pdf_table(
            complexity_rows,
            COMPLEXITY_COLUMNS,
            complexity_pdf,
            title="Time Complexity Summary",
            rows_per_page=8,
            wrap_width=34,
        )
        paths["time_complexity_summary_pdf"] = str(complexity_pdf)

    return paths
