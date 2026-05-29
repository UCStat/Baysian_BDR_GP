"""
BDR Plot - Visualization Functions for Bayesian Dimensionality Reduction

This module contains all plotting functions for diagnostics and visualization:
    - Trace plots (multiple chains)
    - Density plots (with mean/median lines)
    - Autocorrelation plots
    - Histogram plots
    - W matrix trace plots
    - Predicted vs Actual plots
    - Convergence diagnostic plots
    - Single and grouped layer boxplots by dimension/method
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from scipy.stats import gaussian_kde
from typing import Any, List, Mapping, Optional, Sequence, Tuple


def _dimension_sort_key(dimension: Any):
    """Sort numeric dimension labels numerically, then non-numeric labels alphabetically."""
    try:
        return (0, int(dimension))
    except (TypeError, ValueError):
        dimension_text = str(dimension)
        if "-" in dimension_text:
            prefix, suffix = dimension_text.split("-", 1)
            try:
                prefix_number = int(prefix)
            except ValueError:
                pass
            else:
                suffix_order = {"BDR": 0, "W/o": 1, "Oracle": 2, "No_W": 3}
                return (1, suffix_order.get(suffix, 99), prefix_number, suffix)
        return (2, dimension_text)


def _mapping_get(mapping: Mapping, key: Any):
    """Return a dictionary value using exact, string, or integer key variants."""
    if key in mapping:
        return mapping[key]

    key_str = str(key)
    if key_str in mapping:
        return mapping[key_str]

    try:
        key_int = int(key)
    except (TypeError, ValueError):
        return None

    return mapping.get(key_int)


def _save_plot(save_path: Optional[str], dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """Save the current figure and add a PDF copy for non-PDF outputs."""
    if not save_path:
        return

    path = Path(save_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if path.suffix.lower() != ".pdf":
        plt.savefig(path.with_suffix(".pdf"), bbox_inches=bbox_inches)


def _finite_plot_values(values: Sequence[Any]) -> np.ndarray:
    finite_values = []
    for value in values:
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        array = array[np.isfinite(array)]
        if array.size:
            finite_values.append(array)

    if not finite_values:
        return np.asarray([], dtype=float)
    return np.concatenate(finite_values)


def _apply_small_value_yaxis_scale(
    ax,
    values: Sequence[Any],
    *,
    ylim: Optional[Tuple[float, float]] = None,
    threshold: float = 1e-2,
    fontsize: int = 14,
) -> None:
    """Use a math-text scientific y-axis multiplier for very small values."""
    scale_values = list(values)
    if ylim is not None:
        scale_values.append(ylim)

    finite = _finite_plot_values(scale_values)
    if finite.size == 0:
        return

    max_abs = float(np.max(np.abs(finite)))
    if max_abs == 0 or max_abs >= threshold:
        return

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_offset_position("left")
    ax.yaxis.get_offset_text().set_fontsize(fontsize)


def _infer_symlog_linthresh(values: Sequence[Any]) -> float:
    finite = np.abs(_finite_plot_values(values))
    finite = finite[finite > 0]
    if finite.size == 0:
        return 1.0

    min_abs = float(np.min(finite))
    max_abs = float(np.max(finite))
    return max(min_abs / 10.0, max_abs * 1e-6, np.finfo(float).tiny)


def _apply_yaxis_scale(
    ax,
    values: Sequence[Any],
    *,
    yscale: str = "linear",
    symlog_linthresh: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
    fontsize: int = 14,
) -> None:
    scale_values = list(values)
    if ylim is not None:
        scale_values.append(ylim)

    if (yscale or "linear").lower() == "symlog":
        linthresh = symlog_linthresh
        if linthresh is None or not np.isfinite(linthresh) or linthresh <= 0:
            linthresh = _infer_symlog_linthresh(scale_values)
        ax.set_yscale("symlog", linthresh=linthresh)
        return

    _apply_small_value_yaxis_scale(ax, values, ylim=ylim, fontsize=fontsize)


def plot_trace(chains: List[np.ndarray], param_name: str, save_path: Optional[str] = None):
    """
    Trace plot for multiple chains (handles scalars and vectors).
    
    Args:
        chains: List of chains, each (n_samples,) or (n_samples, D)
        param_name: Parameter name for title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['black', 'blue', 'red', 'green', 'purple']
    
    # Check if parameter is vector
    if chains[0].ndim > 1:
        D = chains[0].shape[1]
        for d in range(D):
            for i, chain in enumerate(chains):
                color = colors[i % len(colors)]
                ax.plot(chain[:, d], color=color, alpha=0.5, linewidth=0.8,
                       label=f'Chain {i+1}, dim {d+1}' if i == 0 else '')
    else:
        for i, chain in enumerate(chains):
            color = colors[i % len(colors)]
            ax.plot(chain, color=color, alpha=0.7, linewidth=0.8, label=f'Chain {i+1}')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel(f'{param_name}', fontsize=14)
    ax.set_title(f'Trace Plot: {param_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_density(chains: List[np.ndarray], param_name: str, save_path: Optional[str] = None):
    """
    Density plot with vertical lines for mean and median (handles vectors).
    
    Args:
        chains: List of chains
        param_name: Parameter name
        save_path: Path to save figure
    """
    # Handle vector parameters
    if chains[0].ndim > 1:
        D = chains[0].shape[1]
        fig, axes = plt.subplots(1, D, figsize=(8*D, 6))
        if D == 1:
            axes = [axes]
        
        for d in range(D):
            ax = axes[d]
            all_samples = np.concatenate([c[:, d] for c in chains])
            
            # Histogram
            ax.hist(all_samples, bins=50, density=True, alpha=0.6, 
                   color='skyblue', edgecolor='black', label='Histogram')
            
            # KDE
            try:
                kde = gaussian_kde(all_samples)
                x_range = np.linspace(all_samples.min(), all_samples.max(), 200)
                ax.plot(x_range, kde(x_range), color='blue', linewidth=2, label='Density')
            except:
                pass
            
            # Mean and median
            mean_val = np.mean(all_samples)
            median_val = np.median(all_samples)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean = {mean_val:.4f}')
            ax.axvline(median_val, color='green', linestyle='-', linewidth=2, 
                      label=f'Median = {median_val:.4f}')
            
            ax.set_xlabel(f'{param_name}[{d+1}]', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.set_title(f'{param_name}[{d+1}]', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Posterior Density: {param_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        _save_plot(save_path)
        plt.close()
    else:
        # Scalar parameter
        fig, ax = plt.subplots(figsize=(10, 6))
        all_samples = np.concatenate(chains)
        
        # Histogram
        ax.hist(all_samples, bins=50, density=True, alpha=0.6, 
               color='skyblue', edgecolor='black', label='Histogram')
        
        # KDE
        try:
            kde = gaussian_kde(all_samples)
            x_range = np.linspace(all_samples.min(), all_samples.max(), 200)
            ax.plot(x_range, kde(x_range), color='blue', linewidth=2, label='Density')
        except:
            pass
        
        # Mean and median
        mean_val = np.mean(all_samples)
        median_val = np.median(all_samples)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean = {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=2, 
                  label=f'Median = {median_val:.4f}')
        
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'Posterior Density: {param_name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        _save_plot(save_path)
        plt.close()


def plot_histogram(chains: List[np.ndarray], param_name: str, save_path: Optional[str] = None):
    """
    Histogram plot for parameter (handles vectors).
    
    Args:
        chains: List of chains
        param_name: Parameter name
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chains[0].ndim > 1:
        # For vectors, plot all dimensions
        D = chains[0].shape[1]
        all_samples = np.concatenate(chains, axis=0)
        
        for d in range(D):
            ax.hist(all_samples[:, d], bins=30, alpha=0.5, 
                   label=f'Dimension {d+1}', edgecolor='black')
    else:
        all_samples = np.concatenate(chains)
        ax.hist(all_samples, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    ax.set_xlabel(param_name, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'Histogram: {param_name}', fontsize=16, fontweight='bold')
    if chains[0].ndim > 1:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_autocorrelation(chain: np.ndarray, param_name: str, max_lag: int = 50,
                        save_path: Optional[str] = None):
    """
    Autocorrelation plot (handles vectors).
    
    Args:
        chain: Single chain (n_samples,) or (n_samples, D)
        param_name: Parameter name
        max_lag: Maximum lag
        save_path: Save path
    """
    # Handle vectors
    if chain.ndim > 1:
        D = chain.shape[1]
        fig, axes = plt.subplots(1, D, figsize=(8*D, 6))
        if D == 1:
            axes = [axes]
        
        for d in range(D):
            ax = axes[d]
            acf_values = []
            for lag in range(min(max_lag + 1, len(chain))):
                if lag == 0:
                    acf_values.append(1.0)
                elif len(chain[:, d]) > lag:
                    acf = np.corrcoef(chain[:-lag, d], chain[lag:, d])[0, 1]
                    acf_values.append(acf)
            
            ax.bar(range(len(acf_values)), acf_values, color='steelblue', alpha=0.7)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax.axhline(1.96/np.sqrt(len(chain)), color='red', linestyle='--')
            ax.axhline(-1.96/np.sqrt(len(chain)), color='red', linestyle='--')
            
            ax.set_xlabel('Lag', fontsize=14)
            ax.set_ylabel('ACF', fontsize=14)
            ax.set_title(f'{param_name}[{d+1}]', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Autocorrelation: {param_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        _save_plot(save_path)
        plt.close()
    else:
        # Scalar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        acf_values = []
        for lag in range(min(max_lag + 1, len(chain))):
            if lag == 0:
                acf_values.append(1.0)
            elif len(chain) > lag:
                acf = np.corrcoef(chain[:-lag], chain[lag:])[0, 1]
                acf_values.append(acf)
        
        ax.bar(range(len(acf_values)), acf_values, color='steelblue', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(1.96/np.sqrt(len(chain)), color='red', linestyle='--', label='95% CI')
        ax.axhline(-1.96/np.sqrt(len(chain)), color='red', linestyle='--')
        
        ax.set_xlabel('Lag', fontsize=14)
        ax.set_ylabel('Autocorrelation', fontsize=14)
        ax.set_title(f'Autocorrelation: {param_name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        _save_plot(save_path)
        plt.close()


def plot_W_trace_multichain(chains_W: List[np.ndarray], save_path: Optional[str] = None):
    """
    Trace plot for W matrix (all elements, multiple chains).
    
    Args:
        chains_W: List of W chains, each (n_samples, p, D)
        save_path: Save path
    """
    p, D = chains_W[0].shape[1], chains_W[0].shape[2]
    n_plots = min(p * D, 12)  # Limit to 12 plots for readability
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    colors = ['black', 'blue', 'red', 'green', 'purple']
    
    idx = 0
    for i in range(p):
        for j in range(D):
            if idx >= n_plots:
                break
            
            ax = axes[idx]
            
            for chain_id, chain_W in enumerate(chains_W):
                W_ij = chain_W[:, i, j]
                ax.plot(W_ij, color=colors[chain_id % len(colors)], 
                       alpha=0.7, linewidth=0.8, label=f'Chain {chain_id+1}')
            
            ax.set_ylabel(f'W_{i+1}{j+1}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8, ncol=len(chains_W))
            
            idx += 1
        if idx >= n_plots:
            break
    
    axes[-1].set_xlabel('Iteration', fontsize=12)
    plt.suptitle(f'Trace Plots: W Matrix ({p}×{D}, showing first {n_plots})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_W_projection_trace_multichain(chains_W: List[np.ndarray], save_path: Optional[str] = None):
    """
    Trace plot for W W^T projection matrix entries across multiple chains.

    Args:
        chains_W: List of W chains, each (n_samples, p, D)
        save_path: Save path
    """
    p = chains_W[0].shape[1]
    n_plots = min(p * p, 12)  # Limit to 12 plots for readability

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2*n_plots))
    if n_plots == 1:
        axes = [axes]

    colors = ['black', 'blue', 'red', 'green', 'purple']
    chains_WWT = [np.einsum('nkd,nld->nkl', chain_W, chain_W) for chain_W in chains_W]

    idx = 0
    for i in range(p):
        for j in range(p):
            if idx >= n_plots:
                break

            ax = axes[idx]

            for chain_id, chain_WWT in enumerate(chains_WWT):
                WWT_ij = chain_WWT[:, i, j]
                ax.plot(WWT_ij, color=colors[chain_id % len(colors)],
                        alpha=0.7, linewidth=0.8, label=f'Chain {chain_id+1}')

            ax.set_ylabel(f'WWT_{i+1}{j+1}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8, ncol=len(chains_W))

            idx += 1
        if idx >= n_plots:
            break

    axes[-1].set_xlabel('Iteration', fontsize=12)
    plt.suptitle(f'Trace Plots: W W^T Matrix ({p}x{p}, showing first {n_plots})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


# def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
#                              ci_bounds: np.ndarray, save_path: Optional[str] = None):
#     """
#     Actual vs predicted scatter plot with confidence intervals.
#
#     Args:
#         y_true: True values (n,)
#         y_pred: Predicted values (n,)
#         ci_bounds: Confidence interval bounds (n, 2) - [lower, upper]
#         save_path: Path to save figure
#     """
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     # Error bars
#     errors = np.array([
#         y_pred - ci_bounds[:, 0],
#         ci_bounds[:, 1] - y_pred
#     ])
#
#     ax.errorbar(y_true, y_pred, yerr=errors, fmt='o', alpha=0.5,
#                capsize=3, markersize=6, ecolor='gray', markerfacecolor='blue')
#
#     # Identity line (perfect prediction)
#     min_val = min(y_true.min(), y_pred.min())
#     max_val = max(y_true.max(), y_pred.max())
#     ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
#
#     ax.set_xlabel('Actual', fontsize=14)
#     ax.set_ylabel('Predicted', fontsize=14)
#     ax.set_title('Predicted vs Actual with 95% CI', fontsize=16, fontweight='bold')
#     ax.legend(fontsize=12)
#     ax.grid(True, alpha=0.3)
#     ax.set_aspect('equal')
#
#     plt.tight_layout()
#     _save_plot(save_path)
#     plt.close()


def plot_actual_vs_predicted(
    actual_y,
    predicted_y,
    training_confidence_interval_bounds,
    title=None,
    xlabel="Actual",
    ylabel="Predicted",
    ax=None,
    save_path: Optional[str] = None,
):
    actual_y = np.asarray(actual_y).reshape(-1)
    predicted_y = np.asarray(predicted_y).reshape(-1)
    training_confidence_interval_bounds = np.asarray(training_confidence_interval_bounds)

    predicted_col = predicted_y.reshape(-1, 1)
    errors = np.concatenate(
        (
            predicted_col - training_confidence_interval_bounds[:, 0:1],
            training_confidence_interval_bounds[:, 1:2] - predicted_col,
        ),
        axis=1,
    ).T

    mmin = min(np.min(actual_y), np.min(predicted_y))
    mmax = max(np.max(actual_y), np.max(predicted_y))
    padding = (mmax - mmin) * 0.1
    bounds = [mmin - padding, mmax + padding]

    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.errorbar(actual_y, predicted_y, yerr=errors, fmt="o", alpha=0.2)

    ax.set_title("Predicted vs. Actual (Training)")
    ax.plot(bounds, bounds, "--")
    ax.set_aspect(1.0)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    _save_plot(save_path)
    if created_ax:
        plt.close()


def plot_convergence_diagnostics(r_hats: dict, hw_results: dict, save_path: Optional[str] = None):
    """
    Plot convergence diagnostics summary.
    
    Args:
        r_hats: Dictionary of R-hat values
        hw_results: Dictionary of Heidelberg-Welch results
        save_path: Save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R-hat plot
    params = list(r_hats.keys())
    r_hat_vals = list(r_hats.values())
    
    colors = ['green' if r < 1.1 else 'orange' if r < 1.2 else 'red' for r in r_hat_vals]
    ax1.bar(range(len(params)), r_hat_vals, color=colors, edgecolor='black', alpha=0.7)
    ax1.axhline(1.0, color='blue', linestyle='--', linewidth=2, label='Perfect (1.0)')
    ax1.axhline(1.1, color='orange', linestyle='--', linewidth=1.5, label='Threshold (1.1)')
    ax1.set_xticks(range(len(params)))
    ax1.set_xticklabels(params, rotation=45, ha='right', fontsize=12)
    ax1.set_ylabel('R-hat', fontsize=14)
    ax1.set_title('Gelman-Rubin R-hat Statistic', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Heidelberg-Welch plot
    params_hw = list(hw_results.keys())
    passed = [hw_results[p].get('passed', False) for p in params_hw]
    colors_hw = ['green' if p else 'red' for p in passed]
    
    ax2.bar(range(len(params_hw)), [1 if p else 0 for p in passed], 
           color=colors_hw, edgecolor='black', alpha=0.7)
    ax2.set_xticks(range(len(params_hw)))
    ax2.set_xticklabels(params_hw, rotation=45, ha='right', fontsize=12)
    ax2.set_ylabel('Test Result', fontsize=14)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Failed', 'Passed'], fontsize=12)
    ax2.set_title('Heidelberg-Welch Diagnostic', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_single_layer_by_dimension(
    mean_log_scores: Mapping[Any, Mapping[Any, Mapping[Any, Sequence[float]]]],
    sample_size: Any,
    layer: Any,
    save_path: Optional[str] = None,
    xlabel: str = "Method",
    ylabel: str = "RMSPE",
    facecolor: str = "skyblue",
    ylim: Optional[Tuple[float, float]] = None,
    yscale: str = "linear",
    symlog_linthresh: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 6),
    show: bool = False
):
    """
    Boxplot scores for one layer and sample size across dimensions or methods.

    Args:
        mean_log_scores: Nested dictionary with structure
            {dimension_or_method: {sample_size: {layer: scores}}}
        sample_size: Sample size key to plot
        layer: Layer key to plot
        save_path: Optional path to save the figure
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        facecolor: Box fill color
        ylim: Optional y-axis limits; set to None for automatic scaling
        yscale: Y-axis scale, either "linear" or "symlog"
        symlog_linthresh: Optional linear threshold for symlog scaling
        figsize: Figure size
        show: Whether to display the plot interactively
    """
    dimensions = sorted(mean_log_scores.keys(), key=_dimension_sort_key)

    data = []
    x_labels = []

    for dim in dimensions:
        sample_scores = _mapping_get(mean_log_scores[dim], sample_size)
        if sample_scores is None:
            continue

        scores = _mapping_get(sample_scores, layer)
        if scores is None:
            continue

        scores = np.asarray(scores, dtype=float).ravel()
        if scores.size == 0:
            continue

        data.append(scores)
        x_labels.append(str(dim))

    if not data:
        print(f"No data available for layer {layer} and sample size {sample_size}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        data,
        patch_artist=True,
        boxprops=dict(facecolor=facecolor, edgecolor='black'),
        medianprops=dict(color='black')
    )

    ax.set_xticks(range(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels, fontsize=26)
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)

    if ylim is not None:
        ax.set_ylim(*ylim)

    _apply_yaxis_scale(
        ax,
        data,
        yscale=yscale,
        symlog_linthresh=symlog_linthresh,
        ylim=ylim,
        fontsize=26,
    )

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=26)

    plt.tight_layout()

    _save_plot(save_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_grouped_boxplot_by_dimension(
    mean_log_scores: Mapping[Any, Mapping[Any, Mapping[Any, Sequence[float]]]],
    sample_size: Any,
    save_path: Optional[str] = None,
    model_names: Optional[Any] = None,
    colors: Optional[Sequence[str]] = None,
    xlabel: str = "Method",
    ylabel: str = "RMSPE",
    ylim: Optional[Tuple[float, float]] = None,
    yscale: str = "linear",
    symlog_linthresh: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 6),
    group_spacing: float = 4.0,
    box_width: float = 0.5,
    layer_spacing: float = 0.6,
    show: bool = False
):
    """
    Grouped boxplot of all available layers for one sample size.

    Args:
        mean_log_scores: Nested dictionary with structure
            {dimension_or_method: {sample_size: {layer: scores}}}
        sample_size: Sample size key to plot
        save_path: Optional path to save the figure
        model_names: Optional layer labels as a dict keyed by layer or a sequence
            ordered like the sorted layers
        colors: Optional box colors ordered by layer
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        ylim: Optional y-axis limits; set to None for automatic scaling
        yscale: Y-axis scale, either "linear" or "symlog"
        symlog_linthresh: Optional linear threshold for symlog scaling
        figsize: Figure size
        group_spacing: Distance between x-axis dimension/method groups
        box_width: Width of each boxplot
        layer_spacing: Distance between boxes within each group
        show: Whether to display the plot interactively
    """
    default_names = ["Standard GP", "2-layer DGP", "3-layer DGP"]
    colors = colors or ['skyblue', 'salmon', 'lightgreen', 'plum', 'orange']

    dimensions = sorted(mean_log_scores.keys(), key=_dimension_sort_key)
    sample_scores_by_dim = {}
    layer_ids = {}

    for dim in dimensions:
        sample_scores = _mapping_get(mean_log_scores[dim], sample_size)
        if not isinstance(sample_scores, Mapping):
            continue

        sample_scores_by_dim[dim] = sample_scores
        for layer in sample_scores.keys():
            try:
                normalized_layer = int(layer)
            except (TypeError, ValueError):
                normalized_layer = str(layer)
            layer_ids.setdefault(normalized_layer, layer)

    model_layers = sorted(layer_ids.keys(), key=_dimension_sort_key)

    if not sample_scores_by_dim or not model_layers:
        print(f"No data available for sample size {sample_size}")
        return

    def layer_label(layer, index):
        if isinstance(model_names, Mapping):
            label = _mapping_get(model_names, layer)
            if label is not None:
                return str(label)
        elif model_names is not None:
            try:
                return str(model_names[index])
            except (IndexError, TypeError):
                pass

        if index < len(default_names):
            return default_names[index]
        return f"{layer}-layer"

    data = []
    positions = []
    color_list = []
    xtick_labels = []
    xtick_positions = []

    offsets = [
        (j - (len(model_layers) - 1) / 2) * layer_spacing
        for j in range(len(model_layers))
    ]

    for i, dim in enumerate(dimensions):
        sample_scores = sample_scores_by_dim.get(dim)
        if sample_scores is None:
            continue

        group_has_data = False
        group_position = len(xtick_labels) * group_spacing

        for j, layer in enumerate(model_layers):
            scores = _mapping_get(sample_scores, layer)
            if scores is None:
                continue

            scores = np.asarray(scores, dtype=float).ravel()
            if scores.size == 0:
                continue

            data.append(scores)
            positions.append(group_position + offsets[j])
            color_list.append(colors[j % len(colors)])
            group_has_data = True

        if group_has_data:
            xtick_labels.append(str(dim))
            xtick_positions.append(group_position)

    if not data:
        print(f"No data available for sample size {sample_size}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=True)

    for patch, color in zip(bp['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    for median in bp['medians']:
        median.set_color('black')

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=24)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    if ylim is not None:
        ax.set_ylim(*ylim)

    _apply_yaxis_scale(
        ax,
        data,
        yscale=yscale,
        symlog_linthresh=symlog_linthresh,
        ylim=ylim,
        fontsize=24,
    )

    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    legend_items = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i % len(colors)],
            lw=10,
            label=layer_label(model_layers[i], i)
        )
        for i in range(len(model_layers))
    ]
    ax.legend(
        handles=legend_items,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(model_layers),
        fontsize=24,
        frameon=False
    )

    plt.tight_layout()

    _save_plot(save_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_metrics_boxplot(metrics_chains: List[dict], save_path: Optional[str] = None):
    """
    Boxplot of performance metrics across chains.
    
    Args:
        metrics_chains: List of metric dictionaries from each chain
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = [
        ('rmspe', 'RMSPE'),
        ('nsme', 'NSME'),
        ('crps', 'CRPS'),
        ('score', 'Score'),
        ('mlppd', 'MLPPD'),
        ('bic', 'BIC'),
        ('cp', 'CP'),
        ('alci', 'ALCI')
    ]
    data = []
    labels = []
    
    for metric_lower, metric_upper in metric_names:
        metric_key = None
        if metric_lower in metrics_chains[0]:
            metric_key = metric_lower
        elif metric_upper in metrics_chains[0]:
            metric_key = metric_upper
        
        if metric_key is not None:
            values = [m[metric_key] for m in metrics_chains]
            data.append(values)
            labels.append(metric_upper)
    
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', edgecolor='black'),
                       medianprops=dict(color='red', linewidth=2))
        
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title('Performance Metrics Across Chains', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        _apply_yaxis_scale(ax, data, fontsize=12)
    
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_metrics_comparison_table(
    metrics_summary: dict,
    save_path: Optional[str] = None,
    title: str = 'Performance Metrics Summary'
):
    """
    Create a visual table of metrics with mean, median, CI.
    
    Args:
        metrics_summary: Dictionary with metric summaries
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Metric', 'Mean', 'Median', 'Std', '95% CI'])
    
    for metric_name, metric_vals in metrics_summary.items():
        if isinstance(metric_vals, dict) and 'mean' in metric_vals:
            row = [
                metric_name.upper(),
                f"{metric_vals['mean']:.4f}",
                f"{metric_vals['median']:.4f}",
                f"{metric_vals['std']:.4f}",
                f"[{metric_vals['ci_lower']:.4f}, {metric_vals['ci_upper']:.4f}]"
            ]
            table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    _save_plot(save_path)
    plt.close()
