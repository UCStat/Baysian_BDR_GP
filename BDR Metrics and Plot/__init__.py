"""BDR Metrics and Plot Module for GP Bayesian Framework."""

from .BDR_metrics import (
    compute_RMSPE, compute_NSME, compute_CRPS, compute_score,
    compute_BIC, compute_MLPPD, compute_CP, compute_ALCI,
    compute_coverage_probability, compute_interval_length,
    compute_all_metrics_summary, compute_iteration_metrics,
    parameter_diagnostics, W_diagnostics, compute_multichain_parameter_diagnostics
)

from .BDR_plot import (
    plot_trace, plot_density, plot_histogram, plot_autocorrelation,
    plot_W_trace_multichain, plot_actual_vs_predicted,
    plot_convergence_diagnostics, plot_metrics_boxplot,
    plot_metrics_comparison_table, plot_single_layer_by_dimension,
    plot_grouped_boxplot_by_dimension
)

__all__ = [
    # Metrics
    'compute_RMSPE', 'compute_NSME', 'compute_CRPS', 'compute_score',
    'compute_BIC', 'compute_MLPPD', 'compute_CP', 'compute_ALCI',
    'compute_coverage_probability', 'compute_interval_length',
    'compute_all_metrics_summary', 'compute_iteration_metrics',
    'parameter_diagnostics', 'W_diagnostics', 'compute_multichain_parameter_diagnostics',
    # Plots
    'plot_trace', 'plot_density', 'plot_histogram', 'plot_autocorrelation',
    'plot_W_trace_multichain', 'plot_actual_vs_predicted',
    'plot_convergence_diagnostics', 'plot_metrics_boxplot',
    'plot_metrics_comparison_table', 'plot_single_layer_by_dimension',
    'plot_grouped_boxplot_by_dimension'
]
