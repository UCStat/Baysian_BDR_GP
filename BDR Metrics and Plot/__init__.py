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
    plot_W_trace_multichain, plot_W_projection_trace_multichain,
    plot_actual_vs_predicted,
    plot_convergence_diagnostics, plot_metrics_boxplot,
    plot_metrics_comparison_table, plot_single_layer_by_dimension,
    plot_grouped_boxplot_by_dimension
)

from .BDR_summaries import (
    build_posterior_parameter_summary, build_time_complexity_summary,
    build_metrics_comparison_rows, compact_model_label_for_run,
    model_label_for_run, write_layer_metric_boxplots,
    write_metrics_comparison_tables, write_run_summary_tables
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
    'plot_W_trace_multichain', 'plot_W_projection_trace_multichain',
    'plot_actual_vs_predicted',
    'plot_convergence_diagnostics', 'plot_metrics_boxplot',
    'plot_metrics_comparison_table', 'plot_single_layer_by_dimension',
    'plot_grouped_boxplot_by_dimension',
    # Summary tables
    'build_posterior_parameter_summary', 'build_time_complexity_summary',
    'build_metrics_comparison_rows', 'compact_model_label_for_run',
    'model_label_for_run', 'write_layer_metric_boxplots',
    'write_metrics_comparison_tables', 'write_run_summary_tables'
]
