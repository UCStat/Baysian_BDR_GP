# BDR Metrics and Plot Module

This module provides all performance metrics and diagnostic plotting functions for Bayesian Dimensionality Reduction models, supporting both full models and layer variants.

## Contents

- `BDR_metrics.py` - All performance metric computations
- `BDR_plot.py` - All diagnostic plotting functions
- `__init__.py` - Package initialization

## Performance Metrics

### 1. RMSPE (Root Mean Square Predictive Error)

```python
from BDR_metrics import compute_RMSPE

rmspe = compute_RMSPE(y_true, y_pred)
```
- **Lower is better**
- Measures point prediction accuracy
- Formula: √(1/n_test ∑(y_true - y_pred)²)

### 2. NSME (Nash-Sutcliffe Model Efficiency)

```python
from BDR_metrics import compute_NSME

nsme = compute_NSME(y_true, y_pred)
```
- **Range:** (-∞, 1], where 1 is perfect
- Measures predictive skill vs. naive baseline
- Formula: 1 - ∑(y_true - y_pred)² / ∑(y_true - ȳ)²

### 3. CRPS (Continuous Ranked Probability Score)

```python
from BDR_metrics import compute_CRPS

crps = compute_CRPS(y_true, y_pred_samples, y_pred_vars)
```
- **Lower is better**
- Measures calibration of probability forecasts
- Integrates over all threshold levels
- Handles full predictive distributions

### 4. BIC (Bayesian Information Criterion)

```python
from BDR_metrics import compute_BIC

bic = compute_BIC(log_likelihood, n_samples, n_parameters)
```
- **Lower is better**
- Balances fit and complexity
- Formula: -2·loglik + k·log(n)
- **Critical for JUQ paper:** For multi-layer models, sums log-likelihoods across layers
- Correctly handles D=1 and D>1 cases

### 5. MLPPD (Mean Log Pointwise Predictive Density)

```python
from BDR_metrics import compute_MLPPD

mlppd = compute_MLPPD(y_true, y_pred_samples, y_pred_vars)
```
- **Higher is better**
- Measures predictive distribution quality
- Formula: (1/n_test) ∑ log N(y_true | y_pred_mean, y_pred_var)
- Uses full posterior predictive distribution

### 6. Score (Predictive Log-Likelihood)

```python
from BDR_metrics import compute_score

score = compute_score(y_true, y_pred_mean, y_pred_cov)
```
- **Higher is better**
- Overall model fit measure
- Uses covariance matrix of predictions
- Formula: -0.5·[n·log(2π) + log|Σ| + (y - μ)ᵀΣ⁻¹(y - μ)]

### Comprehensive Summary

```python
from BDR_metrics import compute_all_metrics_summary

metrics = compute_all_metrics_summary(
    y_true=y_test,
    y_pred_samples=y_pred_samples,  # (n_samples, n_test)
    y_pred_vars=y_pred_vars,        # (n_samples, n_test)
    BIC_samples=bic_samples          # (n_samples,)
)
```

Returns all 6 metrics with summary statistics (mean, median, std, CI).

## Diagnostic Plots

### 1. Trace Plots

```python
from BDR_plot import plot_trace

plot_trace(
    chains=[chain1_tau2, chain2_tau2, chain3_tau2],
    param_name='tau2',
    save_path='./trace_tau2.png'
)
```
- Visualizes parameter evolution over iterations
- Multiple chains shown in different colors
- Assesses convergence and mixing
- Handles both scalar and vector parameters

### 2. Density Plots

```python
from BDR_plot import plot_density

plot_density(
    chains=[chain1_tau2, chain2_tau2],
    param_name='tau2',
    save_path='./density_tau2.png'
)
```
- Shows posterior distribution
- Vertical lines for mean and median
- Multiple chains overlay
- KDE smoothing for smooth curves

### 3. Histograms

```python
from BDR_plot import plot_histogram

plot_histogram(
    samples=tau2_samples,
    param_name='tau2',
    bins=50,
    save_path='./hist_tau2.png'
)
```
- Distribution visualization
- Configurable number of bins
- Shows empirical distribution

### 4. Autocorrelation Plots

```python
from BDR_plot import plot_autocorrelation

plot_autocorrelation(
    samples=tau2_samples,
    param_name='tau2',
    max_lag=50,
    save_path='./acf_tau2.png'
)
```
- Assesses chain mixing
- Shows correlation at different lags
- Lower autocorrelation indicates better mixing
- Helps determine thinning interval

### 5. W Trace Plots (Multi-Chain, Full Models Only)

```python
from BDR_plot import plot_W_trace_multichain

plot_W_trace_multichain(
    W_chains=[W_chain1, W_chain2, W_chain3],
    save_path='./W_trace.png'
)
```
- Special handling for projection matrix W
- Shows all elements of W
- Multiple chains overlay
- Useful for assessing W convergence

### 6. Actual vs. Predicted

```python
from BDR_plot import plot_actual_vs_predicted

plot_actual_vs_predicted(
    y_true=y_test,
    pred_mean=y_pred_mean,
    pred_lower=y_pred - 1.96*y_pred_std,
    pred_upper=y_pred + 1.96*y_pred_std,
    output_dir='./diagnostics',
    variant='W_Known'  # Optional, for variants
)
```
- Model fit visualization
- Shows prediction intervals
- Perfect prediction = 45° line
- Includes uncertainty bands

### 7. Convergence Diagnostics Table

```python
from BDR_plot import plot_convergence_diagnostics

plot_convergence_diagnostics(
    convergence_dict=results['convergence'],
    output_dir='./diagnostics'
)
```
- Tabular display of R̂ and HW statistics
- Color-coded for easy interpretation
- Green = good (R̂ < 1.1), yellow = check, red = poor
- Shows all parameters

### 8. Metrics Boxplots

```python
from BDR_plot import plot_metrics_boxplot

plot_metrics_boxplot(
    metrics_dict=results['metrics_summary'],
    output_dir='./diagnostics'
)
```
- Compare metrics across chains
- Shows distribution and outliers
- Useful for multi-chain analysis

### 9. Metrics Comparison Table

```python
from BDR_plot import plot_metrics_comparison_table

plot_metrics_comparison_table(
    chains_metrics=results['chains_metrics'],
    output_dir='./diagnostics'
)
```
- Comprehensive metrics table
- Includes mean, median, std, CI for all chains
- Publication-ready format
- Shows all 6 metrics

### 10. Single Layer Boxplot by Dimension or Method

```python
from BDR_plot import plot_single_layer_by_dimension

plot_single_layer_by_dimension(
    mean_log_scores=mean_log_scores,
    sample_size=480,
    layer=3,
    save_path='./plot/1d_rmse_480_l3.pdf',
    xlabel='Method',
    ylabel='RMSPE',
    ylim=(-0.01, 0.25)
)
```
- Expects nested score data: `{dimension_or_method: {sample_size: {layer: scores}}}`
- Compares one layer and sample size across dimensions or method labels
- Automatically creates the output directory when `save_path` is provided
- Use `show=True` to display the plot interactively, or `ylim=None` to use automatic y-axis scaling

### 11. Grouped Layer Boxplot by Dimension or Method

```python
from BDR_plot import plot_grouped_boxplot_by_dimension

plot_grouped_boxplot_by_dimension(
    mean_log_scores=mean_log_scores,
    sample_size=280,
    save_path='./plot/rmspe_case1a_280.pdf',
    ylim=(-0.005, 0.35)
)
```
- Expects nested score data: `{dimension_or_method: {sample_size: {layer: scores}}}`
- Plots all available layers side by side for each dimension or method label
- Uses default legend labels `Standard GP`, `2-layer DGP`, and `3-layer DGP`
- Override labels with `model_names={1: 'Standard GP', 2: '2-layer DGP', 3: '3-layer DGP'}`

## Usage Example

```python
import numpy as np
from BDR_metrics import compute_RMSPE, compute_BIC, compute_all_metrics_summary
from BDR_plot import plot_trace, plot_actual_vs_predicted

# Compute metrics
y_true = np.array([...])
y_pred = np.array([...])
y_pred_samples = np.array([...])  # (n_samples, n_test)
y_pred_vars = np.array([...])      # (n_samples, n_test)

rmspe = compute_RMSPE(y_true, y_pred)
print(f"RMSPE: {rmspe:.4f}")

# Comprehensive summary
metrics = compute_all_metrics_summary(
    y_true=y_true,
    y_pred_samples=y_pred_samples,
    y_pred_vars=y_pred_vars,
    BIC_samples=bic_samples
)

# Create plots
plot_trace(
    chains=[chain1, chain2, chain3],
    param_name='tau2',
    save_path='./trace.png'
)

plot_actual_vs_predicted(
    y_true=y_true,
    pred_mean=y_pred,
    output_dir='./diagnostics'
)
```

## Integration with Multichain

The multichain samplers automatically use these functions:

```python
from multichain_sampler_D1 import MultiChainSampler

multichain = MultiChainSampler(...)
results = multichain.run_chains(...)

# Metrics automatically computed
print(results['metrics_summary'])

# Plots automatically generated
multichain.create_all_diagnostics(output_dir='./diagnostics')
```

## Metric Interpretation

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| RMSPE | Lower | Better point predictions |
| NSME | Closer to 1 | Better predictive skill |
| CRPS | Lower | Better probabilistic forecasts |
| BIC | Higher | Better model (penalized by complexity) |
| MLPPD | Higher | Better predictive distributions |
| Score | Higher | Better overall model fit |

## BIC Computation Details

For multi-layer models, BIC correctly sums log-likelihoods:

**1-Layer:**
```

BIC = -2·loglik_y + k·log(n) (smaller means better)  or loglik_y - o.5 ·k·log(n)(higher means better) 
```

**2-Layer:**
```
BIC = -2·(loglik_y + loglik_q) + k·log(n) (smaller means better) or loglik_y + loglik_q - 0.5· k·log(n) (higher means better)
```

**3-Layer:**
```
BIC = -2·(loglik_y + loglik_q + loglik_r) + k·log(n)(smaller means better) or loglik_y + loglik_q + loglik_r -0.5 · k·log(n)(higher means better)
```

For D>1 with separable kernels, latent layer log-likelihoods are summed across dimensions.

## See Also

- `../Multichain/` - Uses these metrics and plots automatically
- `../run_multichains.py` - Main interface
- `../run_multichains.ipynb` - Jupyter notebook with examples
