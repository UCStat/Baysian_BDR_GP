# Multichain Module

This module implements multi-chain MCMC sampling with comprehensive convergence diagnostics and performance evaluation, supporting both full models and layer variants.

## Contents

- `multichain_sampler_D1.py` - Multi-chain sampler for D=1 (full models)
- `multichain_sampler_Dgeneral.py` - Multi-chain sampler for D>1 (full models)
- `multichain_sampler_L1_variants.py` - Multi-chain sampler for Layer 1 variants
- `multichain_sampler_L2_variants.py` - Multi-chain sampler for Layer 2 variants
- `multichain_sampler_L3_variants.py` - Multi-chain sampler for Layer 3 variants
- `__init__.py` - Package initialization

## Features

- ✅ Multiple independent MCMC chains
- ✅ Convergence diagnostics (Gelman-Rubin, Heidelberg-Welch)
- ✅ Performance metrics (RMSPE, NSME, CRPS, BIC, MLPPD, Score)
- ✅ Automated diagnostic plots
- ✅ Posterior predictive inference
- ✅ Support for full models and layer variants

## Usage

### Full Models (D=1)

```python
from multichain_sampler_D1 import MultiChainSampler

# Initialize
multichain = MultiChainSampler(
    n_chains=3,
    layer=1,
    n_iterations=2000,
    burn_in=500,
    thin=2,
    use_mle_all=True,
    use_tf_gradients=False,
    kernel_type='isotropic_squared_exponential',
    prior_M=prior_M,
    prior_V=prior_V
)

# Run chains
results = multichain.run_chains(Y_train, X_train, Y_test, X_test, verbose=True)

# Create diagnostics
multichain.create_all_diagnostics(output_dir='./diagnostics')

# Compute summary
summary = multichain.compute_summary()
```

### Full Models (D>1)

```python
from multichain_sampler_Dgeneral import MultiChainSampler

# Initialize with D=2
multichain = MultiChainSampler(
    D=2,
    n_chains=3,
    layer=2,
    n_iterations=2000,
    burn_in=500,
    thin=2,
    use_mle_all=True,
    use_tf_gradients=True,  # Recommended for D>1
    kernel_type='separable_squared_exponential',
    prior_M=prior_M,
    prior_V=prior_V
)

results = multichain.run_chains(Y_train, X_train, Y_test, X_test)
```

### Layer 1 Variants

```python
from multichain_sampler_L1_variants import MultiChainSampler_L1_Variants

# W_Known variant
multichain = MultiChainSampler_L1_Variants(
    variant='W_Known',
    W_fixed=W_fixed,  # Required for W_Known
    n_chains=3,
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g=False,
    use_mle_theta=True,
    kernel_type='separable_squared_exponential'
)

results = multichain.run_chains(Y_train, X_train, Y_test, X_test)
```

### Layer 2 Variants

```python
from multichain_sampler_L2_variants import MultiChainSampler_L2_Variants

# No_W variant
multichain = MultiChainSampler_L2_Variants(
    variant='No_W',
    n_chains=3,
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g_y=False,
    use_mle_theta_y=True,
    kernel_type='separable_squared_exponential'
)

results = multichain.run_chains(Y_train, X_train, Y_test, X_test)
```

### Layer 3 Variants

```python
from multichain_sampler_L3_variants import MultiChainSampler_L3_Variants

# No_W_Selective variant
multichain = MultiChainSampler_L3_Variants(
    variant='No_W_Selective',
    D=3,
    column_indices=np.array([0, 1, 2]),
    n_chains=3,
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g_y=True,
    use_mle_theta_y=True,
    kernel_type='separable_squared_exponential'
)

results = multichain.run_chains(Y_train, X_train, Y_test, X_test)
```

## Convergence Diagnostics

### Gelman-Rubin Statistic (R̂)

- Compares within-chain and between-chain variance
- Values < 1.1 indicate convergence
- Computed for all parameters:
  - **Layer 1:** τ², g, θ, W, M, V, Λ
  - **Layer 2:** τ²_y, g_y, θ_y, θ_q, Q, W, M, V, Λ
  - **Layer 3:** τ²_y, g_y, θ_y, θ_q, θ_r, Q, R, W, M, V, Λ
  - **Variants:** Only parameters that are sampled

### Heidelberg-Welch Test

- Tests for stationarity and halfwidth mean
- Based on Cramer-von Mises statistic
- Returns: "Passed", "Failed", or "Not enough samples"

## Performance Metrics

All metrics include mean, median, std, and 95% credible intervals:

1. **RMSPE** (Root Mean Square Predictive Error)
   - Lower is better
   - Measures point prediction accuracy
   - Formula: √(1/n_test ∑(y_true - y_pred)²)

2. **NSME** (Nash-Sutcliffe Model Efficiency)
   - Range: (-∞, 1], 1 is perfect
   - Measures predictive skill vs. naive baseline
   - Formula: 1 - ∑(y_true - y_pred)² / ∑(y_true - ȳ)²

3. **CRPS** (Continuous Ranked Probability Score)
   - Lower is better
   - Measures calibration of probability forecasts
   - Integrates over all threshold levels

4. **BIC** (Bayesian Information Criterion)
   - Lower is better
   - **Critical for JUQ paper:** Sums log-likelihoods across layers
   - Correctly handles D=1 and D>1 cases
   - Formula: -2·(loglik_y + loglik_q + loglik_r) + k·log(n)

5. **MLPPD** (Mean Log Pointwise Predictive Density)
   - Higher is better
   - Measures predictive distribution quality
   - Formula: (1/n_test) ∑ log N(y_true | y_pred_mean, y_pred_var)

6. **Score** (Predictive Log-Likelihood)
   - Higher is better
   - Overall model fit measure
   - Uses covariance of predictions across samples

## BIC Computation (Important!)

The BIC is computed correctly for all layer configurations:

**1-Layer:**
```
BIC = -2·loglik_y + k·log(n)
```

**2-Layer:**
```
BIC = -2·(loglik_y + loglik_q) + k·log(n)
```

**3-Layer:**
```
BIC = -2·(loglik_y + loglik_q + loglik_r) + k·log(n)
```

For D>1, latent layer log-likelihoods are summed across dimensions.

For variants, BIC computation adapts to sampled parameters only.

## Output Structure

`results` dictionary contains:

```python
{
    'chains_samples': [
        # Chain 1 samples
        {'tau2': array, 'g': array, 'theta_D': array, 'W': array, ...},
        # Chain 2 samples
        ...
    ],
    'chains_metrics': [
        # Chain 1 metrics
        {'RMSPE': float, 'NSME': float, 'CRPS': float, 'BIC': float, ...},
        # Chain 2 metrics
        ...
    ],
    'convergence': {
        'r_hat_tau2': float,
        'r_hat_g': float,  # or r_hat_g_y, r_hat_g_q, r_hat_g_r
        'r_hat_theta': float,  # or r_hat_theta_y, etc.
        # ... for all parameters
    },
    'metrics_summary': {
        'RMSPE': {'mean': ..., 'median': ..., 'std': ..., 'ci_lower': ..., 'ci_upper': ...},
        'NSME': {...},
        'CRPS': {...},
        'BIC': {...},
        'MLPPD': {...},
        'Score': {...}
    },
    'computation_times': [time_chain1, time_chain2, ...],
    'variant': 'W_Known'  # For variants only
}
```

## Diagnostic Plots

The `create_all_diagnostics()` method generates:

1. **Trace plots** - Parameter evolution over iterations (all chains)
2. **Density plots** - Posterior distributions with mean/median (all chains combined)
3. **Histograms** - Sample distributions
4. **Autocorrelation plots** - Assess mixing
5. **W trace plots** - Special handling for matrix W (full models only)
6. **Actual vs. Predicted** - Model fit visualization
7. **Metrics boxplots** - Performance comparison across chains
8. **Convergence table** - R̂ and HW statistics

## Layer-Dependent Parameters

The samplers automatically adapt diagnostics based on layer:

| Layer | Parameters Tracked (Full Models) |
|-------|-----------------------------------|
| 1 | τ², g, θ, W, M, V, Λ |
| 2 | τ²_y, g_y, g_q, θ_y, θ_q, Q, W, M, V, Λ |
| 3 | τ²_y, g_y, g_q, g_r, θ_y, θ_q, θ_r, R, Q, W, M, V, Λ |

| Variant | Parameters Tracked |
|---------|-------------------|
| L1 Variants | τ², g, θ_D |
| L2 Variants | τ²_y, g_y, θ_y, θ_q, Q |
| L3 Variants | τ²_y, g_y, θ_y, θ_q, θ_r, Q, R |

## Prediction

All multi-chain samplers compute posterior predictive distributions:

- **Predictive mean:** Average across all samples and chains
- **Predictive variance:** Total variance (within-sample + between-sample)
- **Uncertainty quantification:** Full posterior predictive intervals

## Integration with Main Interface

The multi-chain samplers are automatically used by `run_multichain_analysis()`:

```python
from run_multichains import run_multichain_analysis

# Full model
results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    D=1, layer=1,
    n_chains=3,
    **config
)

# Variant model
results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    layer=1, variant='W_Known',
    W_fixed=W_fixed,
    n_chains=3,
    **config
)
```

## See Also

- `../run_multichains.py` - Main interface with all hyperparameters exposed
- `../run_multichains.ipynb` - Jupyter notebook with examples
- `../BDR Metrics and Plot/` - Metric and plot implementations
- `../Gibbs Sampling/` - Single-chain samplers
