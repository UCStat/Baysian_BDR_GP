# Parameter Sampler Module

This module implements MCMC sampling functions for all parameters in Bayesian Dimensionality Reduction models, supporting both D=1 and D>1 cases, as well as multi-layer models.

## Contents

- `parameter_sampler_D1.py` - Sampling for D=1 (single dimension)
- `parameter_sampler_Dgeneral.py` - Sampling for D>1 (multi-dimensional)
- `__init__.py` - Package initialization

## Available Samplers

### D=1 Case

```python
from parameter_sampler_D1 import (
    # Hyperparameters
    sample_tau2,           # Observation noise (Gibbs)
    sample_g,              # Nugget (Metropolis-Hastings)
    sample_theta_D,        # Lengthscale (Metropolis-Hastings)
    
    # MLE alternatives
    estimate_tau2_MLE,
    estimate_g_MLE,
    estimate_theta_D_MLE,
    
    # Projection matrix and priors
    sample_W_HMC_stiefel,  # Projection matrix (HMC on Stiefel)
    sample_M,              # Prior mean for W
    sample_V,              # Prior variance for W
    sample_Lambda_elliptical_slice,  # Concentration matrix (ESS)
    
    # Latent variables (multi-layer)
    sample_Q_2layer_ESS,   # Q for 2-layer (ESS)
    sample_R_3layer_ESS,   # R for 3-layer (ESS)
    sample_Q_3layer_ESS,   # Q for 3-layer (ESS)
    
    # Kernel factory
    get_kernel_instance
)
```

### D>1 Case

```python
from parameter_sampler_Dgeneral import (
    # Same interface as D=1, but handles vectors
    sample_tau2,           # Returns scalar
    sample_g,              # Returns scalar
    sample_theta_D,        # Returns vector (D,)
    
    # MLE alternatives
    estimate_tau2_MLE,
    estimate_g_MLE,
    estimate_theta_D_MLE,
    
    # Projection matrix and priors
    sample_W_HMC_stiefel,  # HMC for matrix W (p×D)
    sample_M, sample_V, sample_Lambda_elliptical_slice,
    
    # Latent variables (multi-layer)
    sample_Q_2layer_ESS,   # Q for 2-layer (ESS, dimension-wise)
    sample_R_3layer_ESS,   # R for 3-layer (ESS, dimension-wise)
    sample_Q_3layer_ESS,   # Q for 3-layer (ESS, dimension-wise)
    
    # Kernel factory
    get_kernel_instance
)
```

## Key Functions

### 1. Observation Noise (τ²)

**Gibbs Sampling:**
```python
tau2_new = sample_tau2(
    Y, Z, g, theta_D,
    alpha1=1.0,      # Inverse Gamma shape
    alpha2=1000.0,   # Inverse Gamma rate
    kernel_type='isotropic_squared_exponential'
)
```
- Prior: InvGamma(α₁, α₂)
- Returns: Scalar τ²
- Posterior: Conjugate update

**MLE:**
```python
tau2_mle = estimate_tau2_MLE(
    Y, Z, g, theta_D,
    kernel_type='isotropic_squared_exponential'
)
```

### 2. Nugget (g)

**Metropolis-Hastings:**
```python
g_new = sample_g(
    g_current, Y, Z, tau2, theta_D,
    beta1=0.01,      # Gamma shape
    beta2=0.005,     # Gamma rate
    l=1.0,           # MH proposal lower bound
    u=2.0,           # MH proposal upper bound
    kernel_type='isotropic_squared_exponential'
)
```
- Prior: Gamma(β₁, β₂)
- Proposal: U[g·l/u, g·u/l]
- Returns: Scalar g

**MLE:**
```python
g_mle = estimate_g_MLE(
    Y, Z, theta_D, tau2,
    n_grid=20,      # Grid search points
    kernel_type='isotropic_squared_exponential'
)
```

### 3. Lengthscale (θ)

**Metropolis-Hastings:**
```python
# D=1: Returns scalar
theta_new = sample_theta_D(
    theta_current, Y, Z, tau2, g,
    gamma1=1.5,     # Gamma shape (3/2)
    gamma2=3.9,     # Gamma rate
    l=1.0, u=2.0,
    kernel_type='isotropic_squared_exponential'
)

# D>1: Returns vector (D,)
# For separable kernels, samples dimension-wise
theta_new = sample_theta_D_Dgen(
    theta_current, Y, Z, W, tau2, g,
    gamma1=1.5, gamma2=3.9,
    l=1.0, u=2.0,
    kernel_type='separable_squared_exponential'
)
```
- Prior: Gamma(γ₁, γ₂)
- For separable kernels (D>1): Loops over dimensions
- Returns: Scalar (D=1) or vector (D>1)

**MLE:**
```python
# D=1
theta_mle = estimate_theta_D_MLE(Y, Z, g, tau2, kernel_type='...')

# D>1: Returns vector
theta_mle = estimate_theta_D_MLE_Dgen(Y, Z, W, g, tau2, kernel_type='...')
```

### 4. Projection Matrix (W)

**HMC on Stiefel Manifold:**
```python
W_new = sample_W_HMC_stiefel(
    Y, X, W_current,
    layer=1,
    tau2_y=tau2, theta_D_y=theta_D, g_y=g,
    F_Wprior=F_Wprior,
    use_tf=False,       # Use TensorFlow for gradients?
    T_step=17,          # Leapfrog steps
    eps=0.001           # Step size
)
```
- Required layer-specific arguments:
  - `layer=1`: `tau2_y`, `theta_D_y`, `g_y`
  - `layer=2`: plus `tau2_q`, `theta_D_q`, `g_q`, and `Q`
  - `layer=3`: plus `tau2_r`, `theta_D_r`, `g_r`, and `R`
- Constraint: W ∈ St(p, D) (Stiefel manifold)
- Sampler: HMC with geodesic flows
- Uses Cayley transform for manifold constraint
- Gradient options: NumPy analytical or TensorFlow automatic

### 5. Prior Parameters (M, V, Λ)

**M (Prior Mean):**
```python
M_new = sample_M(
    W, Lambda, V, p,
    prior_M=prior_M,
    M_prev=M_current,
    mv_sampler="python",   # or "rstiefel"
)
```
- Conjugate matrix-von-Mises-Fisher Gibbs update
- Returns: (p, D) matrix
- `mv_sampler="rstiefel"` calls `rstiefel::rmf.matrix.gibbs(M, X)` through
  `rpy2`; R, `rpy2`, and the R package `rstiefel` must be installed.

**V (Prior Variance):**
```python
V_new = sample_V(
    W, Lambda, M_current, D,
    prior_V=prior_V,
    V_prev=V_current,
    mv_sampler="python",   # or "rstiefel"
)
```
- Conjugate matrix-von-Mises-Fisher Gibbs update
- Returns: (D, D) orthonormal matrix

**Λ (Concentration Matrix):**
```python
Lambda_new = sample_Lambda_elliptical_slice(
    Lambda_current, M, W, V,
    nu=10.0,           # Prior parameter
    epsilon=2.0,      # Positivity constraint
    max_iter=1000     # Maximum ESS iterations
)
```
- Elliptical Slice Sampling (ESS)
- For D>1: Ensures strict descending order of diagonal values
- Returns: (D, D) matrix

### 6. Latent Variables (Multi-Layer Models)

**Q for 2-Layer:**
```python
# D=1
Q_new = sample_Q_2layer_ESS(
    Y, Q_current, Z, g_y, theta_y, theta_q, g_q, tau2,
    kernel_type='isotropic_squared_exponential'
)

# D>1: Dimension-wise ESS
Q_new = sample_Q_2layer_ESS_Dgen(
    Y, Q_current, Z, g_y, theta_y, theta_q, g_q, tau2,
    kernel_type='separable_squared_exponential'
)
```

**R and Q for 3-Layer:**
```python
# D=1
R_new = sample_R_3layer_ESS(Y, R_current, Q, g_q, theta_q, theta_r, g_r, ...)
Q_new = sample_Q_3layer_ESS(Q_current, R, Y, g_y, theta_y, g_q, theta_q, ...)

# D>1: Dimension-wise ESS
R_new = sample_R_3layer_ESS_Dgen(Y, R_current, Z, g_q, theta_q, theta_r, g_r, ...)
Q_new = sample_Q_3layer_ESS_Dgen(Q_current, R, Y, g_y, theta_y, g_q, theta_q, ...)
```

## Hyperparameter Configuration

| Parameter | Prior | Hyperparameters | MH Bounds | Sampler |
|-----------|-------|-----------------|-----------|---------|
| τ² | InvGamma | α₁=1.0, α₂=1000 | - | Gibbs |
| g | Gamma | β₁=0.01, β₂=0.005 | l=1.0, u=2.0 | MH |
| θ | Gamma | γ₁=1.5, γ₂=3.9 | l=1.0, u=2.0 | MH |
| θ_q (2-layer) | Gamma | γ₁=1.5, γ₂=3.9/3 | l=1.0, u=2.0 | MH |
| θ_r (3-layer) | Gamma | γ₁=1.5, γ₂=3.9/6 | l=1.0, u=2.0 | MH |
| Λ | - | ν=10·ones(D), ε=2.0 | - | ESS |
| M | Normal | M₀, κ | - | Conjugate |
| V | InvGamma | a, b | - | Conjugate |

## MLE Options

All hyperparameters support MLE as an alternative to MCMC:

```python
# Individual MLE
tau2_mle = estimate_tau2_MLE(Y, Z, g, theta_D, kernel_type='...')
g_mle = estimate_g_MLE(Y, Z, theta_D, tau2, kernel_type='...')
theta_mle = estimate_theta_D_MLE(Y, Z, g, tau2, kernel_type='...')

# Note: W, M, V, Lambda, Q, R always use MCMC (no MLE option)
```

## Special Features

### Dimension-Wise Sampling for D>1

For separable kernels with D>1:
- `sample_theta_D` loops over dimensions: `for m in range(D)`
- Each dimension sampled independently
- Input: Column-wise Z[:, m], response: Full Y
- Returns: Vector of lengthscales (D,)

### Strict Descending Order for Lambda (D>1)

For D>1, `sample_Lambda_elliptical_slice` ensures:
- Diagonal values in strict descending order
- Resamples if order violated
- Uses ESS with constraint checking

### Latent Layer Hyperparameters

For multi-layer models:
- **Q layer**: `g_q = 0.0` (fixed), `tau2_q = 1.0` (fixed), only `theta_q` sampled
- **R layer**: `g_r = 0.0` (fixed), `tau2_r = 1.0` (fixed), only `theta_r` sampled
- Latent layers use ESS for Q and R sampling

## See Also

- `../Gibbs Sampling/` - Uses these samplers in full Gibbs loops
- `../Multichain/` - Multi-chain framework using these samplers
- `../run_multichains.py` - Main interface with all hyperparameters exposed
- `../SAMPLING_README.md` - Detailed algorithm documentation
