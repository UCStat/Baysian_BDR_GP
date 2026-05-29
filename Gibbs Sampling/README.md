# Gibbs Sampling Module

This module implements Gibbs samplers for 1-layer, 2-layer, and 3-layer Deep GP models, supporting both D=1 and D>1 cases, as well as simplified variant models.

## Contents

- `gibbs_sampler_layers_D1.py` - Full Gibbs samplers for D=1
- `gibbs_sampler_layers_Dgeneral.py` - Full Gibbs samplers for D>1
- `gibbs_sampler_layers_L1_variants.py` - Layer 1 variant samplers (W_Known, No_W, No_W_Selective)
- `gibbs_sampler_layers_L2_variants.py` - Layer 2 variant samplers (W_Known, No_W, No_W_Selective)
- `gibbs_sampler_layers_L3_variants.py` - Layer 3 variant samplers (W_Known, No_W, No_W_Selective)
- `__init__.py` - Package initialization

## Model Architectures

### 1-Layer GP (Full Model)
- **Structure:** X → Z → Y
- **Parameters:** τ², g, θ, W, M, V, Λ
- **Sampler:** `GibbsSampler1Layer` (D1 or Dgeneral)

### 2-Layer Deep GP (Full Model)
- **Structure:** X → Z → Q → Y
- **Parameters:** τ²_y, g_y, θ_y, Q, g_q, tau2_q, θ_q, W, M, V, Λ
- **Note:** g_q=0.0 (fixed), tau2_q=1.0 (fixed), only θ_q sampled
- **Sampler:** `GibbsSampler2Layer` (D1 or Dgeneral)

### 3-Layer Deep GP (Full Model)
- **Structure:** X → Z → R → Q → Y
- **Parameters:** τ²_y, g_y, θ_y, Q, θ_q, R, θ_r, W, M, V, Λ
- **Note:** g_q=g_r=0.0 (fixed), tau2_q=tau2_r=1.0 (fixed)
- **Sampler:** `GibbsSampler3Layer` (D1 or Dgeneral)

## Full Model Samplers

### D=1 Case

```python
from gibbs_sampler_layers_D1 import (
    GibbsSampler1Layer,
    GibbsSampler2Layer,
    GibbsSampler3Layer
)

# 1-Layer
sampler = GibbsSampler1Layer(
    Y=Y_train, X=X_train,
    n_iterations=2000,
    burn_in=500,
    thin=2,
    use_mle_tau2=False,
    use_mle_g=False,
    use_mle_theta=False,
    use_tf_gradients=False,
    kernel_type='isotropic_squared_exponential',
    prior_M=prior_M,
    prior_V=prior_V
)

samples = sampler.run(verbose=True)
```

### D>1 Case

```python
from gibbs_sampler_layers_Dgeneral import (
    GibbsSampler1Layer,
    GibbsSampler2Layer,
    GibbsSampler3Layer
)

# 2-Layer with D=2
sampler = GibbsSampler2Layer(
    Y=Y_train, X=X_train, D=2,
    n_iterations=2000,
    burn_in=500,
    thin=2,
    use_mle_tau2=True,
    use_mle_g_y=True,
    use_mle_theta_y=True,
    use_tf_gradients=True,  # Recommended for D>1
    kernel_type='separable_squared_exponential',
    prior_M=prior_M,
    prior_V=prior_V
)

samples = sampler.run(verbose=True)
```

## Layer Variant Samplers

### Layer 1 Variants

**1. W_Known Variant:**
```python
from gibbs_sampler_layers_L1_variants import GibbsSampler1Layer_W_Known

sampler = GibbsSampler1Layer_W_Known(
    Y=Y_train, X=X_train, W_fixed=W_fixed,  # W is known
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g=False,
    use_mle_theta=True,
    kernel_type='separable_squared_exponential'
)
# Samples: tau2, g, theta_D (W, M, V, Lambda not sampled)
```

**2. No_W Variant:**
```python
from gibbs_sampler_layers_L1_variants import GibbsSampler1Layer_No_W

sampler = GibbsSampler1Layer_No_W(
    Y=Y_train, X=X_train,  # Use X directly
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g=True,
    use_mle_theta=False,
    kernel_type='separable_squared_exponential'
)
# Samples: tau2, g, theta_D (W, M, V, Lambda not needed)
```

**3. No_W_Selective Variant:**
```python
from gibbs_sampler_layers_L1_variants import GibbsSampler1Layer_No_W_Selective

sampler = GibbsSampler1Layer_No_W_Selective(
    Y=Y_train, X=X_train, D=3,
    column_indices=np.array([0, 1, 2]),  # Use selected columns
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g=True,
    use_mle_theta=True,
    kernel_type='separable_squared_exponential'
)
# Samples: tau2, g, theta_D (W, M, V, Lambda not needed)
```

### Layer 2 Variants

**1. W_Known Variant:**
```python
from gibbs_sampler_layers_L2_variants import GibbsSampler2Layer_W_Known

sampler = GibbsSampler2Layer_W_Known(
    Y=Y_train, X=X_train, W_fixed=W_fixed,
    n_iterations=2000,
    use_mle_tau2=True,    # Y layer
    use_mle_g_y=False,    # Y layer
    use_mle_theta_y=True, # Y layer
    kernel_type='separable_squared_exponential'
)
# Samples: Q (ESS), theta_q (MCMC), tau2_y, g_y, theta_y (MLE/MCMC)
# Fixed: g_q=0.0, tau2_q=1.0
```

**2. No_W Variant:**
```python
from gibbs_sampler_layers_L2_variants import GibbsSampler2Layer_No_W

sampler = GibbsSampler2Layer_No_W(
    Y=Y_train, X=X_train,
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g_y=False,
    use_mle_theta_y=True,
    kernel_type='separable_squared_exponential'
)
```

**3. No_W_Selective Variant:**
```python
from gibbs_sampler_layers_L2_variants import GibbsSampler2Layer_No_W_Selective

sampler = GibbsSampler2Layer_No_W_Selective(
    Y=Y_train, X=X_train, D=3,
    column_indices=np.array([0, 1, 2]),
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g_y=True,
    use_mle_theta_y=True,
    kernel_type='separable_squared_exponential'
)
```

### Layer 3 Variants

**1. W_Known Variant:**
```python
from gibbs_sampler_layers_L3_variants import GibbsSampler3Layer_W_Known

sampler = GibbsSampler3Layer_W_Known(
    Y=Y_train, X=X_train, W_fixed=W_fixed,
    n_iterations=2000,
    use_mle_tau2=True,    # Y layer
    use_mle_g_y=False,    # Y layer
    use_mle_theta_y=True, # Y layer
    kernel_type='separable_squared_exponential'
)
# Samples: R (ESS), Q (ESS), theta_r (MCMC), theta_q (MCMC), tau2_y, g_y, theta_y (MLE/MCMC)
# Fixed: g_r=g_q=0.0, tau2_r=tau2_q=1.0
```

**2. No_W Variant:**
```python
from gibbs_sampler_layers_L3_variants import GibbsSampler3Layer_No_W

sampler = GibbsSampler3Layer_No_W(
    Y=Y_train, X=X_train,
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g_y=False,
    use_mle_theta_y=True,
    kernel_type='separable_squared_exponential'
)
```

**3. No_W_Selective Variant:**
```python
from gibbs_sampler_layers_L3_variants import GibbsSampler3Layer_No_W_Selective

sampler = GibbsSampler3Layer_No_W_Selective(
    Y=Y_train, X=X_train, D=3,
    column_indices=np.array([0, 1, 2]),
    n_iterations=2000,
    use_mle_tau2=True,
    use_mle_g_y=True,
    use_mle_theta_y=True,
    kernel_type='separable_squared_exponential'
)
```

## Sampling Options

### Estimation Methods

1. **Full MCMC:** All parameters sampled
   ```python
   sampler = GibbsSampler1Layer(use_mle_all=False)
   ```

2. **Individual MLE Flags:** Choose MLE or MCMC per parameter
   ```python
   sampler = GibbsSampler1Layer(
       use_mle_tau2=True,   # MLE for tau2
       use_mle_g=False,     # MCMC for g
       use_mle_theta=True   # MLE for theta
   )
   ```

3. **All MLE:** Fastest option
   ```python
   sampler = GibbsSampler1Layer(use_mle_all=True)
   ```

### Gradient Computation for W

- **NumPy (default):** Analytical gradients
  ```python
  sampler = GibbsSampler1Layer(use_tf_gradients=False)
  ```

- **TensorFlow:** Automatic differentiation
  ```python
  sampler = GibbsSampler1Layer(use_tf_gradients=True)  # Recommended for D>1
  ```

### Kernel Selection

All samplers support kernel type selection:
- `'isotropic_squared_exponential'` (D=1 or isotropic)
- `'separable_squared_exponential'` (D>1, recommended)
- `'isotropic_matern32'` (D=1 or isotropic)
- `'separable_matern32'` (D>1)

## Output Format

All samplers return a dictionary:

```python
{
    'tau2': array,        # Observation noise samples (or tau2_y for multi-layer)
    'g': array,           # Nugget samples (or g_y, g_q, g_r for multi-layer)
    'theta_D': array,     # Lengthscale samples (or theta_y, theta_q, theta_r)
    'W': array,           # Projection matrix samples (n_samples, p, D) - Full models only
    'M': array,           # Prior mean samples - Full models only
    'V': array,           # Prior variance samples - Full models only
    'Lambda': array,      # Concentration matrix samples - Full models only
    'Q': array,           # Latent Q (2-layer and 3-layer only)
    'R': array,           # Latent R (3-layer only)
    'computation_time': float
}
```

## Potential Energy Functions

The samplers use geodesic flows on the Stiefel manifold with negative log-posterior as potential energy:

**1-Layer:**
```
U(W) = -log p(Y|Z,τ²,g,θ) - log p(W|M,V,Λ)
```

**2-Layer:**
```
U(W) = -log p(Y|Q,τ²,g_y,θ_y) - log p(Q|Z,g_q,θ_q) - log p(W|M,V,Λ)
```

**3-Layer:**
```
U(W) = -log p(Y|Q,τ²,g_y,θ_y) - log p(Q|R,g_q,θ_q) - log p(R|Z,g_r,θ_r) - log p(W|M,V,Λ)
```

**Note:** Variant models skip the W prior term (W is fixed or not needed).

## Hierarchical Lengthscale Priors (Multi-Layer)

For multi-layer models, lengthscale priors differ by layer:

- **Layer 2:** 
  - `gamma2_y = 3.9` (outer layer Y)
  - `gamma2_q = 3.9/3` (middle layer Q)

- **Layer 3:**
  - `gamma2_y = 3.9` (outer layer Y)
  - `gamma2_q = 3.9/3` (middle layer Q)
  - `gamma2_r = 3.9/6` (inner layer R)

## When to Use Variants

- **W_Known:** When you have a known projection matrix (e.g., from PCA)
- **No_W:** When you do not need dimensionality reduction and want to use all
  columns of `X` directly
- **No_W_Selective:** When you want fixed column-selection reduction using
  `X[:, column_indices]`; if `D=p` and the default columns are used, this is
  effectively the same input as `No_W`

**Advantages:**
- Faster sampling (skip W, M, V, Lambda)
- Simpler models
- Still get full hyperparameter inference

## See Also

- `../Parameter Sampler/` - Individual parameter sampling functions
- `../Multichain/` - Multi-chain framework with diagnostics
- `../run_multichains.py` - Main interface with all options
- `../run_multichains.ipynb` - Jupyter notebook with examples
