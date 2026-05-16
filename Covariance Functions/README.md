# Covariance Functions Module

This module implements covariance kernel functions with their log-likelihoods and gradients with respect to the projection matrix W.

## Contents

- `covariance_kernel_functions_and_gradients_W.py` - All kernel implementations
- `__init__.py` - Package initialization

## Available Kernels

### 1. Isotropic Squared Exponential

**Use when:** All dimensions share the same lengthscale

```python
from covariance_kernel_functions_and_gradients_W import get_kernel_instance

# For D=1 or isotropic case
kernel = get_kernel_instance('isotropic_squared_exponential', theta=1.0, g=0.01, tau2=1.0, D=1)

# Compute covariance
K = kernel.compute_covariance(Z1, Z2)

# Log-likelihood
loglik = kernel.log_likelihood(Y, Z)

# Gradient w.r.t. W
grad = kernel.gradient_log_likelihood_W(Y, Z, W, X, use_tf=False)
```

### 2. Separable Squared Exponential

**Use when:** Each dimension has its own lengthscale (D>1)

```python
# For D>1 with dimension-wise lengthscales
kernel = get_kernel_instance('separable_squared_exponential', theta=np.array([1.0, 1.5]), g=0.01, tau2=1.0, D=2)

# theta is a vector of shape (D,)
K = kernel.compute_covariance(Z1, Z2)
loglik = kernel.log_likelihood(Y, Z)
grad = kernel.gradient_log_likelihood_W(Y, Z, W, X, use_tf=False)
```

### 3. Isotropic Matérn-3/2

**Use when:** Need less smooth functions than squared exponential

```python
kernel = get_kernel_instance('isotropic_matern32', theta=1.0, g=0.01, tau2=1.0, D=1)
```

### 4. Separable Matérn-3/2

**Use when:** D>1 with dimension-wise Matérn-3/2 kernels

```python
kernel = get_kernel_instance('separable_matern32', theta=np.array([1.0, 1.5]), g=0.01, tau2=1.0, D=2)
```

## Kernel Factory Function

The recommended way to create kernels:

```python
from parameter_sampler_D1 import get_kernel_instance  # For D=1
from parameter_sampler_Dgeneral import get_kernel_instance  # For D>1

# Automatically selects correct kernel class
kernel = get_kernel_instance(
    kernel_type='separable_squared_exponential',
    theta=theta,  # Scalar for isotropic, vector for separable
    g=g,          # Nugget parameter
    tau2=tau2,    # Noise variance
    D=D           # Reduced dimension
)
```

## Features

### Covariance Computation

```python
# Between two sets of inputs
K = kernel.compute_covariance(Z1, Z2)
# Returns: (n1, n2) covariance matrix

# Self-covariance
K = kernel.compute_covariance(Z, Z)
# Returns: (n, n) covariance matrix
```

### Log-Likelihood

```python
loglik = kernel.log_likelihood(Y, Z)
# Y: (n,) response vector
# Z: (n, D) input matrix
# Returns: scalar log-likelihood
```

### Gradients with Respect to W

**NumPy Analytical Gradients:**
```python
grad = kernel.gradient_log_likelihood_W(
    Y=Y, Z=Z, W=W, X=X,
    use_tf=False  # Use analytical gradients
)
# Returns: (p, D) gradient matrix
```

**TensorFlow Automatic Differentiation:**
```python
grad = kernel.gradient_log_likelihood_W(
    Y=Y, Z=Z, W=W, X=X,
    use_tf=True  # Use TensorFlow gradients
)
# Returns: (p, D) gradient matrix
```

## Gradient Options

All kernels support two gradient computation methods:

1. **Analytical (NumPy):** Hand-derived derivatives
   - Set `use_tf=False`
   - Faster for small problems
   - No dependencies on TensorFlow
   - Recommended for D=1

2. **Automatic Differentiation (TensorFlow):**
   - Set `use_tf=True`
   - Accurate for complex cases
   - Requires TensorFlow
   - Recommended for D>1

## Kernel Selection Guide

| Kernel Type | When to Use | D | theta Shape |
|------------|-------------|---|-------------|
| `isotropic_squared_exponential` | All dimensions share lengthscale | D=1 or any | Scalar |
| `separable_squared_exponential` | Each dimension has own lengthscale | D>1 | Vector (D,) |
| `isotropic_matern32` | Less smooth, isotropic | D=1 or any | Scalar |
| `separable_matern32` | Less smooth, separable | D>1 | Vector (D,) |

## Integration with Framework

Kernels are automatically selected based on:
- `kernel_type` parameter in configuration
- `D` value (determines isotropic vs separable)
- Default: `isotropic_squared_exponential` for D=1, `separable_squared_exponential` for D>1

```python
from run_multichains import create_config_D2_L1

config = create_config_D2_L1(
    p=10,
    seed=42,
    kernel_type='separable_squared_exponential'  # Specify kernel
)
```

## See Also

- `../Parameter Sampler/` - Uses kernels for MCMC sampling
- `../Gibbs Sampling/` - Uses kernels in Gibbs samplers
- `../run_multichains.py` - Main interface with kernel selection
