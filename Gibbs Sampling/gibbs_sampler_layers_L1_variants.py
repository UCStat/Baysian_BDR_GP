"""
Gibbs Sampler Variants for Layer 1 - Simplified Versions

This module provides three variants of Layer 1 Gibbs samplers:
1. W_known: W is fixed/known, Z = XW, still sample tau2, g, theta
2. No_W: No dimensionality reduction, use X directly, sample tau2, g, theta
3. No_W_selective: Use selected columns of X, sample tau2, g, theta

All variants support:
- Kernel type selection
- Individual MLE options for tau2, g, theta
- Both D=1 and D>1 cases
"""

import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Union
import time

# Add parent directory to path for imports
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "Parameter Sampler"))
sys.path.insert(0, str(base_dir / "Covariance Functions"))

# Import parameter sampling functions
try:
    from parameter_sampler_D1 import (
        sample_tau2, estimate_tau2_MLE,
        sample_g, estimate_g_MLE,
        sample_theta_D, estimate_theta_D_MLE
    )
    from parameter_sampler_Dgeneral import (
        sample_tau2 as sample_tau2_Dgen, estimate_tau2_MLE as estimate_tau2_MLE_Dgen,
        sample_g as sample_g_Dgen, estimate_g_MLE as estimate_g_MLE_Dgen,
        sample_theta_D as sample_theta_D_Dgen, estimate_theta_D_MLE as estimate_theta_D_MLE_Dgen
    )
except ImportError:
    # Fallback if imports fail
    print("Warning: Could not import some parameter sampler functions")

# Import kernel functions
try:
    from covariance_kernel_functions_and_gradients_W import get_kernel_instance
except ImportError:
    print("Warning: Could not import kernel functions")


def _initial_value(value: Union[float, np.ndarray], D: int):
    arr = np.asarray(value, dtype=float)
    if D == 1:
        return float(arr.reshape(-1)[0])
    if arr.ndim == 0:
        return np.full(D, float(arr))
    if arr.size != D:
        raise ValueError(f"Initial value must be scalar or have {D} entries, got shape {arr.shape}")
    return arr.reshape(D).astype(float, copy=True)


def _copy_initial(value):
    if np.isscalar(value):
        return float(value)
    return np.array(value, dtype=float, copy=True)


class GibbsSampler1Layer_W_Known:
    """
    Gibbs sampler for 1-layer GP model with known/fixed W.
    
    Model:
        Y | X, W_fixed, θ, g, τ² ~ GP(0, τ²(C_y + g*I))
        where Z = XW_fixed and C_y is the covariance from Z
        W is fixed (not sampled), so we don't sample M, V, Lambda either
    
    Parameters to sample:
        - τ² (tau2): Observation noise variance
        - g: Nugget parameter
        - θ (theta_D): Lengthscale
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray, W_fixed: np.ndarray,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g: bool = False,
                 use_mle_theta: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 alpha1: float = 1.0, alpha2: float = 1000.0,
                 beta1: float = 0.01, beta2: float = 0.005,
                 gamma1: float = 1.5, gamma2: float = 3.9,
                 l: float = 1.0, u: float = 2.0,
                 tau2_init: float = 0.005,
                 g_init: float = 0.00009,
                 theta_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize 1-layer Gibbs sampler with known W.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p)
            W_fixed: Fixed projection matrix (p, D) - must be provided
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for τ² instead of MCMC (default: False)
            use_mle_g: Use MLE for g instead of MCMC (default: False)
            use_mle_theta: Use MLE for θ instead of MCMC (default: False)
            kernel_type: Kernel type ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 
                        'separable_matern32')
            alpha1, alpha2: Inverse Gamma prior parameters for tau2
            beta1, beta2: Gamma prior parameters for g
            gamma1, gamma2: Gamma prior parameters for theta
            l, u: Proposal parameters for Metropolis-Hastings
        """
        self.Y = Y.flatten()
        self.X = X
        self.n, self.p = X.shape
        self.W_fixed = W_fixed
        
        # Check W dimensions
        if W_fixed.shape[0] != self.p:
            raise ValueError(f"W_fixed must have {self.p} rows, got {W_fixed.shape[0]}")
        
        self.D = W_fixed.shape[1]
        
        # Compute Z = XW_fixed
        self.Z = self.X @ self.W_fixed  # (n, D)
        
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        
        # Kernel type
        self.kernel_type = kernel_type
        
        # MLE options (independent flags)
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g = use_mle_g
        self.use_mle_theta = use_mle_theta
        
        # Hyperparameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.l = l
        self.u = u
        self.tau2_init = float(tau2_init)
        self.g_init = float(g_init)
        self.theta_init = _initial_value(theta_init, self.D)
        
        # Storage
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.g_samples = np.zeros(self.n_saved)
        if self.D == 1:
            self.theta_D_samples = np.zeros(self.n_saved)
        else:
            self.theta_D_samples = np.zeros((self.n_saved, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameter starting values."""
        self.tau2 = self.tau2_init
        self.g = self.g_init
        self.theta_D = _copy_initial(self.theta_init)
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the Gibbs sampler."""
        save_idx = 0
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print(f"Running 1-Layer Gibbs Sampler with Known W (D={self.D})")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation: tau2={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g={'MLE' if self.use_mle_g else 'MCMC'}, "
                  f"theta={'MLE' if self.use_mle_theta else 'MCMC'}")
            print("-"*70)
        
        # Use identity matrix for W in sampling functions (since Z is already computed)
        W_identity = np.eye(self.D)
        
        for iter in range(self.n_iterations):
            # Sample or estimate hyperparameters (no W, M, V, Lambda sampling)
            
            # tau2
            if self.use_mle_tau2:
                if self.D == 1:
                    self.tau2 = estimate_tau2_MLE(
                        self.Y, self.Z, W_identity, self.theta_D, self.g,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = estimate_tau2_MLE_Dgen(
                        self.Y, self.Z, W_identity, self.theta_D, self.g,
                        kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.tau2 = sample_tau2(
                        self.Y, self.Z, W_identity, self.tau2,
                        self.theta_D, self.g, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = sample_tau2_Dgen(
                        self.Y, self.Z, W_identity, self.tau2,
                        self.theta_D, self.g, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
            
            # g
            if self.use_mle_g:
                if self.D == 1:
                    self.g = estimate_g_MLE(
                        self.Y, self.Z, W_identity, self.theta_D, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
                else:
                    self.g = estimate_g_MLE_Dgen(
                        self.Y, self.Z, W_identity, self.theta_D, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.g = sample_g(
                        self.Y, self.Z, W_identity, self.g,
                        self.theta_D, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                else:
                    self.g = sample_g_Dgen(
                        self.Y, self.Z, W_identity, self.g,
                        self.theta_D, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
            
            # theta_D
            if self.use_mle_theta:
                if self.D == 1:
                    self.theta_D = estimate_theta_D_MLE(
                        self.Y, self.Z, W_identity, self.g, self.tau2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.theta_D = estimate_theta_D_MLE_Dgen(
                        self.Y, self.Z, W_identity, self.g, self.tau2, self.D,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                # For separable kernels, sample dimension-wise
                if 'separable' in self.kernel_type and self.D > 1:
                    for m in range(self.D):
                        Z_m = self.Z[:, m].reshape(-1, 1)  # (n, 1)
                        W_identity_m = np.array([[1.0]])  # (1, 1)
                        
                        theta_m_new = sample_theta_D_Dgen(
                            self.Y, Z_m, W_identity_m,
                            np.array([self.theta_D[m]]),
                            self.tau2, self.g,
                            self.gamma1, self.gamma2,
                            self.l, self.u,
                            kernel_type=self.kernel_type
                        )
                        self.theta_D[m] = theta_m_new[0]
                else:
                    # For isotropic or D=1, sample full vector/scalar
                    if self.D == 1:
                        self.theta_D = sample_theta_D(
                            self.Y, self.Z, W_identity, self.theta_D,
                            self.tau2, self.g, self.gamma1, self.gamma2,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
                    else:
                        self.theta_D = sample_theta_D_Dgen(
                            self.Y, self.Z, W_identity, self.theta_D,
                            self.tau2, self.g, self.gamma1, self.gamma2,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.g_samples[save_idx] = self.g
                if self.D == 1:
                    self.theta_D_samples[save_idx] = self.theta_D
                else:
                    self.theta_D_samples[save_idx] = self.theta_D
                save_idx += 1
            
            if verbose and (iter + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iter+1}/{self.n_iterations} | tau2={self.tau2:.4f} | Time: {elapsed:.1f}s")
        
        if verbose:
            print("-"*70)
            print(f"Complete! Total time: {time.time() - start_time:.1f}s")
            print("="*70)
        
        return {
            'tau2_y': self.tau2_samples,
            'g_y': self.g_samples,
            'theta_D_y': self.theta_D_samples,
            'W_fixed': self.W_fixed,  # Return fixed W for reference
            'Z': self.Z  # Return computed Z for reference
        }


class GibbsSampler1Layer_No_W:
    """
    Gibbs sampler for 1-layer GP model without dimensionality reduction.
    
    Model:
        Y | X, θ, g, τ² ~ GP(0, τ²(C_y + g*I))
        where C_y is the covariance from X directly (no W, no Z)
    
    Parameters to sample:
        - τ² (tau2): Observation noise variance
        - g: Nugget parameter
        - θ (theta_D): Lengthscale
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g: bool = False,
                 use_mle_theta: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 alpha1: float = 1.0, alpha2: float = 1000.0,
                 beta1: float = 0.01, beta2: float = 0.005,
                 gamma1: float = 1.5, gamma2: float = 3.9,
                 l: float = 1.0, u: float = 2.0,
                 tau2_init: float = 0.005,
                 g_init: float = 0.00009,
                 theta_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize 1-layer Gibbs sampler without W.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p) - used directly (no projection)
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for τ² instead of MCMC (default: False)
            use_mle_g: Use MLE for g instead of MCMC (default: False)
            use_mle_theta: Use MLE for θ instead of MCMC (default: False)
            kernel_type: Kernel type ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 
                        'separable_matern32')
            alpha1, alpha2: Inverse Gamma prior parameters for tau2
            beta1, beta2: Gamma prior parameters for g
            gamma1, gamma2: Gamma prior parameters for theta
            l, u: Proposal parameters for Metropolis-Hastings
        """
        self.Y = Y.flatten()
        self.X = X
        self.n, self.p = X.shape
        
        # D is the number of dimensions in X (p)
        self.D = self.p
        
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        
        # Kernel type
        self.kernel_type = kernel_type
        
        # MLE options (independent flags)
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g = use_mle_g
        self.use_mle_theta = use_mle_theta
        
        # Hyperparameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.l = l
        self.u = u
        self.tau2_init = float(tau2_init)
        self.g_init = float(g_init)
        self.theta_init = _initial_value(theta_init, self.D)
        
        # Storage
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.g_samples = np.zeros(self.n_saved)
        if self.D == 1:
            self.theta_D_samples = np.zeros(self.n_saved)
        else:
            self.theta_D_samples = np.zeros((self.n_saved, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameter starting values."""
        self.tau2 = self.tau2_init
        self.g = self.g_init
        self.theta_D = _copy_initial(self.theta_init)
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the Gibbs sampler."""
        save_idx = 0
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print(f"Running 1-Layer Gibbs Sampler without W (D={self.D}, using all {self.p} columns)")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation: tau2={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g={'MLE' if self.use_mle_g else 'MCMC'}, "
                  f"theta={'MLE' if self.use_mle_theta else 'MCMC'}")
            print("-"*70)
        
        # Use identity matrix for W (X is used directly)
        W_identity = np.eye(self.D)
        
        for iter in range(self.n_iterations):
            # Sample or estimate hyperparameters (no W, M, V, Lambda sampling)
            
            # tau2
            if self.use_mle_tau2:
                if self.D == 1:
                    self.tau2 = estimate_tau2_MLE(
                        self.Y, self.X, W_identity, self.theta_D, self.g,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = estimate_tau2_MLE_Dgen(
                        self.Y, self.X, W_identity, self.theta_D, self.g,
                        kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.tau2 = sample_tau2(
                        self.Y, self.X, W_identity, self.tau2,
                        self.theta_D, self.g, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = sample_tau2_Dgen(
                        self.Y, self.X, W_identity, self.tau2,
                        self.theta_D, self.g, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
            
            # g
            if self.use_mle_g:
                if self.D == 1:
                    self.g = estimate_g_MLE(
                        self.Y, self.X, W_identity, self.theta_D, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
                else:
                    self.g = estimate_g_MLE_Dgen(
                        self.Y, self.X, W_identity, self.theta_D, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.g = sample_g(
                        self.Y, self.X, W_identity, self.g,
                        self.theta_D, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                else:
                    self.g = sample_g_Dgen(
                        self.Y, self.X, W_identity, self.g,
                        self.theta_D, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
            
            # theta_D
            if self.use_mle_theta:
                if self.D == 1:
                    self.theta_D = estimate_theta_D_MLE(
                        self.Y, self.X, W_identity, self.g, self.tau2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.theta_D = estimate_theta_D_MLE_Dgen(
                        self.Y, self.X, W_identity, self.g, self.tau2, self.D,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                # For separable kernels, sample dimension-wise
                if 'separable' in self.kernel_type and self.D > 1:
                    for m in range(self.D):
                        X_m = self.X[:, m].reshape(-1, 1)  # (n, 1)
                        W_identity_m = np.array([[1.0]])  # (1, 1)
                        
                        theta_m_new = sample_theta_D_Dgen(
                            self.Y, X_m, W_identity_m,
                            np.array([self.theta_D[m]]),
                            self.tau2, self.g,
                            self.gamma1, self.gamma2,
                            self.l, self.u,
                            kernel_type=self.kernel_type
                        )
                        self.theta_D[m] = theta_m_new[0]
                else:
                    # For isotropic or D=1, sample full vector/scalar
                    if self.D == 1:
                        self.theta_D = sample_theta_D(
                            self.Y, self.X, W_identity, self.theta_D,
                            self.tau2, self.g, self.gamma1, self.gamma2,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
                    else:
                        self.theta_D = sample_theta_D_Dgen(
                            self.Y, self.X, W_identity, self.theta_D,
                            self.tau2, self.g, self.gamma1, self.gamma2,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.g_samples[save_idx] = self.g
                if self.D == 1:
                    self.theta_D_samples[save_idx] = self.theta_D
                else:
                    self.theta_D_samples[save_idx] = self.theta_D
                save_idx += 1
            
            if verbose and (iter + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iter+1}/{self.n_iterations} | tau2={self.tau2:.4f} | Time: {elapsed:.1f}s")
        
        if verbose:
            print("-"*70)
            print(f"Complete! Total time: {time.time() - start_time:.1f}s")
            print("="*70)
        
        return {
            'tau2_y': self.tau2_samples,
            'g_y': self.g_samples,
            'theta_D_y': self.theta_D_samples
        }


class GibbsSampler1Layer_No_W_Selective:
    """
    Gibbs sampler for 1-layer GP model without dimensionality reduction, 
    using selected columns of X.
    
    Model:
        Y | X_selected, θ, g, τ² ~ GP(0, τ²(C_y + g*I))
        where C_y is the covariance from selected columns of X
    
    Parameters to sample:
        - τ² (tau2): Observation noise variance
        - g: Nugget parameter
        - θ (theta_D): Lengthscale
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray, D: int,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g: bool = False,
                 use_mle_theta: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 alpha1: float = 1.0, alpha2: float = 1000.0,
                 beta1: float = 0.01, beta2: float = 0.005,
                 gamma1: float = 1.5, gamma2: float = 3.9,
                 l: float = 1.0, u: float = 2.0,
                 column_indices: Optional[np.ndarray] = None,
                 tau2_init: float = 0.005,
                 g_init: float = 0.00009,
                 theta_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize 1-layer Gibbs sampler without W, using selected columns.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p)
            D: Number of columns to use from X (must be <= p)
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for τ² instead of MCMC (default: False)
            use_mle_g: Use MLE for g instead of MCMC (default: False)
            use_mle_theta: Use MLE for θ instead of MCMC (default: False)
            kernel_type: Kernel type ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 
                        'separable_matern32')
            alpha1, alpha2: Inverse Gamma prior parameters for tau2
            beta1, beta2: Gamma prior parameters for g
            gamma1, gamma2: Gamma prior parameters for theta
            l, u: Proposal parameters for Metropolis-Hastings
            column_indices: Optional array of column indices to use. 
                          If None, uses first D columns. Shape: (D,)
        """
        self.Y = Y.flatten()
        self.X = X
        self.n, self.p = X.shape
        
        if D > self.p:
            raise ValueError(f"D ({D}) cannot be greater than p ({self.p})")
        
        self.D = D
        
        # Select columns
        if column_indices is None:
            # Use first D columns
            self.column_indices = np.arange(D)
        else:
            if len(column_indices) != D:
                raise ValueError(f"column_indices must have length {D}, got {len(column_indices)}")
            if np.any(column_indices < 0) or np.any(column_indices >= self.p):
                raise ValueError(f"column_indices must be in [0, {self.p-1}]")
            self.column_indices = np.array(column_indices)
        
        # Extract selected columns
        self.X_selected = self.X[:, self.column_indices]  # (n, D)
        
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        
        # Kernel type
        self.kernel_type = kernel_type
        
        # MLE options (independent flags)
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g = use_mle_g
        self.use_mle_theta = use_mle_theta
        
        # Hyperparameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.l = l
        self.u = u
        self.tau2_init = float(tau2_init)
        self.g_init = float(g_init)
        self.theta_init = _initial_value(theta_init, self.D)
        
        # Storage
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.g_samples = np.zeros(self.n_saved)
        if self.D == 1:
            self.theta_D_samples = np.zeros(self.n_saved)
        else:
            self.theta_D_samples = np.zeros((self.n_saved, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameter starting values."""
        self.tau2 = self.tau2_init
        self.g = self.g_init
        self.theta_D = _copy_initial(self.theta_init)
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the Gibbs sampler."""
        save_idx = 0
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print(f"Running 1-Layer Gibbs Sampler without W (D={self.D}, using columns {self.column_indices})")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation: tau2={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g={'MLE' if self.use_mle_g else 'MCMC'}, "
                  f"theta={'MLE' if self.use_mle_theta else 'MCMC'}")
            print("-"*70)
        
        # Use identity matrix for W (X_selected is used directly)
        W_identity = np.eye(self.D)
        
        for iter in range(self.n_iterations):
            # Sample or estimate hyperparameters (no W, M, V, Lambda sampling)
            
            # tau2
            if self.use_mle_tau2:
                if self.D == 1:
                    self.tau2 = estimate_tau2_MLE(
                        self.Y, self.X_selected, W_identity, self.theta_D, self.g,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = estimate_tau2_MLE_Dgen(
                        self.Y, self.X_selected, W_identity, self.theta_D, self.g,
                        kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.tau2 = sample_tau2(
                        self.Y, self.X_selected, W_identity, self.tau2,
                        self.theta_D, self.g, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = sample_tau2_Dgen(
                        self.Y, self.X_selected, W_identity, self.tau2,
                        self.theta_D, self.g, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
            
            # g
            if self.use_mle_g:
                if self.D == 1:
                    self.g = estimate_g_MLE(
                        self.Y, self.X_selected, W_identity, self.theta_D, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
                else:
                    self.g = estimate_g_MLE_Dgen(
                        self.Y, self.X_selected, W_identity, self.theta_D, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.g = sample_g(
                        self.Y, self.X_selected, W_identity, self.g,
                        self.theta_D, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                else:
                    self.g = sample_g_Dgen(
                        self.Y, self.X_selected, W_identity, self.g,
                        self.theta_D, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
            
            # theta_D
            if self.use_mle_theta:
                if self.D == 1:
                    self.theta_D = estimate_theta_D_MLE(
                        self.Y, self.X_selected, W_identity, self.g, self.tau2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.theta_D = estimate_theta_D_MLE_Dgen(
                        self.Y, self.X_selected, W_identity, self.g, self.tau2, self.D,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                # For separable kernels, sample dimension-wise
                if 'separable' in self.kernel_type and self.D > 1:
                    for m in range(self.D):
                        X_m = self.X_selected[:, m].reshape(-1, 1)  # (n, 1)
                        W_identity_m = np.array([[1.0]])  # (1, 1)
                        
                        theta_m_new = sample_theta_D_Dgen(
                            self.Y, X_m, W_identity_m,
                            np.array([self.theta_D[m]]),
                            self.tau2, self.g,
                            self.gamma1, self.gamma2,
                            self.l, self.u,
                            kernel_type=self.kernel_type
                        )
                        self.theta_D[m] = theta_m_new[0]
                else:
                    # For isotropic or D=1, sample full vector/scalar
                    if self.D == 1:
                        self.theta_D = sample_theta_D(
                            self.Y, self.X_selected, W_identity, self.theta_D,
                            self.tau2, self.g, self.gamma1, self.gamma2,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
                    else:
                        self.theta_D = sample_theta_D_Dgen(
                            self.Y, self.X_selected, W_identity, self.theta_D,
                            self.tau2, self.g, self.gamma1, self.gamma2,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.g_samples[save_idx] = self.g
                if self.D == 1:
                    self.theta_D_samples[save_idx] = self.theta_D
                else:
                    self.theta_D_samples[save_idx] = self.theta_D
                save_idx += 1
            
            if verbose and (iter + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iter+1}/{self.n_iterations} | tau2={self.tau2:.4f} | Time: {elapsed:.1f}s")
        
        if verbose:
            print("-"*70)
            print(f"Complete! Total time: {time.time() - start_time:.1f}s")
            print("="*70)
        
        return {
            'tau2_y': self.tau2_samples,
            'g_y': self.g_samples,
            'theta_D_y': self.theta_D_samples,
            'column_indices': self.column_indices,  # Return which columns were used
            'X_selected': self.X_selected  # Return selected X for reference
        }


if __name__ == "__main__":
    print("="*70)
    print("Gibbs Sampler Variants for Layer 1 - Test")
    print("="*70)
    
    np.random.seed(42)
    n, p = 20, 5
    
    X = np.random.randn(n, p)
    Y = np.random.randn(n)
    
    print("\n1. Testing W_Known variant:")
    W_fixed = np.random.randn(p, 2)
    W_fixed, _ = np.linalg.qr(W_fixed)
    
    sampler1 = GibbsSampler1Layer_W_Known(
        Y=Y, X=X, W_fixed=W_fixed,
        n_iterations=2, burn_in=0, thin=1,
        kernel_type='separable_squared_exponential'
    )
    samples1 = sampler1.run(verbose=False)
    print(f"   ✅ W_Known: {len(samples1['tau2_y'])} samples")
    
    print("\n2. Testing No_W variant:")
    sampler2 = GibbsSampler1Layer_No_W(
        Y=Y, X=X,
        n_iterations=2, burn_in=0, thin=1,
        kernel_type='separable_squared_exponential'
    )
    samples2 = sampler2.run(verbose=False)
    print(f"   ✅ No_W: {len(samples2['tau2_y'])} samples")
    
    print("\n3. Testing No_W_Selective variant:")
    sampler3 = GibbsSampler1Layer_No_W_Selective(
        Y=Y, X=X, D=3,
        n_iterations=2, burn_in=0, thin=1,
        kernel_type='separable_squared_exponential'
    )
    samples3 = sampler3.run(verbose=False)
    print(f"   ✅ No_W_Selective: {len(samples3['tau2_y'])} samples")
    
    print("\n" + "="*70)
    print("✅ All variants tested successfully!")
    print("="*70)
