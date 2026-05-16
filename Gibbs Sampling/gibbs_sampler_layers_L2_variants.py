"""
Gibbs Sampler Variants for Layer 2 - Simplified Versions

This module provides three variants of Layer 2 Gibbs samplers:
1. W_known: W is fixed/known, Z = XW, still sample Q, g_q, tau2_q, theta_q, tau2_y, g_y, theta_y
2. No_W: No dimensionality reduction, use X directly, sample Q, g_q, tau2_q, theta_q, tau2_y, g_y, theta_y
3. No_W_selective: Use selected columns of X, sample Q, g_q, tau2_q, theta_q, tau2_y, g_y, theta_y

All variants support:
- Kernel type selection
- Individual MLE options for tau2_y, g_y, theta_y (Q layer always uses MCMC)
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
        sample_theta_D, estimate_theta_D_MLE,
        sample_Q_2layer_ESS
    )
    from parameter_sampler_Dgeneral import (
        sample_tau2 as sample_tau2_Dgen, estimate_tau2_MLE as estimate_tau2_MLE_Dgen,
        sample_g as sample_g_Dgen, estimate_g_MLE as estimate_g_MLE_Dgen,
        sample_theta_D as sample_theta_D_Dgen, estimate_theta_D_MLE as estimate_theta_D_MLE_Dgen,
        sample_Q_2layer_ESS as sample_Q_2layer_ESS_Dgen
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


class GibbsSampler2Layer_W_Known:
    """
    Gibbs sampler for 2-layer Deep GP model with known/fixed W.
    
    Model:
        Y | Q, θ_y, g_y, τ² ~ GP(0, τ²(C_y + g_y*I))
        Q | Z, θ_q, g_q ~ GP(0, C_q)
        where Z = XW_fixed (W is fixed, not sampled)
    
    Parameters to sample:
        - τ²_y (tau2): Observation noise variance
        - g_y: Nugget parameter for Y layer
        - θ_y (theta_y): Lengthscale for Y layer
        - Q: Latent layer (n, D)
        - g_q: Nugget parameter for Q layer (held fixed at configured value)
        - tau2_q: Variance for Q layer (held fixed at configured value)
        - θ_q (theta_q): Lengthscale for Q layer (per dimension)
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray, W_fixed: np.ndarray,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g_y: bool = False,
                 use_mle_theta_y: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 alpha1: float = 1.0, alpha2: float = 1000.0,
                 beta1: float = 0.01, beta2: float = 0.005,
                 gamma1: float = 1.5, gamma2_y: float = 3.9, gamma2_q: float = 3.9/3,
                 l: float = 1.0, u: float = 2.0,
                 tau2_y_init: float = 0.005,
                 tau2_q_init: Union[float, np.ndarray] = 0.005,
                 g_y_init: float = 0.00009,
                 g_q_init: Union[float, np.ndarray] = 0.00009,
                 theta_y_init: Union[float, np.ndarray] = 1.0,
                 theta_q_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize 2-layer Gibbs sampler with known W.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p)
            W_fixed: Fixed projection matrix (p, D) - must be provided
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for τ²_y instead of MCMC (default: False)
            use_mle_g_y: Use MLE for g_y instead of MCMC (default: False)
            use_mle_theta_y: Use MLE for θ_y instead of MCMC (default: False)
            kernel_type: Kernel type ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 
                        'separable_matern32')
            alpha1, alpha2: Inverse Gamma prior parameters for tau2_y
            beta1, beta2: Gamma prior parameters for g_y
            gamma1: Gamma shape parameter (3/2)
            gamma2_y: Gamma rate parameter for theta_y (3.9)
            gamma2_q: Gamma rate parameter for theta_q (3.9/3)
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
        
        # MLE options (independent flags, only for Y layer)
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g_y = use_mle_g_y
        self.use_mle_theta_y = use_mle_theta_y
        
        # Hyperparameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2_y = gamma2_y
        self.gamma2_q = gamma2_q
        self.l = l
        self.u = u
        self.tau2_y_init = float(tau2_y_init)
        self.tau2_q_fixed = _initial_value(tau2_q_init, self.D)
        self.g_y_init = float(g_y_init)
        self.g_q_fixed = _initial_value(g_q_init, self.D)
        self.theta_y_init = _initial_value(theta_y_init, self.D)
        self.theta_q_init = _initial_value(theta_q_init, self.D)
        
        # Storage
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.g_y_samples = np.zeros(self.n_saved)
        if self.D == 1:
            self.theta_y_samples = np.zeros(self.n_saved)
            self.theta_q_samples = np.zeros(self.n_saved)
            self.g_q_samples = np.zeros(self.n_saved)
            self.tau2_q_samples = np.zeros(self.n_saved)
            self.Q_samples = np.zeros((self.n_saved, self.n))
        else:
            self.theta_y_samples = np.zeros((self.n_saved, self.D))
            self.theta_q_samples = np.zeros((self.n_saved, self.D))
            self.g_q_samples = np.zeros((self.n_saved, self.D))
            self.tau2_q_samples = np.zeros((self.n_saved, self.D))
            self.Q_samples = np.zeros((self.n_saved, self.n, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameter starting values."""
        # Initialize Q as projection of X
        if self.D == 1:
            self.Q = (self.X @ self.W_fixed).flatten()
        else:
            self.Q = self.X @ self.W_fixed  # (n, D)
        
        # Y layer hyperparameters
        self.tau2 = self.tau2_y_init
        self.g_y = self.g_y_init
        self.theta_y = _copy_initial(self.theta_y_init)
        
        # Q layer hyperparameters (per dimension)
        self.g_q = _copy_initial(self.g_q_fixed)
        self.tau2_q = _copy_initial(self.tau2_q_fixed)
        self.theta_q = _copy_initial(self.theta_q_init)
    
    def _sample_Q(self):
        """Sample latent layer Q using Elliptical Slice Sampling."""
        if self.D == 1:
            self.Q = sample_Q_2layer_ESS(
                Y=self.Y,
                Q_current=self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                Z=self.Z.reshape(-1, 1) if self.Z.ndim == 1 else self.Z,
                g_y=self.g_y,
                theta_y=self.theta_y,
                theta_q=self.theta_q,
                g_q=self.g_q,
                tau2_y=self.tau2,
                tau2_q=self.tau2_q,
                kernel_type=self.kernel_type
            )
            if self.Q.ndim == 2:
                self.Q = self.Q.flatten()
        else:
            self.Q = sample_Q_2layer_ESS_Dgen(
                Y=self.Y,
                Q_current=self.Q,
                Z=self.Z,
                g_y=self.g_y,
                theta_y=self.theta_y,
                theta_q=self.theta_q,
                g_q=self.g_q,
                tau2_y=self.tau2,
                tau2_q=self.tau2_q,
                kernel_type=self.kernel_type
            )
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the Gibbs sampler."""
        save_idx = 0
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print(f"Running 2-Layer Deep GP Gibbs Sampler with Known W (D={self.D})")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation (Y layer): tau2={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g_y={'MLE' if self.use_mle_g_y else 'MCMC'}, "
                  f"theta_y={'MLE' if self.use_mle_theta_y else 'MCMC'}")
            print("-"*70)
        
        # Use identity matrix for W in sampling functions (since Z is already computed)
        W_identity = np.eye(self.D)
        
        for iter in range(self.n_iterations):
            # Sample hyperparameters for Q layer (always MCMC, dimension-wise for separable)
            # The variant keeps latent nugget and variance fixed at configured values.
            # Only theta_q is sampled per dimension
            for m in range(self.D):
                # Fixed hyperparameters for latent layers
                if self.D == 1:
                    self.g_q = float(self.g_q_fixed)
                    self.tau2_q = float(self.tau2_q_fixed)
                else:
                    self.g_q[m] = self.g_q_fixed[m]
                    self.tau2_q[m] = self.tau2_q_fixed[m]
                
                # Sample theta_q[m] for dimension m
                if self.D == 1:
                    Q_m = self.Q
                    Z_m = self.Z.reshape(-1, 1) if self.Z.ndim == 1 else self.Z
                    W_identity_m = np.array([[1.0]])
                    
                    theta_m_new = sample_theta_D(
                        Q_m, Z_m, W_identity_m, np.array([self.theta_q]),
                        self.tau2_q, self.g_q, self.gamma1, self.gamma2_q,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                    self.theta_q = theta_m_new[0] if hasattr(theta_m_new, '__len__') else theta_m_new
                else:
                    Q_m = self.Q[:, m]  # (n,)
                    Z_m = self.Z[:, m].reshape(-1, 1)  # (n, 1)
                    W_identity_m = np.array([[1.0]])  # (1, 1)
                    
                    theta_m_new = sample_theta_D_Dgen(
                        Q_m, Z_m, W_identity_m, np.array([self.theta_q[m]]),
                        self.tau2_q[m], self.g_q[m], self.gamma1, self.gamma2_q,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                    self.theta_q[m] = theta_m_new[0]
            
            # Sample Q using ESS
            self._sample_Q()
            
            # Sample/estimate hyperparameters for Y layer (with individual MLE options)
            # tau2_y
            if self.use_mle_tau2:
                if self.D == 1:
                    self.tau2 = estimate_tau2_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.theta_y, self.g_y,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = estimate_tau2_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.theta_y, self.g_y,
                        kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.tau2 = sample_tau2(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.tau2,
                        self.theta_y, self.g_y, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = sample_tau2_Dgen(
                        self.Y, self.Q, W_identity, self.tau2,
                        self.theta_y, self.g_y, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
            
            # g_y
            if self.use_mle_g_y:
                if self.D == 1:
                    self.g_y = estimate_g_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.theta_y, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
                else:
                    self.g_y = estimate_g_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.theta_y, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.g_y = sample_g(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.g_y,
                        self.theta_y, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                else:
                    self.g_y = sample_g_Dgen(
                        self.Y, self.Q, W_identity, self.g_y,
                        self.theta_y, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
            
            # theta_y
            if self.use_mle_theta_y:
                if self.D == 1:
                    self.theta_y = estimate_theta_D_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.g_y, self.tau2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.theta_y = estimate_theta_D_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.g_y, self.tau2, self.D,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                # For separable kernels, sample dimension-wise
                if 'separable' in self.kernel_type and self.D > 1:
                    for m in range(self.D):
                        Q_m = self.Q[:, m].reshape(-1, 1)  # (n, 1)
                        W_identity_m = np.array([[1.0]])  # (1, 1)
                        
                        theta_m_new = sample_theta_D_Dgen(
                            self.Y, Q_m, W_identity_m,
                            np.array([self.theta_y[m]]),
                            self.tau2, self.g_y,
                            self.gamma1, self.gamma2_y,
                            self.l, self.u,
                            kernel_type=self.kernel_type
                        )
                        self.theta_y[m] = theta_m_new[0]
                else:
                    # For isotropic or D=1, sample full vector/scalar
                    if self.D == 1:
                        self.theta_y = sample_theta_D(
                            self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                            W_identity, self.theta_y,
                            self.tau2, self.g_y, self.gamma1, self.gamma2_y,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
                    else:
                        self.theta_y = sample_theta_D_Dgen(
                            self.Y, self.Q, W_identity, self.theta_y,
                            self.tau2, self.g_y, self.gamma1, self.gamma2_y,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.g_y_samples[save_idx] = self.g_y
                if self.D == 1:
                    self.theta_y_samples[save_idx] = self.theta_y
                    self.theta_q_samples[save_idx] = self.theta_q
                    self.g_q_samples[save_idx] = self.g_q
                    self.tau2_q_samples[save_idx] = self.tau2_q
                    self.Q_samples[save_idx] = self.Q
                else:
                    self.theta_y_samples[save_idx] = self.theta_y
                    self.theta_q_samples[save_idx] = self.theta_q
                    self.g_q_samples[save_idx] = self.g_q
                    self.tau2_q_samples[save_idx] = self.tau2_q
                    self.Q_samples[save_idx] = self.Q
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
            'g_y': self.g_y_samples,
            'theta_y': self.theta_y_samples,
            'g_q': self.g_q_samples,
            'tau2_q': self.tau2_q_samples,
            'theta_q': self.theta_q_samples,
            'Q': self.Q_samples,
            'W_fixed': self.W_fixed,  # Return fixed W for reference
            'Z': self.Z  # Return computed Z for reference
        }


class GibbsSampler2Layer_No_W:
    """
    Gibbs sampler for 2-layer Deep GP model without dimensionality reduction.
    
    Model:
        Y | Q, θ_y, g_y, τ² ~ GP(0, τ²(C_y + g_y*I))
        Q | X, θ_q, g_q ~ GP(0, C_q)
        where X is used directly (no W, no Z)
    
    Parameters to sample:
        - τ²_y (tau2): Observation noise variance
        - g_y: Nugget parameter for Y layer
        - θ_y (theta_y): Lengthscale for Y layer
        - Q: Latent layer (n, D)
        - g_q: Nugget parameter for Q layer (held fixed at configured value)
        - tau2_q: Variance for Q layer (held fixed at configured value)
        - θ_q (theta_q): Lengthscale for Q layer (per dimension)
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g_y: bool = False,
                 use_mle_theta_y: bool = False,
                 kernel_type: str = 'separable_squared_exponential',
                 alpha1: float = 1.0, alpha2: float = 1000.0,
                 beta1: float = 0.01, beta2: float = 0.005,
                 gamma1: float = 1.5, gamma2_y: float = 3.9, gamma2_q: float = 3.9/3,
                 l: float = 1.0, u: float = 2.0,
                 tau2_y_init: float = 0.005,
                 tau2_q_init: Union[float, np.ndarray] = 0.005,
                 g_y_init: float = 0.00009,
                 g_q_init: Union[float, np.ndarray] = 0.00009,
                 theta_y_init: Union[float, np.ndarray] = 1.0,
                 theta_q_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize 2-layer Gibbs sampler without W.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p) - used directly (no projection)
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for τ²_y instead of MCMC (default: False)
            use_mle_g_y: Use MLE for g_y instead of MCMC (default: False)
            use_mle_theta_y: Use MLE for θ_y instead of MCMC (default: False)
            kernel_type: Kernel type ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 
                        'separable_matern32')
            alpha1, alpha2: Inverse Gamma prior parameters for tau2_y
            beta1, beta2: Gamma prior parameters for g_y
            gamma1: Gamma shape parameter (3/2)
            gamma2_y: Gamma rate parameter for theta_y (3.9)
            gamma2_q: Gamma rate parameter for theta_q (3.9/3)
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
        
        # MLE options (independent flags, only for Y layer)
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g_y = use_mle_g_y
        self.use_mle_theta_y = use_mle_theta_y
        
        # Hyperparameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2_y = gamma2_y
        self.gamma2_q = gamma2_q
        self.l = l
        self.u = u
        self.tau2_y_init = float(tau2_y_init)
        self.tau2_q_fixed = _initial_value(tau2_q_init, self.D)
        self.g_y_init = float(g_y_init)
        self.g_q_fixed = _initial_value(g_q_init, self.D)
        self.theta_y_init = _initial_value(theta_y_init, self.D)
        self.theta_q_init = _initial_value(theta_q_init, self.D)
        
        # Storage
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.g_y_samples = np.zeros(self.n_saved)
        if self.D == 1:
            self.theta_y_samples = np.zeros(self.n_saved)
            self.theta_q_samples = np.zeros(self.n_saved)
            self.g_q_samples = np.zeros(self.n_saved)
            self.tau2_q_samples = np.zeros(self.n_saved)
            self.Q_samples = np.zeros((self.n_saved, self.n))
        else:
            self.theta_y_samples = np.zeros((self.n_saved, self.D))
            self.theta_q_samples = np.zeros((self.n_saved, self.D))
            self.g_q_samples = np.zeros((self.n_saved, self.D))
            self.tau2_q_samples = np.zeros((self.n_saved, self.D))
            self.Q_samples = np.zeros((self.n_saved, self.n, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameter starting values."""
        # Initialize Q as copy of X (or first column if D=1)
        if self.D == 1:
            self.Q = self.X[:, 0].copy()
        else:
            self.Q = self.X.copy()
        
        # Y layer hyperparameters
        self.tau2 = self.tau2_y_init
        self.g_y = self.g_y_init
        self.theta_y = _copy_initial(self.theta_y_init)
        
        # Q layer hyperparameters (per dimension)
        self.g_q = _copy_initial(self.g_q_fixed)
        self.tau2_q = _copy_initial(self.tau2_q_fixed)
        self.theta_q = _copy_initial(self.theta_q_init)
    
    def _sample_Q(self):
        """Sample latent layer Q using Elliptical Slice Sampling."""
        if self.D == 1:
            self.Q = sample_Q_2layer_ESS(
                Y=self.Y,
                Q_current=self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                Z=self.X.reshape(-1, 1) if self.X.ndim == 1 else self.X,
                g_y=self.g_y,
                theta_y=self.theta_y,
                theta_q=self.theta_q,
                g_q=self.g_q,
                tau2_y=self.tau2,
                tau2_q=self.tau2_q,
                kernel_type=self.kernel_type
            )
            if self.Q.ndim == 2:
                self.Q = self.Q.flatten()
        else:
            self.Q = sample_Q_2layer_ESS_Dgen(
                Y=self.Y,
                Q_current=self.Q,
                Z=self.X,  # Use X directly
                g_y=self.g_y,
                theta_y=self.theta_y,
                theta_q=self.theta_q,
                g_q=self.g_q,
                tau2_y=self.tau2,
                tau2_q=self.tau2_q,
                kernel_type=self.kernel_type
            )
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the Gibbs sampler."""
        save_idx = 0
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print(f"Running 2-Layer Deep GP Gibbs Sampler without W (D={self.D}, using all {self.p} columns)")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation (Y layer): tau2={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g_y={'MLE' if self.use_mle_g_y else 'MCMC'}, "
                  f"theta_y={'MLE' if self.use_mle_theta_y else 'MCMC'}")
            print("-"*70)
        
        # Use identity matrix for W (X is used directly)
        W_identity = np.eye(self.D)
        
        for iter in range(self.n_iterations):
            # Sample hyperparameters for Q layer (always MCMC, dimension-wise for separable)
            # The variant keeps latent nugget and variance fixed at configured values.
            # Only theta_q is sampled per dimension
            for m in range(self.D):
                # Fixed hyperparameters for latent layers
                if self.D == 1:
                    self.g_q = float(self.g_q_fixed)
                    self.tau2_q = float(self.tau2_q_fixed)
                else:
                    self.g_q[m] = self.g_q_fixed[m]
                    self.tau2_q[m] = self.tau2_q_fixed[m]
                
                # Sample theta_q[m] for dimension m
                if self.D == 1:
                    Q_m = self.Q
                    X_m = self.X.reshape(-1, 1) if self.X.ndim == 1 else self.X
                    W_identity_m = np.array([[1.0]])
                    
                    theta_m_new = sample_theta_D(
                        Q_m, X_m, W_identity_m, np.array([self.theta_q]),
                        self.tau2_q, self.g_q, self.gamma1, self.gamma2_q,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                    self.theta_q = theta_m_new[0] if hasattr(theta_m_new, '__len__') else theta_m_new
                else:
                    Q_m = self.Q[:, m]  # (n,)
                    X_m = self.X[:, m].reshape(-1, 1)  # (n, 1)
                    W_identity_m = np.array([[1.0]])  # (1, 1)
                    
                    theta_m_new = sample_theta_D_Dgen(
                        Q_m, X_m, W_identity_m, np.array([self.theta_q[m]]),
                        self.tau2_q[m], self.g_q[m], self.gamma1, self.gamma2_q,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                    self.theta_q[m] = theta_m_new[0]
            
            # Sample Q using ESS
            self._sample_Q()
            
            # Sample/estimate hyperparameters for Y layer (with individual MLE options)
            # tau2_y
            if self.use_mle_tau2:
                if self.D == 1:
                    self.tau2 = estimate_tau2_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.theta_y, self.g_y,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = estimate_tau2_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.theta_y, self.g_y,
                        kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.tau2 = sample_tau2(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.tau2,
                        self.theta_y, self.g_y, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = sample_tau2_Dgen(
                        self.Y, self.Q, W_identity, self.tau2,
                        self.theta_y, self.g_y, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
            
            # g_y
            if self.use_mle_g_y:
                if self.D == 1:
                    self.g_y = estimate_g_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.theta_y, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
                else:
                    self.g_y = estimate_g_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.theta_y, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.g_y = sample_g(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.g_y,
                        self.theta_y, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                else:
                    self.g_y = sample_g_Dgen(
                        self.Y, self.Q, W_identity, self.g_y,
                        self.theta_y, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
            
            # theta_y
            if self.use_mle_theta_y:
                if self.D == 1:
                    self.theta_y = estimate_theta_D_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.g_y, self.tau2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.theta_y = estimate_theta_D_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.g_y, self.tau2, self.D,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                # For separable kernels, sample dimension-wise
                if 'separable' in self.kernel_type and self.D > 1:
                    for m in range(self.D):
                        Q_m = self.Q[:, m].reshape(-1, 1)  # (n, 1)
                        W_identity_m = np.array([[1.0]])  # (1, 1)
                        
                        theta_m_new = sample_theta_D_Dgen(
                            self.Y, Q_m, W_identity_m,
                            np.array([self.theta_y[m]]),
                            self.tau2, self.g_y,
                            self.gamma1, self.gamma2_y,
                            self.l, self.u,
                            kernel_type=self.kernel_type
                        )
                        self.theta_y[m] = theta_m_new[0]
                else:
                    # For isotropic or D=1, sample full vector/scalar
                    if self.D == 1:
                        self.theta_y = sample_theta_D(
                            self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                            W_identity, self.theta_y,
                            self.tau2, self.g_y, self.gamma1, self.gamma2_y,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
                    else:
                        self.theta_y = sample_theta_D_Dgen(
                            self.Y, self.Q, W_identity, self.theta_y,
                            self.tau2, self.g_y, self.gamma1, self.gamma2_y,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.g_y_samples[save_idx] = self.g_y
                if self.D == 1:
                    self.theta_y_samples[save_idx] = self.theta_y
                    self.theta_q_samples[save_idx] = self.theta_q
                    self.g_q_samples[save_idx] = self.g_q
                    self.tau2_q_samples[save_idx] = self.tau2_q
                    self.Q_samples[save_idx] = self.Q
                else:
                    self.theta_y_samples[save_idx] = self.theta_y
                    self.theta_q_samples[save_idx] = self.theta_q
                    self.g_q_samples[save_idx] = self.g_q
                    self.tau2_q_samples[save_idx] = self.tau2_q
                    self.Q_samples[save_idx] = self.Q
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
            'g_y': self.g_y_samples,
            'theta_y': self.theta_y_samples,
            'g_q': self.g_q_samples,
            'tau2_q': self.tau2_q_samples,
            'theta_q': self.theta_q_samples,
            'Q': self.Q_samples
        }


class GibbsSampler2Layer_No_W_Selective:
    """
    Gibbs sampler for 2-layer Deep GP model without dimensionality reduction, 
    using selected columns of X.
    
    Model:
        Y | Q, θ_y, g_y, τ² ~ GP(0, τ²(C_y + g_y*I))
        Q | X_selected, θ_q, g_q ~ GP(0, C_q)
        where X_selected is used directly (no W, no Z)
    
    Parameters to sample:
        - τ²_y (tau2): Observation noise variance
        - g_y: Nugget parameter for Y layer
        - θ_y (theta_y): Lengthscale for Y layer
        - Q: Latent layer (n, D)
        - g_q: Nugget parameter for Q layer (held fixed at configured value)
        - tau2_q: Variance for Q layer (held fixed at configured value)
        - θ_q (theta_q): Lengthscale for Q layer (per dimension)
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray, D: int,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g_y: bool = False,
                 use_mle_theta_y: bool = False,
                 kernel_type: str = 'separable_squared_exponential',
                 alpha1: float = 1.0, alpha2: float = 1000.0,
                 beta1: float = 0.01, beta2: float = 0.005,
                 gamma1: float = 1.5, gamma2_y: float = 3.9, gamma2_q: float = 3.9/3,
                 l: float = 1.0, u: float = 2.0,
                 column_indices: Optional[np.ndarray] = None,
                 tau2_y_init: float = 0.005,
                 tau2_q_init: Union[float, np.ndarray] = 0.005,
                 g_y_init: float = 0.00009,
                 g_q_init: Union[float, np.ndarray] = 0.00009,
                 theta_y_init: Union[float, np.ndarray] = 1.0,
                 theta_q_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize 2-layer Gibbs sampler without W, using selected columns.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p)
            D: Number of columns to use from X (must be <= p)
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for τ²_y instead of MCMC (default: False)
            use_mle_g_y: Use MLE for g_y instead of MCMC (default: False)
            use_mle_theta_y: Use MLE for θ_y instead of MCMC (default: False)
            kernel_type: Kernel type ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 
                        'separable_matern32')
            alpha1, alpha2: Inverse Gamma prior parameters for tau2_y
            beta1, beta2: Gamma prior parameters for g_y
            gamma1: Gamma shape parameter (3/2)
            gamma2_y: Gamma rate parameter for theta_y (3.9)
            gamma2_q: Gamma rate parameter for theta_q (3.9/3)
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
        
        # MLE options (independent flags, only for Y layer)
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g_y = use_mle_g_y
        self.use_mle_theta_y = use_mle_theta_y
        
        # Hyperparameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2_y = gamma2_y
        self.gamma2_q = gamma2_q
        self.l = l
        self.u = u
        self.tau2_y_init = float(tau2_y_init)
        self.tau2_q_fixed = _initial_value(tau2_q_init, self.D)
        self.g_y_init = float(g_y_init)
        self.g_q_fixed = _initial_value(g_q_init, self.D)
        self.theta_y_init = _initial_value(theta_y_init, self.D)
        self.theta_q_init = _initial_value(theta_q_init, self.D)
        
        # Storage
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.g_y_samples = np.zeros(self.n_saved)
        if self.D == 1:
            self.theta_y_samples = np.zeros(self.n_saved)
            self.theta_q_samples = np.zeros(self.n_saved)
            self.g_q_samples = np.zeros(self.n_saved)
            self.tau2_q_samples = np.zeros(self.n_saved)
            self.Q_samples = np.zeros((self.n_saved, self.n))
        else:
            self.theta_y_samples = np.zeros((self.n_saved, self.D))
            self.theta_q_samples = np.zeros((self.n_saved, self.D))
            self.g_q_samples = np.zeros((self.n_saved, self.D))
            self.tau2_q_samples = np.zeros((self.n_saved, self.D))
            self.Q_samples = np.zeros((self.n_saved, self.n, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameter starting values."""
        # Initialize Q as copy of X_selected
        if self.D == 1:
            self.Q = self.X_selected[:, 0].copy()
        else:
            self.Q = self.X_selected.copy()
        
        # Y layer hyperparameters
        self.tau2 = self.tau2_y_init
        self.g_y = self.g_y_init
        self.theta_y = _copy_initial(self.theta_y_init)
        
        # Q layer hyperparameters (per dimension)
        self.g_q = _copy_initial(self.g_q_fixed)
        self.tau2_q = _copy_initial(self.tau2_q_fixed)
        self.theta_q = _copy_initial(self.theta_q_init)
    
    def _sample_Q(self):
        """Sample latent layer Q using Elliptical Slice Sampling."""
        if self.D == 1:
            self.Q = sample_Q_2layer_ESS(
                Y=self.Y,
                Q_current=self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                Z=self.X_selected.reshape(-1, 1) if self.X_selected.ndim == 1 else self.X_selected,
                g_y=self.g_y,
                theta_y=self.theta_y,
                theta_q=self.theta_q,
                g_q=self.g_q,
                tau2_y=self.tau2,
                tau2_q=self.tau2_q,
                kernel_type=self.kernel_type
            )
            if self.Q.ndim == 2:
                self.Q = self.Q.flatten()
        else:
            self.Q = sample_Q_2layer_ESS_Dgen(
                Y=self.Y,
                Q_current=self.Q,
                Z=self.X_selected,  # Use X_selected directly
                g_y=self.g_y,
                theta_y=self.theta_y,
                theta_q=self.theta_q,
                g_q=self.g_q,
                tau2_y=self.tau2,
                tau2_q=self.tau2_q,
                kernel_type=self.kernel_type
            )
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the Gibbs sampler."""
        save_idx = 0
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print(f"Running 2-Layer Deep GP Gibbs Sampler without W (D={self.D}, using columns {self.column_indices})")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation (Y layer): tau2={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g_y={'MLE' if self.use_mle_g_y else 'MCMC'}, "
                  f"theta_y={'MLE' if self.use_mle_theta_y else 'MCMC'}")
            print("-"*70)
        
        # Use identity matrix for W (X_selected is used directly)
        W_identity = np.eye(self.D)
        
        for iter in range(self.n_iterations):
            # Sample hyperparameters for Q layer (always MCMC, dimension-wise for separable)
            # The variant keeps latent nugget and variance fixed at configured values.
            # Only theta_q is sampled per dimension
            for m in range(self.D):
                # Fixed hyperparameters for latent layers
                if self.D == 1:
                    self.g_q = float(self.g_q_fixed)
                    self.tau2_q = float(self.tau2_q_fixed)
                else:
                    self.g_q[m] = self.g_q_fixed[m]
                    self.tau2_q[m] = self.tau2_q_fixed[m]
                
                # Sample theta_q[m] for dimension m
                if self.D == 1:
                    Q_m = self.Q
                    X_m = self.X_selected.reshape(-1, 1) if self.X_selected.ndim == 1 else self.X_selected
                    W_identity_m = np.array([[1.0]])
                    
                    theta_m_new = sample_theta_D(
                        Q_m, X_m, W_identity_m, np.array([self.theta_q]),
                        self.tau2_q, self.g_q, self.gamma1, self.gamma2_q,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                    self.theta_q = theta_m_new[0] if hasattr(theta_m_new, '__len__') else theta_m_new
                else:
                    Q_m = self.Q[:, m]  # (n,)
                    X_m = self.X_selected[:, m].reshape(-1, 1)  # (n, 1)
                    W_identity_m = np.array([[1.0]])  # (1, 1)
                    
                    theta_m_new = sample_theta_D_Dgen(
                        Q_m, X_m, W_identity_m, np.array([self.theta_q[m]]),
                        self.tau2_q[m], self.g_q[m], self.gamma1, self.gamma2_q,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                    self.theta_q[m] = theta_m_new[0]
            
            # Sample Q using ESS
            self._sample_Q()
            
            # Sample/estimate hyperparameters for Y layer (with individual MLE options)
            # tau2_y
            if self.use_mle_tau2:
                if self.D == 1:
                    self.tau2 = estimate_tau2_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.theta_y, self.g_y,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = estimate_tau2_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.theta_y, self.g_y,
                        kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.tau2 = sample_tau2(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.tau2,
                        self.theta_y, self.g_y, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.tau2 = sample_tau2_Dgen(
                        self.Y, self.Q, W_identity, self.tau2,
                        self.theta_y, self.g_y, self.alpha1, self.alpha2,
                        kernel_type=self.kernel_type
                    )
            
            # g_y
            if self.use_mle_g_y:
                if self.D == 1:
                    self.g_y = estimate_g_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.theta_y, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
                else:
                    self.g_y = estimate_g_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.theta_y, self.tau2,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                if self.D == 1:
                    self.g_y = sample_g(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.g_y,
                        self.theta_y, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
                else:
                    self.g_y = sample_g_Dgen(
                        self.Y, self.Q, W_identity, self.g_y,
                        self.theta_y, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, kernel_type=self.kernel_type
                    )
            
            # theta_y
            if self.use_mle_theta_y:
                if self.D == 1:
                    self.theta_y = estimate_theta_D_MLE(
                        self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                        W_identity, self.g_y, self.tau2,
                        kernel_type=self.kernel_type
                    )
                else:
                    self.theta_y = estimate_theta_D_MLE_Dgen(
                        self.Y, self.Q, W_identity, self.g_y, self.tau2, self.D,
                        n_grid=20, kernel_type=self.kernel_type
                    )
            else:
                # For separable kernels, sample dimension-wise
                if 'separable' in self.kernel_type and self.D > 1:
                    for m in range(self.D):
                        Q_m = self.Q[:, m].reshape(-1, 1)  # (n, 1)
                        W_identity_m = np.array([[1.0]])  # (1, 1)
                        
                        theta_m_new = sample_theta_D_Dgen(
                            self.Y, Q_m, W_identity_m,
                            np.array([self.theta_y[m]]),
                            self.tau2, self.g_y,
                            self.gamma1, self.gamma2_y,
                            self.l, self.u,
                            kernel_type=self.kernel_type
                        )
                        self.theta_y[m] = theta_m_new[0]
                else:
                    # For isotropic or D=1, sample full vector/scalar
                    if self.D == 1:
                        self.theta_y = sample_theta_D(
                            self.Y, self.Q.reshape(-1, 1) if self.Q.ndim == 1 else self.Q,
                            W_identity, self.theta_y,
                            self.tau2, self.g_y, self.gamma1, self.gamma2_y,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
                    else:
                        self.theta_y = sample_theta_D_Dgen(
                            self.Y, self.Q, W_identity, self.theta_y,
                            self.tau2, self.g_y, self.gamma1, self.gamma2_y,
                            self.l, self.u, kernel_type=self.kernel_type
                        )
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.g_y_samples[save_idx] = self.g_y
                if self.D == 1:
                    self.theta_y_samples[save_idx] = self.theta_y
                    self.theta_q_samples[save_idx] = self.theta_q
                    self.g_q_samples[save_idx] = self.g_q
                    self.tau2_q_samples[save_idx] = self.tau2_q
                    self.Q_samples[save_idx] = self.Q
                else:
                    self.theta_y_samples[save_idx] = self.theta_y
                    self.theta_q_samples[save_idx] = self.theta_q
                    self.g_q_samples[save_idx] = self.g_q
                    self.tau2_q_samples[save_idx] = self.tau2_q
                    self.Q_samples[save_idx] = self.Q
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
            'g_y': self.g_y_samples,
            'theta_y': self.theta_y_samples,
            'g_q': self.g_q_samples,
            'tau2_q': self.tau2_q_samples,
            'theta_q': self.theta_q_samples,
            'Q': self.Q_samples,
            'column_indices': self.column_indices,  # Return which columns were used
            'X_selected': self.X_selected  # Return selected X for reference
        }


if __name__ == "__main__":
    print("="*70)
    print("Gibbs Sampler Variants for Layer 2 - Test")
    print("="*70)
    
    np.random.seed(42)
    n, p = 20, 5
    
    X = np.random.randn(n, p)
    Y = np.random.randn(n)
    
    print("\n1. Testing W_Known variant:")
    W_fixed = np.random.randn(p, 2)
    W_fixed, _ = np.linalg.qr(W_fixed)
    
    sampler1 = GibbsSampler2Layer_W_Known(
        Y=Y, X=X, W_fixed=W_fixed,
        n_iterations=2, burn_in=0, thin=1,
        kernel_type='separable_squared_exponential'
    )
    samples1 = sampler1.run(verbose=False)
    print(f"   ✅ W_Known: {len(samples1['tau2_y'])} samples")
    
    print("\n2. Testing No_W variant:")
    sampler2 = GibbsSampler2Layer_No_W(
        Y=Y, X=X,
        n_iterations=2, burn_in=0, thin=1,
        kernel_type='separable_squared_exponential'
    )
    samples2 = sampler2.run(verbose=False)
    print(f"   ✅ No_W: {len(samples2['tau2_y'])} samples")
    
    print("\n3. Testing No_W_Selective variant:")
    sampler3 = GibbsSampler2Layer_No_W_Selective(
        Y=Y, X=X, D=3,
        n_iterations=2, burn_in=0, thin=1,
        kernel_type='separable_squared_exponential'
    )
    samples3 = sampler3.run(verbose=False)
    print(f"   ✅ No_W_Selective: {len(samples3['tau2_y'])} samples")
    
    print("\n" + "="*70)
    print("✅ All variants tested successfully!")
    print("="*70)
