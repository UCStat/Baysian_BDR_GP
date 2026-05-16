"""
Gibbs Sampler for 1, 2, and 3-Layer Deep Gaussian Process Models (D=1)

This module implements Gibbs samplers for Deep Gaussian Process models with
Bayesian dimensionality reduction to D=1 for 1-layer, 2-layer, and 3-layer architectures.

Potential Energy Functions:
    1-Layer: U(W) = -log L(Y|X,W,θ,g,τ²) - log π(W;F)
    2-Layer: U(W) = -log L(Y|Q,θ_y,g_y,τ²) - log L(Q|X,W,θ_q,g_q) - log π(W;F)
    3-Layer: U(W) = -log L(Y|Q,θ_y,g_y,τ²) - log L(Q|R,θ_q,g_q) - log L(R|X,W,θ_r,g_r) - log π(W;F)
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import time

# Add parent directories to path for imports
base_dir = Path(__file__).parent.parent
param_sampler_path = str(base_dir / "Parameter Sampler")
if param_sampler_path not in sys.path:
    sys.path.insert(0, param_sampler_path)

# Import parameter sampling functions (dynamic path, works at runtime)
from parameter_sampler_D1 import (  # type: ignore[import]
    sample_tau2, sample_g, sample_theta_D, sample_W_HMC_stiefel,
    sample_M, sample_V, sample_Lambda_slice,
    sample_Q_2layer_ESS, sample_Q_3layer_ESS, sample_R_3layer_ESS,
    rmf_matrix,
    covar_sep, log_likelihood_gp,
    estimate_tau2_MLE, estimate_g_MLE, estimate_theta_D_MLE,
    estimate_all_hyperparameters_MLE
)

# Testing defaults (requested)
HMC_EPS_DEFAULT = 0.09
HMC_T_STEPS_DEFAULT = 15
ALPHA1_DEFAULT = 0.001
ALPHA2_DEFAULT = 0.001
GAMMA_SHAPE_DEFAULT = 1.5
GAMMA_RATE_G_AND_THETA_Y = 3.9
GAMMA_RATE_THETA_Q = 3.9 / 3.0
GAMMA_RATE_THETA_R = 3.9 / 6.0
LAMBDA_GAMMA_SHAPE = 5.0 / 2.0
LAMBDA_GAMMA_RATE = 10.0 / 3.0
LAMBDA_MIN = 2.0 + 1e-6  # Keep compatible with Lambda ESS positivity check.


def _draw_lambda_prior(size: int) -> np.ndarray:
    """Draw Lambda prior values from Gamma(5/2, 10/3), clipped for ESS stability."""
    draws = np.random.gamma(
        shape=LAMBDA_GAMMA_SHAPE,
        scale=1.0 / LAMBDA_GAMMA_RATE,
        size=size
    )
    return np.maximum(draws.astype(float), LAMBDA_MIN)


def _initialize_mvlw_from_svd(p: int, D: int):
    """
    Initialize M, Lambda, V from SVD of a Gaussian matrix and sample W from
    matrix Langevin with parameter M @ diag(Lambda) @ V.T.
    """
    F = np.random.normal(loc=0.0, scale=1.0, size=(p, D))
    M_full, singular_vals, Vt = np.linalg.svd(F, full_matrices=False)
    M_pre = M_full[:, :D]
    V_pre = Vt.T[:, :D]
    Lambda_pre = np.maximum(singular_vals[:D], LAMBDA_MIN)
    W_init = rmf_matrix(M_pre @ np.diag(Lambda_pre) @ V_pre.T)
    if W_init.ndim == 1:
        W_init = W_init.reshape(-1, 1)
    return M_pre, Lambda_pre, V_pre, W_init


def _coerce_float_init(value, default: float) -> float:
    """Coerce scalar/array-like initial value to float."""
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _coerce_matrix_init(value, shape: Tuple[int, int], name: str) -> np.ndarray:
    """Validate and coerce matrix initial value."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return arr.copy()


def _coerce_lambda_init(value, D: int) -> np.ndarray:
    """Coerce Lambda init to vector length D."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        if arr.shape != (D, D):
            raise ValueError(f"Lambda_init must have shape ({D}, {D}) or ({D},), got {arr.shape}.")
        arr = np.diag(arr)
    arr = arr.reshape(-1)
    if arr.size != D:
        raise ValueError(f"Lambda_init must have length {D}, got {arr.size}.")
    return arr.copy()


def _coerce_latent_init(value, n: int, D: int, name: str) -> np.ndarray:
    """Validate latent initial matrix with shape (n, D)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and D == 1 and arr.size == n:
        arr = arr.reshape(n, 1)
    if arr.shape != (n, D):
        raise ValueError(f"{name} must have shape ({n}, {D}), got {arr.shape}.")
    return arr.copy()


class GibbsSampler1Layer:
    """
    Gibbs sampler for 1-layer GP model with dimensionality reduction.
    
    Model:
        Y | X, W, θ, g, τ² ~ GP(0, τ²(C_y + g*I))
        where Z = XW and C_y is the covariance from Z
    
    Parameters to sample:
        - τ² (tau2): Observation noise variance
        - g: Nugget parameter
        - θ (theta_D): Lengthscale
        - W: Projection matrix (p × 1)
        - M, V, Λ: Matrix Langevin prior parameters
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray, D: int = 1,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_tf_gradients: bool = False,
                 use_mle_tau2: bool = False,
                 use_mle_g: bool = False,
                 use_mle_theta: bool = False,
                 use_mle_all: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 prior_M: Optional[np.ndarray] = None,
                 prior_V: Optional[np.ndarray] = None,
                 W_init: Optional[np.ndarray] = None,
                 M_init: Optional[np.ndarray] = None,
                 V_init: Optional[np.ndarray] = None,
                 Lambda_init: Optional[np.ndarray] = None,
                 tau2_y_init: float = 0.005,
                 g_y_init: float = 0.00009,
                 theta_y_init: Optional[Union[float, np.ndarray]] = 1.0,
                 mv_sampler: str = "python",
                 rstiefel_rscol: Optional[int] = None):
        """
        Initialize 1-layer Gibbs sampler.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p)
            D: Reduced dimension (must be 1 for this module)
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_tf_gradients: Use TensorFlow for W gradients
            use_mle_tau2: Use MLE for τ² instead of MCMC (default: False)
            use_mle_g: Use MLE for g instead of MCMC (default: False)
            use_mle_theta: Use MLE for θ instead of MCMC (default: False)
            use_mle_all: Use MLE for all hyperparameters jointly (overrides individual flags)
            kernel_type: Type of kernel to use. Options:
                - 'isotropic_squared_exponential' (default)
                - 'separable_squared_exponential'
                - 'isotropic_matern32'
                - 'separable_matern32'
            mv_sampler: Backend for M/V Gibbs updates: 'python' or 'rstiefel'
            rstiefel_rscol: Optional number of columns for rstiefel simultaneous updates
        """
        self.Y = Y.flatten()
        self.X = X
        self.n, self.p = X.shape
        self.D = D
        
        if D != 1:
            raise ValueError("This module only supports D=1. Use D=1.")
        
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.use_tf = use_tf_gradients
        self.kernel_type = kernel_type
        self.mv_sampler = mv_sampler
        self.rstiefel_rscol = rstiefel_rscol
        
        # MLE options: individual flags are independent, use_mle_all overrides all
        if use_mle_all:
            self.use_mle_tau2 = True
            self.use_mle_g = True
            self.use_mle_theta = True
        else:
            self.use_mle_tau2 = use_mle_tau2
            self.use_mle_g = use_mle_g
            self.use_mle_theta = use_mle_theta
        self.use_mle_all = use_mle_all
        
        # Prior for M and V
        if prior_M is None:
            self.prior_M = None  # Will be set to zeros in _initialize_hyperparameters
        else:
            self.prior_M = prior_M
        
        if prior_V is None:
            self.prior_V = None  # Will be set to zeros in _initialize_hyperparameters
        else:
            self.prior_V = prior_V

        # Initial values
        self.W_init = W_init
        self.M_init = M_init
        self.V_init = V_init
        self.Lambda_init = Lambda_init
        self.tau2_y_init = tau2_y_init
        self.g_y_init = g_y_init
        self.theta_y_init = theta_y_init
        
        # Storage
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
        self._initialize_hyperparameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays for samples."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.g_samples = np.zeros(self.n_saved)
        self.theta_D_samples = np.zeros(self.n_saved)
        self.W_samples = np.zeros((self.n_saved, self.p, self.D))
        self.M_samples = np.zeros((self.n_saved, self.p, self.D))
        self.V_samples = np.zeros((self.n_saved, self.D, self.D))
        self.Lambda_samples = np.zeros((self.n_saved, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameter starting values."""
        # Initialize M, Lambda, V via SVD, then W via matrix Langevin(MΛV^T).
        M0, Lambda0, V0, W0 = _initialize_mvlw_from_svd(self.p, self.D)
        self.M = M0 if self.M_init is None else _coerce_matrix_init(self.M_init, (self.p, self.D), "M_init")
        self.Lambda = Lambda0 if self.Lambda_init is None else _coerce_lambda_init(self.Lambda_init, self.D)
        self.V = V0 if self.V_init is None else _coerce_matrix_init(self.V_init, (self.D, self.D), "V_init")
        self.W = W0 if self.W_init is None else _coerce_matrix_init(self.W_init, (self.p, self.D), "W_init")
        
        # Initialize other parameters
        self.tau2 = _coerce_float_init(self.tau2_y_init, 0.005)
        self.g = _coerce_float_init(self.g_y_init, 0.00009)
        self.theta_D = _coerce_float_init(self.theta_y_init, 1.0)

        # Priors F_M and F_V from matrix Langevin using M_prev and V_prev.
        if self.prior_M is None:
            self.prior_M = rmf_matrix(self.M)
        if self.prior_V is None:
            self.prior_V = rmf_matrix(self.V)
    
    def _initialize_hyperparameters(self):
        """Initialize hyperparameters for priors."""
        # tau2 ~ InvGamma(alpha1, alpha2)
        self.alpha1 = ALPHA1_DEFAULT
        self.alpha2 = ALPHA2_DEFAULT
        
        # g ~ Gamma(3/2, 3.9)
        self.beta1 = GAMMA_SHAPE_DEFAULT
        self.beta2 = GAMMA_RATE_G_AND_THETA_Y
        
        # theta_D ~ Gamma(gamma1, gamma2)
        # For 1-layer: theta ~ Gamma(3/2, 3.9)
        self.gamma1 = GAMMA_SHAPE_DEFAULT
        self.gamma2 = GAMMA_RATE_G_AND_THETA_Y
        
        # Proposal parameters
        self.l = 1.0
        self.u = 2.0
        
        # HMC parameters
        self.eps_hmc = HMC_EPS_DEFAULT
        self.T_step_hmc = HMC_T_STEPS_DEFAULT
        
        # Lambda prior: Gamma(5/2, 10/3)
        self.lambda_b1 = LAMBDA_GAMMA_SHAPE
        self.lambda_b2 = LAMBDA_GAMMA_RATE
        self.nu_lambda = _draw_lambda_prior(self.D)
        
        # Prior for M and V (set defaults if not provided)
        if self.prior_M is None:
            self.prior_M = np.zeros((self.p, self.D))
        
        if self.prior_V is None:
            self.prior_V = np.zeros((self.D, self.D))
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Run the Gibbs sampler.
        
        Args:
            verbose: Print progress
            
        Returns:
            Dictionary with all samples
        """
        save_idx = 0
        start_time = time.time()
        
        if verbose:
            print("="*70)
            print(f"Running 1-Layer Gibbs Sampler (D={self.D})")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Use TensorFlow gradients: {self.use_tf}")
            print(f"Kernel type: {self.kernel_type}")
            if self.use_mle_all:
                print(f"Hyperparameter estimation: MLE (joint)")
            else:
                print(f"Hyperparameter estimation: tau2_y={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                      f"g_y={'MLE' if self.use_mle_g else 'MCMC'}, "
                      f"theta_D_y={'MLE' if self.use_mle_theta else 'MCMC'}")
            print("-"*70)
        
        for iter in range(self.n_iterations):

            # Step 1: sample M
            self.M = sample_M(self.W, self.Lambda, self.V, self.p,
                            prior_M=self.prior_M, M_prev=self.M,
                            mv_sampler=self.mv_sampler, rstiefel_rscol=self.rstiefel_rscol)
            
            # Step 2: sample V
            self.V = sample_V(self.W, self.Lambda, self.M, self.D,
                            prior_V=self.prior_V, V_prev=self.V,
                            mv_sampler=self.mv_sampler, rstiefel_rscol=self.rstiefel_rscol)
            
            # Step 3: sample Lambda
            self.nu_lambda = _draw_lambda_prior(self.D)
            self.Lambda = sample_Lambda_slice(
                self.Lambda, self.nu_lambda, self.M, self.V, self.W, self.p
            )
            
            # Step 4: sample W
            F_Wprior = self.M @ np.diag(self.Lambda) @ self.V.T
            self.W = sample_W_HMC_stiefel(
                self.Y, self.X, self.W,
                F_Wprior=F_Wprior, M=1, eps=self.eps_hmc, T_step=self.T_step_hmc,
                use_tf=self.use_tf, kernel_type=self.kernel_type,
                layer=1, tau2_y=self.tau2, theta_D_y=self.theta_D, g_y=self.g
            )


            # Steps 5-7: sample/update tau2_y, g_y, theta_D_y
            if self.use_mle_all:
                # Joint MLE estimation, then assign in the same order.
                mle_estimates = estimate_all_hyperparameters_MLE(
                    self.Y, self.X, self.W,
                    self.tau2, self.g, self.theta_D,
                    n_iterations=3, verbose=False, kernel_type=self.kernel_type
                )
                # Step 5: tau2_y
                self.tau2 = mle_estimates['tau2']
                # Step 6: g_y
                self.g = mle_estimates['g']
                # Step 7: theta_D_y
                self.theta_D = mle_estimates['theta_D']
            else:
                # Step 5: tau2_y
                if self.use_mle_tau2:
                    self.tau2 = estimate_tau2_MLE(self.Y, self.X, self.W, self.theta_D, self.g, self.kernel_type)
                else:
                    self.tau2 = sample_tau2(
                        self.Y, self.X, self.W, self.tau2,
                        self.theta_D, self.g, self.alpha1, self.alpha2, self.kernel_type
                    )
                
                # Step 6: g_y
                if self.use_mle_g:
                    self.g = estimate_g_MLE(self.Y, self.X, self.W, self.theta_D, self.tau2, kernel_type=self.kernel_type)
                else:
                    self.g = sample_g(
                        self.Y, self.X, self.W, self.g,
                        self.theta_D, self.tau2, self.beta1, self.beta2,
                        self.l, self.u, self.kernel_type
                    )
                
                # Step 7: theta_D_y
                if self.use_mle_theta:
                    self.theta_D = estimate_theta_D_MLE(self.Y, self.X, self.W, self.g, self.tau2, kernel_type=self.kernel_type)
                else:
                    self.theta_D = sample_theta_D(
                        self.Y, self.X, self.W, self.theta_D,
                        self.tau2, self.g, self.gamma1, self.gamma2,
                        self.l, self.u, self.kernel_type
                    )
            
            
            
            # Save if past burn-in and on thin interval
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.g_samples[save_idx] = self.g
                self.theta_D_samples[save_idx] = self.theta_D
                self.W_samples[save_idx] = self.W
                self.M_samples[save_idx] = self.M
                self.V_samples[save_idx] = self.V
                self.Lambda_samples[save_idx] = self.Lambda
                save_idx += 1
            
            # Print progress
            if verbose and (iter + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iter+1}/{self.n_iterations} | "
                      f"tau2_y={self.tau2:.4f}, g_y={self.g:.4f}, theta_D_y={self.theta_D:.4f} | "
                      f"Time: {elapsed:.1f}s")
        
        if verbose:
            total_time = time.time() - start_time
            print("-"*70)
            print(f"Sampling complete! Total time: {total_time:.1f}s")
            print("="*70)
        
        return {
            'tau2_y': self.tau2_samples,
            'g_y': self.g_samples,
            'theta_D_y': self.theta_D_samples,
            'W': self.W_samples,
            'M': self.M_samples,
            'V': self.V_samples,
            'Lambda': self.Lambda_samples
        }


class GibbsSampler2Layer:
    """
    Gibbs sampler for 2-layer Deep GP model with dimensionality reduction.
    
    Model:
        Y | Q, θ_y, g_y, τ²_y ~ GP(0, τ²_y(C_y + g_y*I))
        Q | X, W, θ_q, g_q, τ²_q ~ GP(0, τ²_q(C_q + g_q*I))
        where Z = XW
    
    Parameters to sample:
        - τ²_y: Observation noise variance for Y layer
        - τ²_q: Latent variance for Q layer
        - g_y, g_q: Nugget parameters for each layer
        - θ_y, θ_q: Lengthscales for each layer
        - W: Projection matrix (p × 1)
        - Q: Latent layer (n × 1)
        - M, V, Λ: Matrix Langevin prior parameters
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray, D: int = 1,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_tf_gradients: bool = False,
                 use_mle_tau2: bool = False,
                 use_mle_g_y: bool = False,
                 use_mle_theta_y: bool = False,
                 use_mle_all: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 prior_M: Optional[np.ndarray] = None,
                 prior_V: Optional[np.ndarray] = None,
                 W_init: Optional[np.ndarray] = None,
                 M_init: Optional[np.ndarray] = None,
                 V_init: Optional[np.ndarray] = None,
                 Lambda_init: Optional[np.ndarray] = None,
                 tau2_y_init: float = 0.005,
                 tau2_q_init: float = 0.005,
                 g_y_init: float = 0.00009,
                 g_q_init: float = 0.00009,
                 theta_y_init: Optional[Union[float, np.ndarray]] = 1.0,
                 theta_q_init: Optional[Union[float, np.ndarray]] = 1.0,
                 Q_init: Optional[np.ndarray] = None,
                 mv_sampler: str = "python",
                 rstiefel_rscol: Optional[int] = None):
        """
        Initialize 2-layer Gibbs sampler.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p)
            D: Reduced dimension (must be 1)
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_tf_gradients: Use TensorFlow for gradients
            use_mle_tau2: Use MLE for τ² instead of MCMC (Y layer)
            use_mle_g_y: Use MLE for g_y instead of MCMC (Y layer)
            use_mle_theta_y: Use MLE for θ_y instead of MCMC (Y layer)
            use_mle_all: Use MLE for all hyperparameters jointly (overrides individual flags)
            kernel_type: Kernel type for covariance functions ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 'separable_matern32')
            prior_M: Prior for M (p, D), default: zeros
            prior_V: Prior for V (D, D), default: zeros
            mv_sampler: Backend for M/V Gibbs updates: 'python' or 'rstiefel'
            rstiefel_rscol: Optional number of columns for rstiefel simultaneous updates
        """
        self.Y = Y.flatten()
        self.X = X
        self.n, self.p = X.shape
        self.D = D
        
        if D != 1:
            raise ValueError("This module only supports D=1.")
        
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.use_tf = use_tf_gradients
        
        # Kernel type
        self.kernel_type = kernel_type
        self.mv_sampler = mv_sampler
        self.rstiefel_rscol = rstiefel_rscol
        
        # MLE options (individual flags for Y layer parameters)
        self.use_mle_all = use_mle_all
        self.use_mle_tau2 = use_mle_tau2 or use_mle_all
        self.use_mle_g_y = use_mle_g_y or use_mle_all
        self.use_mle_theta_y = use_mle_theta_y or use_mle_all
        
        # Store priors
        self.prior_M = prior_M
        self.prior_V = prior_V

        # Initial values
        self.W_init = W_init
        self.M_init = M_init
        self.V_init = V_init
        self.Lambda_init = Lambda_init
        self.tau2_y_init = tau2_y_init
        self.tau2_q_init = tau2_q_init
        self.g_y_init = g_y_init
        self.g_q_init = g_q_init
        self.theta_y_init = theta_y_init
        self.theta_q_init = theta_q_init
        self.Q_init = Q_init
        
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
        self._initialize_hyperparameters()
    
    def _initialize_storage(self):
        """Initialize storage arrays."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.tau2_q_samples = np.zeros(self.n_saved)
        self.g_y_samples = np.zeros(self.n_saved)
        self.g_q_samples = np.zeros(self.n_saved)
        self.theta_y_samples = np.zeros(self.n_saved)
        self.theta_q_samples = np.zeros(self.n_saved)
        self.W_samples = np.zeros((self.n_saved, self.p, self.D))
        self.Q_samples = np.zeros((self.n_saved, self.n, self.D))
        self.M_samples = np.zeros((self.n_saved, self.p, self.D))
        self.V_samples = np.zeros((self.n_saved, self.D, self.D))
        self.Lambda_samples = np.zeros((self.n_saved, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameters."""
        # Initialize M, Lambda, V via SVD, then W via matrix Langevin(MΛV^T).
        M0, Lambda0, V0, W0 = _initialize_mvlw_from_svd(self.p, self.D)
        self.M = M0 if self.M_init is None else _coerce_matrix_init(self.M_init, (self.p, self.D), "M_init")
        self.Lambda = Lambda0 if self.Lambda_init is None else _coerce_lambda_init(self.Lambda_init, self.D)
        self.V = V0 if self.V_init is None else _coerce_matrix_init(self.V_init, (self.D, self.D), "V_init")
        self.W = W0 if self.W_init is None else _coerce_matrix_init(self.W_init, (self.p, self.D), "W_init")
        
        # Initialize Q from user value or standard normal.
        if self.Q_init is None:
            self.Q = np.random.normal(loc=0.0, scale=1.0, size=(self.n, self.D))
        else:
            self.Q = _coerce_latent_init(self.Q_init, self.n, self.D, "Q_init")
        
        self.tau2 = _coerce_float_init(self.tau2_y_init, 0.005)
        self.tau2_q = _coerce_float_init(self.tau2_q_init, 0.005)
        self.g_y = _coerce_float_init(self.g_y_init, 0.00009)
        self.g_q = _coerce_float_init(self.g_q_init, 0.00009)
        self.theta_y = _coerce_float_init(self.theta_y_init, 1.0)
        self.theta_q = _coerce_float_init(self.theta_q_init, 1.0)

        # Priors F_M and F_V from matrix Langevin using M_prev and V_prev.
        if self.prior_M is None:
            self.prior_M = rmf_matrix(self.M)
        if self.prior_V is None:
            self.prior_V = rmf_matrix(self.V)
    
    def _initialize_hyperparameters(self):
        """Initialize hyperparameters for 2-layer model."""
        # tau2 ~ InvGamma(alpha1, alpha2)
        self.alpha1 = ALPHA1_DEFAULT
        self.alpha2 = ALPHA2_DEFAULT
        
        # g ~ Gamma(3/2, 3.9)
        self.beta1 = GAMMA_SHAPE_DEFAULT
        self.beta2 = GAMMA_RATE_G_AND_THETA_Y
        
        # theta ~ Gamma(3/2, b) with layer-specific rates
        # 2-layer: theta_y (outer), theta_q (inner)
        self.gamma1 = GAMMA_SHAPE_DEFAULT
        self.gamma2_y = GAMMA_RATE_G_AND_THETA_Y
        self.gamma2_q = GAMMA_RATE_THETA_Q
        
        # Proposal parameters
        self.l = 1.0
        self.u = 2.0
        
        # HMC parameters
        self.eps_hmc = HMC_EPS_DEFAULT
        self.T_step_hmc = HMC_T_STEPS_DEFAULT
        
        # Lambda prior: Gamma(5/2, 10/3)
        self.lambda_b1 = LAMBDA_GAMMA_SHAPE
        self.lambda_b2 = LAMBDA_GAMMA_RATE
        self.nu_lambda = _draw_lambda_prior(self.D)
        
        # Prior for M and V
        if self.prior_M is None:
            self.prior_M = np.zeros((self.p, self.D))
        if self.prior_V is None:
            self.prior_V = np.zeros((self.D, self.D))
    
    def _sample_Q(self):
        """Sample latent layer Q using Elliptical Slice Sampling."""
        Z = self.X @ self.W
        
        # Use ESS for Q sampling (2-layer, D=1)
        self.Q = sample_Q_2layer_ESS(
            Y=self.Y,
            Q_current=self.Q,
            Z=Z,
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
            print(f"Running 2-Layer Deep GP Gibbs Sampler (D={self.D})")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Use TensorFlow gradients: {self.use_tf}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation (Y layer): "
                  f"tau2_y={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g_y={'MLE' if self.use_mle_g_y else 'MCMC'}, "
                  f"theta_D_y={'MLE' if self.use_mle_theta_y else 'MCMC'}")
            print("-"*70)
        
        for iter in range(self.n_iterations):
            # Step 1: sample M
            self.M = sample_M(self.W, self.Lambda, self.V, self.p,
                              prior_M=self.prior_M, M_prev=self.M,
                              mv_sampler=self.mv_sampler, rstiefel_rscol=self.rstiefel_rscol)

            # Step 2: sample V
            self.V = sample_V(self.W, self.Lambda, self.M, self.D,
                              prior_V=self.prior_V, V_prev=self.V,
                              mv_sampler=self.mv_sampler, rstiefel_rscol=self.rstiefel_rscol)

            # Step 3: sample Lambda
            self.nu_lambda = _draw_lambda_prior(self.D)
            self.Lambda = sample_Lambda_slice(
                self.Lambda, self.nu_lambda, self.M, self.V, self.W, self.p
            )

            # Step 4: sample W
            F_Wprior = self.M @ np.diag(self.Lambda) @ self.V.T
            self.W = sample_W_HMC_stiefel(
                self.Y, self.X, self.W,
                F_Wprior=F_Wprior, M=1, eps=self.eps_hmc, T_step=self.T_step_hmc,
                use_tf=self.use_tf, kernel_type=self.kernel_type,
                layer=2, Q=self.Q,
                tau2_y=self.tau2, tau2_q=self.tau2_q,
                theta_D_y=self.theta_y, theta_D_q=self.theta_q,
                g_y=self.g_y, g_q=self.g_q
            )

            # Step 5: sample tau2_q
            self.tau2_q = sample_tau2(
                self.Q.flatten(), self.X, self.W, self.tau2_q,
                self.theta_q, self.g_q, self.alpha1, self.alpha2,
                kernel_type=self.kernel_type
            )

            # Step 6: sample g_q
            self.g_q = sample_g(
                self.Q.flatten(), self.X, self.W, self.g_q,
                self.theta_q, self.tau2_q, self.beta1, self.beta2, self.l, self.u,
                kernel_type=self.kernel_type
            )
            
            # Step 7: sample theta_D_q (stored as theta_q)
            self.theta_q = sample_theta_D(
                self.Q.flatten(), self.X, self.W, self.theta_q,
                self.tau2_q, self.g_q, self.gamma1, self.gamma2_q, self.l, self.u,
                kernel_type=self.kernel_type
            )
            
            # Step 8: sample Q
            self._sample_Q()
            
            # Step 9: sample/update tau2_y
            if self.use_mle_tau2:
                self.tau2 = estimate_tau2_MLE(
                    self.Y, self.Q, np.eye(self.n), self.theta_y, self.g_y,
                    kernel_type=self.kernel_type
                )
            else:
                self.tau2 = sample_tau2(
                    self.Y, self.Q, np.eye(self.n), self.tau2,
                    self.theta_y, self.g_y, self.alpha1, self.alpha2,
                    kernel_type=self.kernel_type
                )
            
            # Step 10: sample/update g_y
            if self.use_mle_g_y:
                self.g_y = estimate_g_MLE(
                    self.Y, self.Q, np.eye(self.n), self.theta_y, self.tau2,
                    kernel_type=self.kernel_type
                )
            else:
                self.g_y = sample_g(
                    self.Y, self.Q, np.eye(self.n), self.g_y,
                    self.theta_y, self.tau2, self.beta1, self.beta2, self.l, self.u,
                    kernel_type=self.kernel_type
                )
            
            # Step 11: sample/update theta_D_y (stored as theta_y)
            if self.use_mle_theta_y:
                self.theta_y = estimate_theta_D_MLE(
                    self.Y, self.Q, np.eye(self.n), self.g_y, self.tau2,
                    kernel_type=self.kernel_type
                )
            else:
                self.theta_y = sample_theta_D(
                    self.Y, self.Q, np.eye(self.n), self.theta_y,
                    self.tau2, self.g_y, self.gamma1, self.gamma2_y, self.l, self.u,
                    kernel_type=self.kernel_type
                )
           
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.tau2_q_samples[save_idx] = self.tau2_q
                self.g_y_samples[save_idx] = self.g_y
                self.g_q_samples[save_idx] = self.g_q
                self.theta_y_samples[save_idx] = self.theta_y
                self.theta_q_samples[save_idx] = self.theta_q
                self.W_samples[save_idx] = self.W
                self.Q_samples[save_idx] = self.Q
                self.M_samples[save_idx] = self.M
                self.V_samples[save_idx] = self.V
                self.Lambda_samples[save_idx] = self.Lambda
                save_idx += 1
            
            if verbose and (iter + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iter+1}/{self.n_iterations} | "
                      f"tau2_q={self.tau2_q:.4f}, g_q={self.g_q:.4f}, theta_D_q={self.theta_q:.4f} | "
                      f"tau2_y={self.tau2:.4f}, g_y={self.g_y:.4f}, theta_D_y={self.theta_y:.4f} | "
                      f"Time: {elapsed:.1f}s")
        
        if verbose:
            print("-"*70)
            print(f"Sampling complete! Total time: {time.time() - start_time:.1f}s")
            print("="*70)
        
        return {
            'tau2_y': self.tau2_samples,
            'tau2_q': self.tau2_q_samples,
            'g_y': self.g_y_samples,
            'g_q': self.g_q_samples,
            'theta_y': self.theta_y_samples,
            'theta_q': self.theta_q_samples,
            'W': self.W_samples,
            'Q': self.Q_samples,
            'M': self.M_samples,
            'V': self.V_samples,
            'Lambda': self.Lambda_samples
        }


class GibbsSampler3Layer:
    """
    Gibbs sampler for 3-layer Deep GP model with dimensionality reduction.
    
    Model:
        Y | Q, θ_y, g_y, τ²_y ~ GP(0, τ²_y(C_y + g_y*I))
        Q | R, θ_q, g_q, τ²_q ~ GP(0, τ²_q(C_q + g_q*I))
        R | X, W, θ_r, g_r, τ²_r ~ GP(0, τ²_r(C_r + g_r*I))
        where Z = XW
    """
    
    def __init__(self, Y: np.ndarray, X: np.ndarray, D: int = 1,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_tf_gradients: bool = False,
                 use_mle_tau2: bool = False,
                 use_mle_g_y: bool = False,
                 use_mle_theta_y: bool = False,
                 use_mle_all: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 prior_M: Optional[np.ndarray] = None,
                 prior_V: Optional[np.ndarray] = None,
                 W_init: Optional[np.ndarray] = None,
                 M_init: Optional[np.ndarray] = None,
                 V_init: Optional[np.ndarray] = None,
                 Lambda_init: Optional[np.ndarray] = None,
                 tau2_y_init: float = 0.005,
                 tau2_q_init: float = 0.005,
                 tau2_r_init: float = 0.005,
                 g_y_init: float = 0.00009,
                 g_q_init: float = 0.00009,
                 g_r_init: float = 0.00009,
                 theta_y_init: Optional[Union[float, np.ndarray]] = 1.0,
                 theta_q_init: Optional[Union[float, np.ndarray]] = 1.0,
                 theta_r_init: Optional[Union[float, np.ndarray]] = 1.0,
                 Q_init: Optional[np.ndarray] = None,
                 R_init: Optional[np.ndarray] = None,
                 mv_sampler: str = "python",
                 rstiefel_rscol: Optional[int] = None):
        """
        Initialize 3-layer Gibbs sampler.
        
        Args:
            Y: Response vector (n,)
            X: Design matrix (n, p)
            D: Reduced dimension (must be 1)
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            thin: Thinning interval
            use_tf_gradients: Use TensorFlow for gradients
            use_mle_tau2: Use MLE for τ² instead of MCMC (Y layer)
            use_mle_g_y: Use MLE for g_y instead of MCMC (Y layer)
            use_mle_theta_y: Use MLE for θ_y instead of MCMC (Y layer)
            use_mle_all: Use MLE for all hyperparameters jointly (overrides individual flags)
            kernel_type: Kernel type for covariance functions ('isotropic_squared_exponential', 
                        'separable_squared_exponential', 'isotropic_matern32', 'separable_matern32')
            prior_M: Prior for M (p, D), default: zeros
            prior_V: Prior for V (D, D), default: zeros
            mv_sampler: Backend for M/V Gibbs updates: 'python' or 'rstiefel'
            rstiefel_rscol: Optional number of columns for rstiefel simultaneous updates
        """
        self.Y = Y.flatten()
        self.X = X
        self.n, self.p = X.shape
        self.D = D
        
        if D != 1:
            raise ValueError("This module only supports D=1.")
        
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.use_tf = use_tf_gradients
        
        # Kernel type
        self.kernel_type = kernel_type
        self.mv_sampler = mv_sampler
        self.rstiefel_rscol = rstiefel_rscol
        
        # MLE options (individual flags for Y layer parameters)
        self.use_mle_all = use_mle_all
        self.use_mle_tau2 = use_mle_tau2 or use_mle_all
        self.use_mle_g_y = use_mle_g_y or use_mle_all
        self.use_mle_theta_y = use_mle_theta_y or use_mle_all
        
        # Store priors
        self.prior_M = prior_M
        self.prior_V = prior_V

        # Initial values
        self.W_init = W_init
        self.M_init = M_init
        self.V_init = V_init
        self.Lambda_init = Lambda_init
        self.tau2_y_init = tau2_y_init
        self.tau2_q_init = tau2_q_init
        self.tau2_r_init = tau2_r_init
        self.g_y_init = g_y_init
        self.g_q_init = g_q_init
        self.g_r_init = g_r_init
        self.theta_y_init = theta_y_init
        self.theta_q_init = theta_q_init
        self.theta_r_init = theta_r_init
        self.Q_init = Q_init
        self.R_init = R_init
        
        self.n_saved = (n_iterations - burn_in) // thin
        self._initialize_storage()
        self._initialize_parameters()
        self._initialize_hyperparameters()
    
    def _initialize_storage(self):
        """Initialize storage."""
        self.tau2_samples = np.zeros(self.n_saved)
        self.tau2_q_samples = np.zeros(self.n_saved)
        self.tau2_r_samples = np.zeros(self.n_saved)
        self.g_y_samples = np.zeros(self.n_saved)
        self.g_q_samples = np.zeros(self.n_saved)
        self.g_r_samples = np.zeros(self.n_saved)
        self.theta_y_samples = np.zeros(self.n_saved)
        self.theta_q_samples = np.zeros(self.n_saved)
        self.theta_r_samples = np.zeros(self.n_saved)
        self.W_samples = np.zeros((self.n_saved, self.p, self.D))
        self.Q_samples = np.zeros((self.n_saved, self.n, self.D))
        self.R_samples = np.zeros((self.n_saved, self.n, self.D))
        self.M_samples = np.zeros((self.n_saved, self.p, self.D))
        self.V_samples = np.zeros((self.n_saved, self.D, self.D))
        self.Lambda_samples = np.zeros((self.n_saved, self.D))
    
    def _initialize_parameters(self):
        """Initialize parameters."""
        # Initialize M, Lambda, V via SVD, then W via matrix Langevin(MΛV^T).
        M0, Lambda0, V0, W0 = _initialize_mvlw_from_svd(self.p, self.D)
        self.M = M0 if self.M_init is None else _coerce_matrix_init(self.M_init, (self.p, self.D), "M_init")
        self.Lambda = Lambda0 if self.Lambda_init is None else _coerce_lambda_init(self.Lambda_init, self.D)
        self.V = V0 if self.V_init is None else _coerce_matrix_init(self.V_init, (self.D, self.D), "V_init")
        self.W = W0 if self.W_init is None else _coerce_matrix_init(self.W_init, (self.p, self.D), "W_init")

        if self.R_init is None:
            self.R = np.random.normal(loc=0.0, scale=1.0, size=(self.n, self.D))
        else:
            self.R = _coerce_latent_init(self.R_init, self.n, self.D, "R_init")

        if self.Q_init is None:
            self.Q = np.random.normal(loc=0.0, scale=1.0, size=(self.n, self.D))
        else:
            self.Q = _coerce_latent_init(self.Q_init, self.n, self.D, "Q_init")
        
        self.tau2 = _coerce_float_init(self.tau2_y_init, 0.005)
        self.tau2_q = _coerce_float_init(self.tau2_q_init, 0.005)
        self.tau2_r = _coerce_float_init(self.tau2_r_init, 0.005)
        self.g_y = _coerce_float_init(self.g_y_init, 0.00009)
        self.g_q = _coerce_float_init(self.g_q_init, 0.00009)
        self.g_r = _coerce_float_init(self.g_r_init, 0.00009)
        self.theta_y = _coerce_float_init(self.theta_y_init, 1.0)
        self.theta_q = _coerce_float_init(self.theta_q_init, 1.0)
        self.theta_r = _coerce_float_init(self.theta_r_init, 1.0)

        # Priors F_M and F_V from matrix Langevin using M_prev and V_prev.
        if self.prior_M is None:
            self.prior_M = rmf_matrix(self.M)
        if self.prior_V is None:
            self.prior_V = rmf_matrix(self.V)
    
    def _initialize_hyperparameters(self):
        """Initialize hyperparameters for 3-layer model."""
        # tau2 ~ InvGamma(alpha1, alpha2)
        self.alpha1 = ALPHA1_DEFAULT
        self.alpha2 = ALPHA2_DEFAULT
        
        # g ~ Gamma(3/2, 3.9)
        self.beta1 = GAMMA_SHAPE_DEFAULT
        self.beta2 = GAMMA_RATE_G_AND_THETA_Y
        
        # theta ~ Gamma(3/2, b) with layer-specific rates
        # 3-layer: theta_y (outer), theta_q (middle), theta_r (inner)
        self.gamma1 = GAMMA_SHAPE_DEFAULT
        self.gamma2_y = GAMMA_RATE_G_AND_THETA_Y
        self.gamma2_q = GAMMA_RATE_THETA_Q
        self.gamma2_r = GAMMA_RATE_THETA_R
        
        # Proposal parameters
        self.l = 1.0
        self.u = 2.0
        
        # HMC parameters
        self.eps_hmc = HMC_EPS_DEFAULT
        self.T_step_hmc = HMC_T_STEPS_DEFAULT
        
        # Lambda prior: Gamma(5/2, 10/3)
        self.lambda_b1 = LAMBDA_GAMMA_SHAPE
        self.lambda_b2 = LAMBDA_GAMMA_RATE
        self.nu_lambda = _draw_lambda_prior(self.D)
        
        # Prior for M and V
        if self.prior_M is None:
            self.prior_M = np.zeros((self.p, self.D))
        if self.prior_V is None:
            self.prior_V = np.zeros((self.D, self.D))
    


    def _sample_R(self):
        """Sample latent layer R using ESS for 3-layer (D=1)."""
        Z = self.X @ self.W
        self.R = sample_R_3layer_ESS(
            Y=self.Q.flatten(),  # Q for likelihood in Q|R
            R_current=self.R,
            Q=Z,  # Z for prior in R|Z
            g_q=self.g_q,
            theta_q=self.theta_q,
            theta_r=self.theta_r,
            g_r=self.g_r,
            tau2_q=self.tau2_q,
            tau2_r=self.tau2_r,
            kernel_type=self.kernel_type
        )
    
    
    def _sample_Q(self):
        """Sample latent layer Q using ESS for 3-layer (D=1)."""
        self.Q = sample_Q_3layer_ESS(
            Q_current=self.Q,
            R=self.R,
            Y=self.Y,
            g_y=self.g_y,
            theta_y=self.theta_y,
            g_q=self.g_q,
            theta_q=self.theta_q,
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
            print(f"Running 3-Layer Deep GP Gibbs Sampler (D={self.D})")
            print("="*70)
            print(f"Iterations: {self.n_iterations}, Burn-in: {self.burn_in}, Thin: {self.thin}")
            print(f"Saved samples: {self.n_saved}")
            print(f"Use TensorFlow gradients: {self.use_tf}")
            print(f"Kernel type: {self.kernel_type}")
            print(f"Hyperparameter estimation (Y layer): "
                  f"tau2_y={'MLE' if self.use_mle_tau2 else 'MCMC'}, "
                  f"g_y={'MLE' if self.use_mle_g_y else 'MCMC'}, "
                  f"theta_D_y={'MLE' if self.use_mle_theta_y else 'MCMC'}")
            print("-"*70)
        
        for iter in range(self.n_iterations):

            # Step 1: sample M
            self.M = sample_M(self.W, self.Lambda, self.V, self.p,
                            prior_M=self.prior_M, M_prev=self.M,
                            mv_sampler=self.mv_sampler, rstiefel_rscol=self.rstiefel_rscol)

            # Step 2: sample V
            self.V = sample_V(self.W, self.Lambda, self.M, self.D,
                            prior_V=self.prior_V, V_prev=self.V,
                            mv_sampler=self.mv_sampler, rstiefel_rscol=self.rstiefel_rscol)

            # Step 3: sample Lambda
            self.nu_lambda = _draw_lambda_prior(self.D)
            self.Lambda = sample_Lambda_slice(
                self.Lambda, self.nu_lambda, self.M, self.V, self.W, self.p
            )
            
            # Step 4: sample W
            F_Wprior = self.M @ np.diag(self.Lambda) @ self.V.T
            self.W = sample_W_HMC_stiefel(
                self.Y, self.X, self.W,
                F_Wprior=F_Wprior, M=1, eps=self.eps_hmc, T_step=self.T_step_hmc,
                use_tf=self.use_tf, kernel_type=self.kernel_type,
                layer=3, Q=self.Q, R=self.R,
                tau2_y=self.tau2, tau2_q=self.tau2_q, tau2_r=self.tau2_r,
                theta_D_y=self.theta_y, theta_D_q=self.theta_q, theta_D_r=self.theta_r,
                g_y=self.g_y, g_q=self.g_q, g_r=self.g_r
            )
            
            # Step 5: sample tau2_r
            self.tau2_r = sample_tau2(
                self.R.flatten(), self.X, self.W, self.tau2_r,
                self.theta_r, self.g_r, self.alpha1, self.alpha2,
                kernel_type=self.kernel_type
            )

            # Step 6: sample g_r
            self.g_r = sample_g(
                self.R.flatten(), self.X, self.W, self.g_r,
                self.theta_r, self.tau2_r, self.beta1, self.beta2, self.l, self.u,
                kernel_type=self.kernel_type
            )
            
            # Step 7: sample theta_D_r (stored as theta_r)
            self.theta_r = sample_theta_D(
                self.R.flatten(), self.X, self.W, self.theta_r,
                self.tau2_r, self.g_r, self.gamma1, self.gamma2_r, self.l, self.u,
                kernel_type=self.kernel_type
            )
            
            # Step 8: sample R
            self._sample_R()
            
            # Step 9: sample tau2_q
            self.tau2_q = sample_tau2(
                self.Q.flatten(), np.eye(self.n), self.R, self.tau2_q,
                self.theta_q, self.g_q, self.alpha1, self.alpha2,
                kernel_type=self.kernel_type
            )

            # Step 10: sample g_q
            self.g_q = sample_g(
                self.Q.flatten(), np.eye(self.n), self.R, self.g_q,
                self.theta_q, self.tau2_q, self.beta1, self.beta2, self.l, self.u,
                kernel_type=self.kernel_type
            )
            
            # Step 11: sample theta_D_q (stored as theta_q)
            self.theta_q = sample_theta_D(
                self.Q.flatten(), np.eye(self.n), self.R, self.theta_q,
                self.tau2_q, self.g_q, self.gamma1, self.gamma2_q, self.l, self.u,
                kernel_type=self.kernel_type
            )
            
            # Step 12: sample Q
            self._sample_Q()
            
            # Steps 13-15: sample/update tau2_y, g_y, theta_D_y
            # Step 13: tau2_y
            if self.use_mle_tau2:
                self.tau2 = estimate_tau2_MLE(
                    self.Y, self.Q, np.eye(self.n), self.theta_y, self.g_y,
                    kernel_type=self.kernel_type
                )
            else:
                self.tau2 = sample_tau2(
                    self.Y, self.Q, np.eye(self.n), self.tau2,
                    self.theta_y, self.g_y, self.alpha1, self.alpha2,
                    kernel_type=self.kernel_type
                )
            
            # Step 14: g_y
            if self.use_mle_g_y:
                self.g_y = estimate_g_MLE(
                    self.Y, self.Q, np.eye(self.n), self.theta_y, self.tau2,
                    kernel_type=self.kernel_type
                )
            else:
                self.g_y = sample_g(
                    self.Y, self.Q, np.eye(self.n), self.g_y,
                    self.theta_y, self.tau2, self.beta1, self.beta2, self.l, self.u,
                    kernel_type=self.kernel_type
                )
            
            # Step 15: theta_D_y (stored as theta_y)
            if self.use_mle_theta_y:
                self.theta_y = estimate_theta_D_MLE(
                    self.Y, self.Q, np.eye(self.n), self.g_y, self.tau2,
                    kernel_type=self.kernel_type
                )
            else:
                self.theta_y = sample_theta_D(
                    self.Y, self.Q, np.eye(self.n), self.theta_y,
                    self.tau2, self.g_y, self.gamma1, self.gamma2_y, self.l, self.u,
                    kernel_type=self.kernel_type
                )
            
            
            # Save samples
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.tau2_samples[save_idx] = self.tau2
                self.tau2_q_samples[save_idx] = self.tau2_q
                self.tau2_r_samples[save_idx] = self.tau2_r
                self.g_y_samples[save_idx] = self.g_y
                self.g_q_samples[save_idx] = self.g_q
                self.g_r_samples[save_idx] = self.g_r
                self.theta_y_samples[save_idx] = self.theta_y
                self.theta_q_samples[save_idx] = self.theta_q
                self.theta_r_samples[save_idx] = self.theta_r
                self.W_samples[save_idx] = self.W
                self.Q_samples[save_idx] = self.Q
                self.R_samples[save_idx] = self.R
                self.M_samples[save_idx] = self.M
                self.V_samples[save_idx] = self.V
                self.Lambda_samples[save_idx] = self.Lambda
                save_idx += 1
            
            if verbose and (iter + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iter+1}/{self.n_iterations} | "
                      f"tau2_r={self.tau2_r:.4f}, g_r={self.g_r:.4f}, theta_D_r={self.theta_r:.4f} | "
                      f"tau2_q={self.tau2_q:.4f}, g_q={self.g_q:.4f}, theta_D_q={self.theta_q:.4f} | "
                      f"tau2_y={self.tau2:.4f}, g_y={self.g_y:.4f}, theta_D_y={self.theta_y:.4f} | "
                      f"Time: {elapsed:.1f}s")
        
        if verbose:
            print("-"*70)
            print(f"Sampling complete! Total time: {time.time() - start_time:.1f}s")
            print("="*70)
        
        return {
            'tau2_y': self.tau2_samples,
            'tau2_q': self.tau2_q_samples,
            'tau2_r': self.tau2_r_samples,
            'g_y': self.g_y_samples,
            'g_q': self.g_q_samples,
            'g_r': self.g_r_samples,
            'theta_y': self.theta_y_samples,
            'theta_q': self.theta_q_samples,
            'theta_r': self.theta_r_samples,
            'W': self.W_samples,
            'Q': self.Q_samples,
            'R': self.R_samples,
            'M': self.M_samples,
            'V': self.V_samples,
            'Lambda': self.Lambda_samples
        }


if __name__ == "__main__":
    print("="*70)
    print("Gibbs Sampler for 1, 2, 3-Layer Deep GP Models (D=1)")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n, p = 50, 10
    
    X = np.random.randn(n, p)
    W_true = np.random.randn(p, 1)
    W_true = W_true / np.linalg.norm(W_true)
    
    Z = X @ W_true
    theta = 1.0
    C = np.exp(-0.5 * np.sum((Z[:, np.newaxis, :] - Z[np.newaxis, :, :])**2, axis=2) / theta)
    Y = np.random.multivariate_normal(np.zeros(n), C + 0.01 * np.eye(n))
    
    print(f"\nTest Data: n={n}, p={p}")
    print("\nTesting 1-Layer Gibbs Sampler...")
    
    # Test 1-layer sampler
    sampler1 = GibbsSampler1Layer(Y, X, D=1, n_iterations=200, burn_in=50, thin=2)
    samples1 = sampler1.run(verbose=True)
    
    print(f"\n✓ 1-Layer sampler complete!")
    print(f"  Saved {len(samples1['tau2_y'])} samples")
    print(f"  Mean tau2_y: {np.mean(samples1['tau2_y']):.4f}")
    print(f"  Mean g_y: {np.mean(samples1['g_y']):.4f}")
    
    print("\n" + "="*70)
    print("All samplers tested successfully!")
    print("="*70)
