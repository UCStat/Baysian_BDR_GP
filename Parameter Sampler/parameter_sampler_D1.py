"""
Parameter Sampling Functions for GP with Bayesian Dimensionality Reduction (D=1)

This module implements MCMC sampling functions for all parameters in 1, 2, and 3-layer
Deep Gaussian Process models with dimensionality reduction to D=1.

Parameters sampled:
    - τ² (tau2): Observation noise variance
    - g: Nugget parameter
    - θ (theta_D): Lengthscale parameter(s)
    - W: Projection matrix on Stiefel manifold
    - M, V, Λ (Lambda): Matrix Langevin prior parameters
    
For multi-layer models, additional latent layer parameters (Q, R) and their hyperparameters.
"""

import numpy as np
from scipy.stats import invgamma, gamma, uniform
from scipy.special import gammaln, ive
from scipy.linalg import svd, qr, null_space
import warnings
from typing import Tuple, Optional, Dict, Union

# Add path for covariance module
import sys
from pathlib import Path
base_dir = Path(__file__).parent.parent
covar_path = str(base_dir / "Covariance Functions")
if covar_path not in sys.path:
    sys.path.insert(0, covar_path)

# Import from covariance kernel module (dynamic path, works at runtime)
from covariance_kernel_functions_and_gradients_W import (  # type: ignore[import]
    IsotropicSquaredExponentialKernel,
    SeparableSquaredExponentialKernel,
    IsotropicMatern32Kernel,
    SeparableMatern32Kernel
)

# Optional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Will use NumPy-only methods.")


# =============================================================================
# Kernel Selection Helper
# =============================================================================

def get_kernel_instance(kernel_type: str, theta_D: Union[float, np.ndarray], 
                        g: float, tau2: float, D: int = 1):
    """
    Create kernel instance based on kernel type.
    
    Args:
        kernel_type: One of 'isotropic_squared_exponential', 'separable_squared_exponential',
                     'isotropic_matern32', 'separable_matern32'
        theta_D: Lengthscale parameter(s)
        g: Nugget parameter
        tau2: Observation noise variance
        D: Reduced dimension (default: 1)
        
    Returns:
        Kernel instance
    """
    if kernel_type == 'isotropic_squared_exponential':
        theta = theta_D if np.isscalar(theta_D) else theta_D[0]
        return IsotropicSquaredExponentialKernel(lengthscale=theta, nugget=g, tau2=tau2)
    elif kernel_type == 'separable_squared_exponential':
        if np.isscalar(theta_D):
            theta_array = np.array([theta_D])
        else:
            theta_array = np.array(theta_D)
        return SeparableSquaredExponentialKernel(lengthscales=theta_array, nugget=g, tau2=tau2)
    elif kernel_type == 'isotropic_matern32':
        theta = theta_D if np.isscalar(theta_D) else theta_D[0]
        return IsotropicMatern32Kernel(lengthscale=theta, nugget=g, tau2=tau2)
    elif kernel_type == 'separable_matern32':
        if np.isscalar(theta_D):
            theta_array = np.array([theta_D])
        else:
            theta_array = np.array(theta_D)
        return SeparableMatern32Kernel(lengthscales=theta_array, nugget=g, tau2=tau2)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Must be one of: "
                        "'isotropic_squared_exponential', 'separable_squared_exponential', "
                        "'isotropic_matern32', 'separable_matern32'")


# =============================================================================
# Utility Functions for Stiefel Manifold Sampling
# =============================================================================

def NullC(M: np.ndarray) -> np.ndarray:
    """
    Compute the null space of a matrix M using QR decomposition.
    
    Args:
        M: Input matrix
        
    Returns:
        Orthonormal basis for the null space
    """
    if M.size == 0 or M.shape[1] == 0:
        return np.eye(M.shape[0])
    
    Q, R = qr(M, mode='full')
    rank = np.linalg.matrix_rank(M)
    
    if rank == 0:
        return np.eye(M.shape[0])
    else:
        return Q[:, rank:]


def rW(kap: float, m: int) -> float:
    """
    Simulate from the W distribution (Wood 1994).
    
    Args:
        kap: Concentration parameter
        m: Dimension
        
    Returns:
        Sample from W distribution
    """
    b = (-2 * kap + np.sqrt(4 * kap**2 + (m - 1)**2)) / (m - 1)
    x0 = (1 - b) / (1 + b)
    c = kap * x0 + (m - 1) * np.log(1 - x0**2)
    
    max_iter = 10000
    for _ in range(max_iter):
        Z = np.random.beta((m - 1) / 2, (m - 1) / 2)
        W = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
        U = np.random.uniform(0, 1)
        if kap * W + (m - 1) * np.log(1 - x0 * W) - c >= np.log(U):
            return W
    
    # Fallback
    return 0.0


def rmf_vector(kmu: np.ndarray) -> np.ndarray:
    """
    Simulate from the vector multivariate Fisher (MF) distribution (Wood 1994).
    
    Args:
        kmu: Input vector (k * mu)
        
    Returns:
        Simulated vector from MF distribution
    """
    kap = np.linalg.norm(kmu)
    mu = kmu / kap if kap != 0 else np.zeros_like(kmu)
    m = len(mu)
    
    if kap == 0:
        u = np.random.normal(size=m)
        u /= np.linalg.norm(u)
        return u.reshape(m, 1)
    
    if m == 1:
        prob = 1 / (1 + np.exp(2 * kap * mu))
        u = (-1) ** np.random.binomial(1, prob)
        return np.array([u])
    
    if m > 1:
        W = rW(kap, m)
        V = np.random.normal(size=m - 1)
        V /= np.linalg.norm(V)
        
        x = np.concatenate(((1 - W**2)**0.5 * V, [W]))
        u = np.hstack((NullC(mu.reshape(-1, 1)), mu.reshape(-1, 1))) @ x
        return u


def rmf_matrix(M: np.ndarray) -> np.ndarray:
    """
    Sample from matrix von Mises-Fisher distribution.
    
    Args:
        M: Parameter matrix
        
    Returns:
        Sampled matrix on Stiefel manifold
    """
    if M.shape[1] == 1:
        return rmf_vector(M[:, 0]).reshape(-1, 1)
    
    U, S, Vt = svd(M, full_matrices=False)
    H = U @ np.diag(S)
    m, R = H.shape
    cmet = False
    
    while not cmet:
        U_sample = np.zeros((m, R))
        U_sample[:, 0] = rmf_vector(H[:, 0]).flatten()
        lr = 0
        
        for j in range(1, R):
            N = NullC(U_sample[:, :j].T)
            NH = N.T @ H[:, j]
            xx = rmf_vector(np.array(NH).reshape(-1, 1))
            Nx = N @ xx.reshape(-1, 1)
            U_sample[:, j] = Nx.flatten()
            
            if S[j] > 0:
                xn = np.linalg.norm(N.T @ H[:, j])
                xd = np.linalg.norm(H[:, j])
                lbr = (
                    np.log(ive(0.5 * (m - j - 1), xn)) -
                    np.log(ive(0.5 * (m - j - 1), xd))
                )
                lbr = 0.5 * (np.log(xd) - np.log(xn)) if np.isnan(lbr) else lbr
                lr += lbr + (xn - xd) + 0.5 * (m - j - 1) * (np.log(xd) - np.log(xn))
        
        cmet = np.log(np.random.uniform()) < lr
    
    XX = U_sample @ Vt
    return XX


def rmf_matrix_gibbs(M: np.ndarray, Xn: np.ndarray, rscol: Optional[int] = None) -> np.ndarray:
    """
    Gibbs sampling for the matrix-variate von Mises-Fisher distribution.
    
    Args:
        M: Parameter matrix
        Xn: Previous sample (current state)
        rscol: Number of columns to resample (default: based on log dimension)
        
    Returns:
        New sample from MF distribution
    """
    if rscol is None:
        rscol = max(2, min(round(np.log(M.shape[0])), M.shape[1]))
    
    U, S, Vt = svd(M, full_matrices=False)
    H = U[:, :M.shape[1]] @ np.diag(S)
    Yn = Xn @ Vt
    m, R = H.shape
    
    for _ in range(round(R / rscol)):
        rn = np.random.choice(np.arange(R), rscol, replace=False)
        Nn = NullC(np.delete(Yn, -rn, axis=1))
        yn = rmf_matrix(Nn.T @ H[:, rn])
        Yn[:, rn] = Nn @ yn
    
    return Yn @ Vt


def rmf_vectorN(kmu: np.ndarray) -> np.ndarray:
    """
    Simulate from the vector multivariate Fisher (MF) distribution (for D>1 cases).
    
    Args:
        kmu: Input vector (k * mu)
        
    Returns:
        Simulated vector from MF distribution
    """
    kap = np.linalg.norm(kmu)
    mu = kmu / kap if kap != 0 else np.zeros_like(kmu)
    m = len(mu)
    
    if kap == 0:
        u = np.random.normal(size=m)
        u /= np.linalg.norm(u)
        return u.reshape(m, 1)
    
    if m == 1:
        prob = 1 / (1 + np.exp(2 * kap * mu))
        u = (-1) ** np.random.binomial(1, prob)
        return np.array([u])
    
    if m > 1:
        W = rW(kap, m)
        V = np.random.normal(size=m - 1)
        V /= np.linalg.norm(V)
        x = np.concatenate(((1 - W**2)**0.5 * V, [W]))
        u = np.hstack((NullC(mu.reshape(-1, 1)), mu.reshape(-1, 1))) @ x
        return u


def rmf_matrixN(M: np.ndarray) -> np.ndarray:
    """
    Sample from matrix von Mises-Fisher distribution (for D>1 cases).
    
    Args:
        M: Parameter matrix
        
    Returns:
        Sampled matrix on Stiefel manifold
    """
    if M.shape[1] == 1:
        XX = rmf_vectorN(M[:, 0]).reshape(-1, 1)
        return XX
    else:
        U, S, Vt = svd(M, full_matrices=False)
        H = U @ np.diag(S)
        m, R = H.shape
        cmet = False
        
        U_sample = np.zeros((m, R))
        
        while not cmet:
            U_sample[:, 0] = rmf_vectorN(H[:, 0]).flatten()
            lr = 0
            
            for j in range(1, R):
                N = null_space(U_sample[:, :j].T)
                xx = rmf_vectorN(N.T @ H[:, j])
                Nx = N @ xx.reshape(-1, 1)
                U_sample[:, j] = Nx.flatten()
                
                if S[j] > 0:
                    xn = np.linalg.norm(N.T @ H[:, j])
                    xd = np.linalg.norm(H[:, j])
                    lbr = (
                        np.log(ive(0.5 * (m - j - 1), xn)) -
                        np.log(ive(0.5 * (m - j - 1), xd))
                    )
                    lbr = 0.5 * (np.log(xd) - np.log(xn)) if np.isnan(lbr) else lbr
                    lr += lbr + (xn - xd) + 0.5 * (m - j - 1) * (np.log(xd) - np.log(xn))
            
            cmet = np.log(np.random.uniform()) < lr
        
        XX = U_sample @ Vt
        return XX


def rmf_matrix_gibbsN(M: np.ndarray, Xn: np.ndarray, rscol: Optional[int] = None) -> np.ndarray:
    """
    Gibbs sampling for the matrix-variate von Mises-Fisher distribution (for D>1 cases).
    
    Args:
        M: Parameter matrix
        Xn: Previous sample (current state)
        rscol: Number of columns to resample (default: based on log dimension)
        
    Returns:
        New sample from MF distribution
    """
    if rscol is None:
        rscol = max(2, min(round(np.log(M.shape[0])), M.shape[1]))
    
    U, S, Vt = svd(M, full_matrices=False)
    H = U[:, :M.shape[1]] @ np.diag(S)
    Yn = Xn @ Vt
    m, R = H.shape
    
    for _ in range(round(R / rscol)):
        rn = np.random.choice(np.arange(R), rscol, replace=False)
        Nn = NullC(np.delete(Yn, -rn, axis=1))
        yn = rmf_matrixN(Nn.T @ H[:, rn])
        Yn[:, rn] = Nn @ yn
    
    return Yn @ Vt


# =============================================================================
# Covariance Functions
# =============================================================================

def covar_sep(Z: np.ndarray, theta: Union[float, np.ndarray], g: float) -> np.ndarray:
    """
    Compute separable squared exponential covariance matrix.
    
    Args:
        Z: Reduced inputs (n, D)
        theta: Lengthscale parameter(s)
        g: Nugget parameter
        
    Returns:
        Covariance matrix Σ = C + g*I
    """
    n = Z.shape[0]
    D = Z.shape[1]
    
    if D == 1:
        theta = np.array([theta]) if np.isscalar(theta) else theta
    
    C_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sum_term = sum((Z[i, k] - Z[j, k])**2 / (2 * theta[k]**2) for k in range(D))
            C_matrix[i, j] = np.exp(-sum_term)
    
    Sigma = C_matrix + g * np.eye(n)
    return Sigma


def log_likelihood_gp(g: float, Z: np.ndarray, Y: np.ndarray, 
                     theta_D: Union[float, np.ndarray], tau2: float,
                     kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Compute log-likelihood for GP model.
    
    Args:
        g: Nugget parameter
        Z: Reduced inputs (n, D)
        Y: Response vector (n,)
        theta_D: Lengthscale parameter(s)
        tau2: Observation noise variance
        kernel_type: Type of kernel to use
        
    Returns:
        Log-likelihood value
    """
    # Use kernel-based computation
    D = Z.shape[1] if len(Z.shape) > 1 else 1
    kernel = get_kernel_instance(kernel_type, theta_D, g, tau2, D)
    return kernel.log_likelihood(Y, Z)


# =============================================================================
# MLE Estimation Functions
# =============================================================================

def estimate_tau2_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                     theta_D: Union[float, np.ndarray], g: float,
                     kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Estimate τ² using Maximum Likelihood Estimation.
    
    MLE estimate:
        τ²_MLE = (1/n) * Y^T Σ^{-1} Y
    
    where Σ = C + g*I (covariance without τ²)
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p) OR latent space Z/Q/R (n, D)
        W: Projection matrix (p, D) OR identity matrix (n, n) for multi-layer
        theta_D: Lengthscale parameter(s)
        g: Nugget parameter
        kernel_type: Type of kernel to use
        
    Returns:
        MLE estimate of τ²
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    # If W is identity matrix (n, n), input_matrix is already Q or R
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        # Ensure Q/R is 2D: (n, 1) for D=1
        if input_matrix.ndim == 1:
            Z = input_matrix.reshape(-1, 1)
        else:
            Z = input_matrix
    else:
        # Single-layer case: compute Z = X @ W
        Z = input_matrix @ W
        # Ensure Z is 2D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    
    # Use kernel to compute covariance (without tau2)
    D = Z.shape[1] if len(Z.shape) > 1 else 1
    kernel = get_kernel_instance(kernel_type, theta_D, g, tau2=1.0, D=D)
    Sigma = kernel.compute_sigma(Z) / 1.0  # Remove tau2 scaling
    
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        # Fallback: use previous value or default
        return 1.0
    
    # MLE estimate
    tau2_mle = (Y.T @ Sigma_inv @ Y) / n
    
    # Ensure positivity
    tau2_mle = max(tau2_mle, 1e-6)
    
    return float(tau2_mle)


def estimate_g_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                  theta_D: Union[float, np.ndarray], tau2: float,
                  bounds: Tuple[float, float] = (1e-6, 0.1),
                  n_grid: int = 50,
                  kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Estimate nugget parameter g using Maximum Likelihood Estimation.
    
    Uses grid search over specified bounds to find MLE.
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p)
        W: Projection matrix (p, D)
        theta_D: Lengthscale parameter(s)
        tau2: Observation noise variance
        bounds: (lower, upper) bounds for g
        n_grid: Number of grid points to evaluate
        kernel_type: Type of kernel to use
        
        Returns:
        MLE estimate of g
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        if input_matrix.ndim == 1:
            Z = input_matrix.reshape(-1, 1)
        else:
            Z = input_matrix
    else:
        # Single-layer case: compute Z = X @ W
        Z = input_matrix @ W
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    
    # Grid search
    g_grid = np.linspace(bounds[0], bounds[1], n_grid)
    log_liks = np.zeros(n_grid)
    
    for i, g_val in enumerate(g_grid):
        log_liks[i] = log_likelihood_gp(g_val, Z, Y, theta_D, tau2, kernel_type)
    
    # Find maximum
    idx_max = np.argmax(log_liks)
    g_mle = g_grid[idx_max]
    
    return float(g_mle)


def estimate_theta_D_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                        g: float, tau2: float,
                        bounds: Tuple[float, float] = (0.01, 10.0),
                        n_grid: int = 50,
                        kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Estimate lengthscale θ using Maximum Likelihood Estimation.
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p) OR latent space Z/Q/R (n, D)
        W: Projection matrix (p, D) OR identity matrix (n, n) for multi-layer
        g: Nugget parameter
        tau2: Observation noise variance
        bounds: Search bounds for theta
        n_grid: Number of grid points
        kernel_type: Type of kernel to use
        
    Returns:
        MLE estimate of θ
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        if input_matrix.ndim == 1:
            Z = input_matrix.reshape(-1, 1)
        else:
            Z = input_matrix
    else:
        # Single-layer case: compute Z = X @ W
        Z = input_matrix @ W
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    
    # Grid search
    theta_grid = np.linspace(bounds[0], bounds[1], n_grid)
    log_liks = np.zeros(n_grid)
    
    for i, theta_val in enumerate(theta_grid):
        log_liks[i] = log_likelihood_gp(g, Z, Y, theta_val, tau2, kernel_type)
    
    # Find maximum
    idx_max = np.argmax(log_liks)
    theta_mle = theta_grid[idx_max]
    
    return float(theta_mle)


def estimate_all_hyperparameters_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                                     tau2_init: float = 1.0, g_init: float = 0.01, 
                                     theta_init: float = 1.0,
                                     n_iterations: int = 5,
                                     g_bounds: Tuple[float, float] = (1e-6, 0.1),
                                     theta_bounds: Tuple[float, float] = (0.01, 10.0),
                                     n_grid: int = 50,
                                     verbose: bool = False,
                                     kernel_type: str = 'isotropic_squared_exponential') -> Dict[str, float]:
    """
    Jointly estimate τ², g, and θ using iterative MLE (coordinate ascent).
    
    Alternates between estimating each parameter while holding others fixed
    until convergence or max iterations.
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p)
        W: Projection matrix (p, D)
        tau2_init: Initial τ² value
        g_init: Initial g value
        theta_init: Initial θ value
        n_iterations: Maximum number of iterations
        g_bounds: Bounds for g
        theta_bounds: Bounds for θ
        n_grid: Grid points for MLE search
        verbose: Print iteration details
        kernel_type: Type of kernel to use
        
    Returns:
        Dictionary with MLE estimates: {'tau2': ..., 'g': ..., 'theta_D': ...}
    """
    tau2 = tau2_init
    g = g_init
    theta_D = theta_init
    
    if verbose:
        print("Iterative MLE Estimation:")
        print(f"{'Iter':<6} {'tau2':<12} {'g':<12} {'theta':<12} {'log_lik':<12}")
        print("-" * 54)
    
    for iter in range(n_iterations):
        # Update tau2
        tau2 = estimate_tau2_MLE(Y, input_matrix, W, theta_D, g, kernel_type)
        
        # Update g
        g = estimate_g_MLE(Y, input_matrix, W, theta_D, tau2, g_bounds, n_grid, kernel_type)
        
        # Update theta_D
        theta_D = estimate_theta_D_MLE(Y, input_matrix, W, g, tau2, theta_bounds, n_grid, kernel_type)
        
        if verbose:
            Z = input_matrix @ W
            log_lik = log_likelihood_gp(g, Z, Y, theta_D, tau2, kernel_type)
            print(f"{iter+1:<6} {tau2:<12.6f} {g:<12.6f} {theta_D:<12.6f} {log_lik:<12.2f}")
    
    if verbose:
        print("-" * 54)
        print("MLE estimation complete!\n")
    
    return {
        'tau2': tau2,
        'g': g,
        'theta_D': theta_D
    }


# =============================================================================
# Parameter Sampling Functions (MCMC)
# =============================================================================

def sample_tau2(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
               tau2_curr: float, theta_D: Union[float, np.ndarray], g: float,
               alpha1: float = 1.0, alpha2: float = 1000.0,
               kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Sample τ² (observation noise variance) from inverse-gamma posterior.
    
    Full conditional:
        τ² | Y, Z, θ, g ~ InvGamma(α₁ + n/2, α₂ + Y^T Σ^{-1} Y / 2)
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p) OR latent space Z/Q/R (n, D)
        W: Projection matrix (p, D) OR identity matrix (n, n) for multi-layer
        tau2_curr: Current τ² value
        theta_D: Lengthscale parameter(s)
        g: Nugget parameter
        alpha1: Prior shape parameter
        alpha2: Prior scale parameter
        kernel_type: Type of kernel to use
        
    Returns:
        Sampled τ² value
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    # If W is identity matrix (n, n), input_matrix is already Q or R
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        # Ensure Q/R is 2D: (n, 1) for D=1
        if input_matrix.ndim == 1:
            Z = input_matrix.reshape(-1, 1)
        else:
            Z = input_matrix
    else:
        # Single-layer case: compute Z = X @ W
        Z = input_matrix @ W
        # Ensure Z is 2D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    
    # Use kernel to compute covariance (without tau2)
    D = Z.shape[1] if len(Z.shape) > 1 else 1
    kernel = get_kernel_instance(kernel_type, theta_D, g, tau2=1.0, D=D)
    Sigma = kernel.compute_sigma(Z) / 1.0  # Remove tau2 scaling
    
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return tau2_curr  # Keep current if matrix is singular
    
    # Posterior parameters
    alpha_post = alpha1 + n / 2
    beta_post = alpha2 + 0.5 * (Y.T @ Sigma_inv @ Y)
    
    # Sample from inverse-gamma
    tau2_sample = invgamma.rvs(alpha_post, scale=beta_post)
    
    return tau2_sample[0] if isinstance(tau2_sample, np.ndarray) else tau2_sample


def sample_g(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
            g_curr: float, theta_D: Union[float, np.ndarray], tau2: float,
            beta1: float = 0.01, beta2: float = 0.005,
            l: float = 1.0, u: float = 2.0,
            kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Sample nugget parameter g using Metropolis-Hastings.
    
    Full conditional:
        p(g | Y, Z, θ, τ²) ∝ L(Y | Z, g, θ, τ²) × Gamma(g | β₁, β₂)
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p)
        W: Projection matrix (p, D)
        g_curr: Current g value
        theta_D: Lengthscale parameter(s)
        tau2: Observation noise variance
        beta1: Prior shape parameter
        beta2: Prior rate parameter
        l, u: Proposal bounds (uniform on [l*g/u, u*g/l])
        kernel_type: Type of kernel to use
        
    Returns:
        Sampled g value
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        if input_matrix.ndim == 1:
            Z = input_matrix.reshape(-1, 1)
        else:
            Z = input_matrix
    else:
        # Single-layer case: compute Z = X @ W
        Z = input_matrix @ W
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    
    # Propose new g
    g_prop = np.random.uniform((l * g_curr) / u, (u * g_curr) / l)
    
    # Ensure positivity
    if g_prop <= 0 or g_prop > 0.1:
        return g_curr
    
    # Compute acceptance ratio
    ru = np.random.uniform(0, 1)
    eps = np.sqrt(np.finfo(float).eps)
    
    # Log-likelihood current
    log_lik_curr = log_likelihood_gp(g_curr, Z, Y, theta_D, tau2, kernel_type)
    
    # Log-prior current (Gamma)
    log_prior_curr = (beta1 - 1) * np.log(g_curr - eps) - beta2 * (g_curr - eps)
    
    # Log-likelihood proposed
    log_lik_prop = log_likelihood_gp(g_prop, Z, Y, theta_D, tau2, kernel_type)
    
    # Log-prior proposed
    log_prior_prop = (beta1 - 1) * np.log(g_prop - eps) - beta2 * (g_prop - eps)
    
    # Acceptance threshold
    lpost_curr = log_lik_curr + log_prior_curr + np.log(ru) - np.log(g_curr) + np.log(g_prop)
    lpost_prop = log_lik_prop + log_prior_prop
    
    # Accept/reject
    if lpost_prop > lpost_curr:
        return g_prop
    else:
        return g_curr


def sample_theta_D(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                  theta_D_curr: Union[float, np.ndarray], tau2: float, g: float,
                  gamma1: float = 0.01, gamma2: float = 0.01/3,
                  l: float = 1.0, u: float = 2.0,
                  kernel_type: str = 'isotropic_squared_exponential') -> Union[float, np.ndarray]:
    """
    Sample lengthscale θ using Metropolis-Hastings.
    
    Full conditional:
        p(θ | Y, Z, g, τ²) ∝ L(Y | Z, θ, g, τ²) × Gamma(θ | γ₁, γ₂)
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p)
        W: Projection matrix (p, D)
        theta_D_curr: Current θ value(s)
        tau2: Observation noise variance
        g: Nugget parameter
        gamma1: Prior shape parameter
        gamma2: Prior rate parameter
        l, u: Proposal bounds
        kernel_type: Type of kernel to use
        
    Returns:
        Sampled θ value(s)
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        if input_matrix.ndim == 1:
            Z = input_matrix.reshape(-1, 1)
        else:
            Z = input_matrix
    else:
        # Single-layer case: compute Z = X @ W
        Z = input_matrix @ W
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    
    # Propose new theta_D
    theta_D_prop = np.random.uniform((l * theta_D_curr) / u, (u * theta_D_curr) / l)
    
    # Ensure positivity
    if np.any(theta_D_prop <= 0) or np.any(theta_D_prop > 20):
        return theta_D_curr
    
    # Compute acceptance ratio
    ru = np.random.uniform(0, 1)
    eps = np.sqrt(np.finfo(float).eps)
    
    # Log-likelihood current
    log_lik_curr = log_likelihood_gp(g, Z, Y, theta_D_curr, tau2, kernel_type)
    
    # Log-prior current (Gamma)
    if np.isscalar(theta_D_curr):
        log_prior_curr = (gamma1 - 1) * np.log(theta_D_curr - eps) - gamma2 * (theta_D_curr - eps)
        log_prior_prop = (gamma1 - 1) * np.log(theta_D_prop - eps) - gamma2 * (theta_D_prop - eps)
    else:
        log_prior_curr = np.sum((gamma1 - 1) * np.log(theta_D_curr - eps) - gamma2 * (theta_D_curr - eps))
        log_prior_prop = np.sum((gamma1 - 1) * np.log(theta_D_prop - eps) - gamma2 * (theta_D_prop - eps))
    
    # Log-likelihood proposed
    log_lik_prop = log_likelihood_gp(g, Z, Y, theta_D_prop, tau2, kernel_type)
    
    # Acceptance threshold
    lpost_curr = log_lik_curr + log_prior_curr + np.log(ru) - np.log(theta_D_curr) + np.log(theta_D_prop)
    lpost_prop = log_lik_prop + log_prior_prop
    
    # Accept/reject
    if lpost_prop > lpost_curr:
        return theta_D_prop
    else:
        return theta_D_curr


# =============================================================================
# W Sampling using HMC on Stiefel Manifold
# =============================================================================

def loglik_and_gradW_numpy(Y: np.ndarray, X: np.ndarray, W: np.ndarray,
                          F_Wprior: Optional[np.ndarray] = None,
                          use_tf: bool = False,
                          kernel_type: str = 'isotropic_squared_exponential',
                          layer: int = 1,
                          Q: Optional[np.ndarray] = None,
                          R: Optional[np.ndarray] = None,
                          tau2_y: Optional[float] = None,
                          tau2_q: Optional[float] = None,
                          tau2_r: Optional[float] = None,
                          theta_D_y: Optional[Union[float, np.ndarray]] = None,
                          theta_D_q: Optional[Union[float, np.ndarray]] = None,
                          theta_D_r: Optional[Union[float, np.ndarray]] = None,
                          g_y: Optional[float] = None,
                          g_q: Optional[float] = None,
                          g_r: Optional[float] = None) -> Tuple[float, np.ndarray]:
    """
    Compute log-posterior and gradient with respect to W.
    
    Supports hierarchical structures:
      layer=1: log p(Y | XW) + log p(W)
      layer=2: log p(Y | Q) + log p(Q | XW) + log p(W)
      layer=3: log p(Y | Q) + log p(Q | R) + log p(R | XW) + log p(W)
    
    Required hyperparameters by layer:
      - layer=1: tau2_y, theta_D_y, g_y
      - layer=2: tau2_y, theta_D_y, g_y, tau2_q, theta_D_q, g_q
      - layer=3: tau2_y, theta_D_y, g_y, tau2_q, theta_D_q, g_q, tau2_r, theta_D_r, g_r
    
    For D=1, gradients w.r.t. W are driven by the innermost W-dependent term.
    """
    p, D = W.shape
    
    def _to_scalar(val: Optional[Union[float, np.ndarray]], name: str, default: Optional[float] = None) -> float:
        if val is None:
            if default is None:
                raise ValueError(f"Missing required parameter: {name}")
            return float(default)
        arr = np.asarray(val, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Parameter {name} is empty.")
        return float(arr[0])
    
    def _to_col(arr: np.ndarray, name: str) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.ndim != 2 or out.shape[1] != 1:
            raise ValueError(f"{name} must have shape (n,) or (n,1) for D=1. Got {out.shape}.")
        return out
    
    def _kernel_gradient(kernel, response: np.ndarray, X_mat: np.ndarray, W_mat: np.ndarray) -> np.ndarray:
        response_vec = np.asarray(response, dtype=float).reshape(-1)
        if use_tf and TF_AVAILABLE:
            y_tf = tf.constant(response_vec, dtype=tf.float64)
            x_tf = tf.constant(X_mat, dtype=tf.float64)
            w_tf = tf.Variable(W_mat, dtype=tf.float64)
            return kernel.gradient_log_likelihood_W_tf(y_tf, x_tf, w_tf).numpy()
        return kernel.gradient_log_likelihood_W(response_vec, X_mat, W_mat)
    
    Y_vec = np.asarray(Y, dtype=float).reshape(-1)
    Z = X @ W
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    if layer == 1:
        tau2_y_eff = _to_scalar(tau2_y, "tau2_y")
        if theta_D_y is None:
            raise ValueError("layer=1 requires theta_D_y.")
        theta_y_eff = theta_D_y
        g_y_eff = _to_scalar(g_y, "g_y")
        kernel = get_kernel_instance(kernel_type, theta_y_eff, g_y_eff, tau2_y_eff, D)
        log_lik = kernel.log_likelihood(Y_vec, Z)
        grad_loglik = _kernel_gradient(kernel, Y_vec, X, W)
    
    elif layer == 2:
        if Q is None:
            raise ValueError("layer=2 requires Q.")
        tau2_y_eff = _to_scalar(tau2_y, "tau2_y")
        tau2_q_eff = _to_scalar(tau2_q, "tau2_q")
        if theta_D_y is None or theta_D_q is None:
            raise ValueError("layer=2 requires theta_D_y and theta_D_q.")
        theta_y_eff = theta_D_y
        theta_q_eff = theta_D_q
        g_y_eff = _to_scalar(g_y, "g_y")
        g_q_eff = _to_scalar(g_q, "g_q")
        Q_col = _to_col(Q, "Q")
        kernel_y = get_kernel_instance(kernel_type, theta_y_eff, g_y_eff, tau2_y_eff, D)
        kernel_q = get_kernel_instance(kernel_type, theta_q_eff, g_q_eff, tau2_q_eff, D)
        
        log_lik_y = kernel_y.log_likelihood(Y_vec, Q_col)
        log_lik_q = kernel_q.log_likelihood(Q_col.reshape(-1), Z)
        log_lik = log_lik_y + log_lik_q
        grad_loglik = _kernel_gradient(kernel_q, Q_col.reshape(-1), X, W)
    
    elif layer == 3:
        if Q is None or R is None:
            raise ValueError("layer=3 requires both Q and R.")
        tau2_y_eff = _to_scalar(tau2_y, "tau2_y")
        tau2_q_eff = _to_scalar(tau2_q, "tau2_q")
        tau2_r_eff = _to_scalar(tau2_r, "tau2_r")
        if theta_D_y is None or theta_D_q is None or theta_D_r is None:
            raise ValueError("layer=3 requires theta_D_y, theta_D_q, and theta_D_r.")
        theta_y_eff = theta_D_y
        theta_q_eff = theta_D_q
        theta_r_eff = theta_D_r
        g_y_eff = _to_scalar(g_y, "g_y")
        g_q_eff = _to_scalar(g_q, "g_q")
        g_r_eff = _to_scalar(g_r, "g_r")
        Q_col = _to_col(Q, "Q")
        R_col = _to_col(R, "R")
        
        kernel_y = get_kernel_instance(kernel_type, theta_y_eff, g_y_eff, tau2_y_eff, D)
        kernel_q = get_kernel_instance(kernel_type, theta_q_eff, g_q_eff, tau2_q_eff, D)
        kernel_r = get_kernel_instance(kernel_type, theta_r_eff, g_r_eff, tau2_r_eff, D)
        
        log_lik_y = kernel_y.log_likelihood(Y_vec, Q_col)
        log_lik_q = kernel_q.log_likelihood(Q_col.reshape(-1), R_col)
        log_lik_r = kernel_r.log_likelihood(R_col.reshape(-1), Z)
        log_lik = log_lik_y + log_lik_q + log_lik_r
        grad_loglik = _kernel_gradient(kernel_r, R_col.reshape(-1), X, W)
    
    else:
        raise ValueError(f"Unknown layer={layer}. Expected 1, 2, or 3.")
    
    # Matrix Langevin prior
    if F_Wprior is not None:
        log_prior = np.trace(F_Wprior.T @ W)
        grad_logprior = F_Wprior
    else:
        log_prior = 0.0
        grad_logprior = np.zeros((p, D))
    
    # Total log-posterior and gradient
    log_post = log_lik + log_prior
    grad_logpost = grad_loglik + grad_logprior
    
    return log_post, grad_logpost


def sample_W_HMC_stiefel(Y: np.ndarray, X: np.ndarray, W: np.ndarray,
                        F_Wprior: Optional[np.ndarray] = None,
                        M: int = 1, eps: float = 0.001, T_step: int = 17,
                        use_tf: bool = False,
                        kernel_type: str = 'isotropic_squared_exponential',
                        layer: int = 1,
                        Q: Optional[np.ndarray] = None,
                        R: Optional[np.ndarray] = None,
                        tau2_y: Optional[float] = None,
                        tau2_q: Optional[float] = None,
                        tau2_r: Optional[float] = None,
                        theta_D_y: Optional[Union[float, np.ndarray]] = None,
                        theta_D_q: Optional[Union[float, np.ndarray]] = None,
                        theta_D_r: Optional[Union[float, np.ndarray]] = None,
                        g_y: Optional[float] = None,
                        g_q: Optional[float] = None,
                        g_r: Optional[float] = None) -> np.ndarray:
    """
    Sample W from Stiefel manifold using Hamiltonian Monte Carlo with geodesic flows.
    
    This implements HMC on the Stiefel manifold St(p, D) = {W ∈ R^{p×D} : W^T W = I_D}.
    
    Args:
        Y: Response vector (n,)
        X: Design matrix (n, p)
        W: Current projection matrix (p, D)
        F_Wprior: Prior parameter F = M @ Λ
        M: Number of HMC samples to draw
        eps: Step size
        T_step: Number of leapfrog steps
        use_tf: Use TensorFlow for gradients
        kernel_type: Type of kernel to use
        layer: Hierarchical layer (1, 2, or 3)
        Q: Latent Q (required for layers 2/3)
        R: Latent R (required for layer 3)
        tau2_y, tau2_q, tau2_r: Layer-specific variances
        theta_D_y, theta_D_q, theta_D_r: Layer-specific lengthscales
        g_y, g_q, g_r: Layer-specific nuggets
            layer 1 requires: tau2_y, theta_D_y, g_y
            layer 2 additionally requires: tau2_q, theta_D_q, g_q
            layer 3 additionally requires: tau2_r, theta_D_r, g_r
        
    Returns:
        Sampled W matrix (or list if M > 1)
    """
    p, D = W.shape
    hmc_samples = []
    nn = 0
    
    if D > 1:
        # Multi-dimensional case
        max_hmc_iterations = 1000  # Prevent infinite loop
        iteration_count = 0
        
        while nn < M and iteration_count < max_hmc_iterations:
            iteration_count += 1
            # Sample momentum
            u = np.random.normal(size=(p, D))
            u_proj = (np.eye(p) - W @ W.T) @ u
            
            # Compute current log-posterior and gradient
            logPosteriorW, grad_logPosteriorW = loglik_and_gradW_numpy(
                Y, X, W, F_Wprior, use_tf, kernel_type,
                layer=layer, Q=Q, R=R,
                tau2_y=tau2_y, tau2_q=tau2_q, tau2_r=tau2_r,
                theta_D_y=theta_D_y, theta_D_q=theta_D_q, theta_D_r=theta_D_r,
                g_y=g_y, g_q=g_q, g_r=g_r
            )
            
            H_old = logPosteriorW - np.linalg.norm(u_proj.flatten())**2 / 2.0
            
            W_star = W.copy()
            
            # Leapfrog integration
            for i in range(T_step):
                # Half step for momentum
                logPosteriorW, grad_logPosteriorW = loglik_and_gradW_numpy(
                    Y, X, W_star, F_Wprior, use_tf, kernel_type,
                    layer=layer, Q=Q, R=R,
                    tau2_y=tau2_y, tau2_q=tau2_q, tau2_r=tau2_r,
                    theta_D_y=theta_D_y, theta_D_q=theta_D_q, theta_D_r=theta_D_r,
                    g_y=g_y, g_q=g_q, g_r=g_r
                )
                u_new = u_proj + eps * grad_logPosteriorW / 2
                u_new_proj = (np.eye(p) - W_star @ W_star.T) @ u_new
                
                # Geodesic flow on Stiefel manifold
                U, S, Vt = svd(u_new_proj, full_matrices=False)
                alpha_vals = S
                
                V_0 = np.hstack([W_star, U])
                rot_block = np.zeros((2 * D, D))
                for d in range(D):
                    c, s = np.cos(alpha_vals[d] * eps), np.sin(alpha_vals[d] * eps)
                    rot_block[d, d] = c
                    rot_block[D + d, d] = s
                
                V_eps = V_0 @ rot_block
                W_star = V_eps[:, :D].copy()
                u_new = V_eps[:, D:].copy()
                
                # Half step for momentum
                logPosteriorW, grad_logPosteriorW = loglik_and_gradW_numpy(
                    Y, X, W_star, F_Wprior, use_tf, kernel_type,
                    layer=layer, Q=Q, R=R,
                    tau2_y=tau2_y, tau2_q=tau2_q, tau2_r=tau2_r,
                    theta_D_y=theta_D_y, theta_D_q=theta_D_q, theta_D_r=theta_D_r,
                    g_y=g_y, g_q=g_q, g_r=g_r
                )
                u_new = u_new + eps * grad_logPosteriorW / 2
                u_new_proj = (np.eye(p) - W_star @ W_star.T) @ u_new
                u_proj = u_new_proj.copy()
            
            # Compute new Hamiltonian
            H_new = logPosteriorW - np.linalg.norm(u_new_proj.flatten())**2 / 2.0
            
            # Accept/reject with numerical stability
            delta_H = H_new - H_old
            if delta_H > 10:  # Prevent overflow
                accept_prob = 0.0
            elif delta_H < -10:  # Always accept
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta_H)
            
            if np.random.uniform() < accept_prob:
                W = W_star.copy()
                hmc_samples.append(W)
                nn += 1
        
        # If we didn't get enough samples, use the last W
        if nn < M:
            warnings.warn(f"HMC: Only generated {nn}/{M} samples after {max_hmc_iterations} iterations. Using last W.")
            while len(hmc_samples) < M:
                hmc_samples.append(W.copy())
    
    elif D == 1:
        # 1D case (vector on sphere)
        max_hmc_iterations = 1000  # Prevent infinite loop
        iteration_count = 0
        
        while nn < M and iteration_count < max_hmc_iterations:
            iteration_count += 1
            
            # Sample momentum
            u = np.random.normal(size=(p, 1))
            u_proj = (np.eye(p) - W @ W.T) @ u
            
            # Compute current log-posterior and gradient
            logPosteriorW, grad_logPosteriorW = loglik_and_gradW_numpy(
                Y, X, W, F_Wprior, use_tf, kernel_type,
                layer=layer, Q=Q, R=R,
                tau2_y=tau2_y, tau2_q=tau2_q, tau2_r=tau2_r,
                theta_D_y=theta_D_y, theta_D_q=theta_D_q, theta_D_r=theta_D_r,
                g_y=g_y, g_q=g_q, g_r=g_r
            )
            
            H_old = logPosteriorW - np.linalg.norm(u_proj.flatten())**2 / 2.0
            
            W_star = W.copy()
            
            # Leapfrog integration
            for i in range(T_step):
                # Half step for momentum
                logPosteriorW, grad_logPosteriorW = loglik_and_gradW_numpy(
                    Y, X, W_star, F_Wprior, use_tf, kernel_type,
                    layer=layer, Q=Q, R=R,
                    tau2_y=tau2_y, tau2_q=tau2_q, tau2_r=tau2_r,
                    theta_D_y=theta_D_y, theta_D_q=theta_D_q, theta_D_r=theta_D_r,
                    g_y=g_y, g_q=g_q, g_r=g_r
                )
                u_new = u_proj + eps * grad_logPosteriorW / 2
                u_new_proj = (np.eye(p) - W_star @ W_star.T) @ u_new
                
                # Geodesic flow (rotation)
                alpha = np.linalg.norm(u_new_proj)
                if alpha < 1e-10:  # Avoid division by zero
                    alpha = 1e-10
                V_0 = np.hstack([W_star, u_new_proj])
                a = np.array([1.0, alpha])
                rot = np.array([[np.cos(alpha * eps), -np.sin(alpha * eps)],
                              [np.sin(alpha * eps), np.cos(alpha * eps)]])
                
                V_eps = V_0 @ np.diag(1.0 / a) @ rot @ np.diag(a)
                W_star = V_eps[:, 0].reshape(p, 1).copy()
                u_new = V_eps[:, 1].reshape(p, 1).copy()
                
                # Half step for momentum
                logPosteriorW, grad_logPosteriorW = loglik_and_gradW_numpy(
                    Y, X, W_star, F_Wprior, use_tf, kernel_type,
                    layer=layer, Q=Q, R=R,
                    tau2_y=tau2_y, tau2_q=tau2_q, tau2_r=tau2_r,
                    theta_D_y=theta_D_y, theta_D_q=theta_D_q, theta_D_r=theta_D_r,
                    g_y=g_y, g_q=g_q, g_r=g_r
                )
                u_new = u_new + eps * grad_logPosteriorW / 2
                u_new_proj = (np.eye(p) - W_star @ W_star.T) @ u_new
                u_proj = u_new_proj.copy()
            
            # Compute new Hamiltonian
            H_new = logPosteriorW - np.linalg.norm(u_new_proj.flatten())**2 / 2.0
            
            # Accept/reject with numerical stability
            delta_H = H_new - H_old
            if delta_H > 10:  # Prevent overflow
                accept_prob = 0.0
            elif delta_H < -10:  # Always accept
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta_H)
            
            if np.random.uniform() < accept_prob:
                W = W_star.copy()
                hmc_samples.append(W)
                nn += 1
        
        # If we didn't get enough samples, use the last W
        if nn < M:
            warnings.warn(f"HMC: Only generated {nn}/{M} samples after {max_hmc_iterations} iterations. Using last W.")
            while len(hmc_samples) < M:
                hmc_samples.append(W.copy())
    
    return hmc_samples[0] if M == 1 else hmc_samples


# =============================================================================
# Matrix Langevin Prior Parameters (M, V, Lambda)
# =============================================================================

def _sample_matrix_vmf_gibbs(
    M_param: np.ndarray,
    X_prev: Optional[np.ndarray],
    *,
    mv_sampler: str,
    rstiefel_rscol: Optional[int],
    python_gibbs_sampler,
    direct_sampler,
) -> np.ndarray:
    """Sample a matrix-vMF update using the Python or optional R rstiefel backend."""
    backend = mv_sampler.lower()
    if backend not in {"python", "rstiefel"}:
        raise ValueError("mv_sampler must be 'python' or 'rstiefel'.")
    if X_prev is None:
        return direct_sampler()
    if backend == "rstiefel":
        from rstiefel_backend import rmf_matrix_gibbs_rstiefel  # type: ignore[import]

        return rmf_matrix_gibbs_rstiefel(M_param, X_prev, rscol=rstiefel_rscol)
    return python_gibbs_sampler(M_param, X_prev)


def sample_M(W: np.ndarray, Lambda: np.ndarray, V: np.ndarray, p: int,
             prior_M: Optional[np.ndarray] = None,
             M_prev: Optional[np.ndarray] = None,
             mv_sampler: str = "python",
             rstiefel_rscol: Optional[int] = None) -> np.ndarray:
    """
    Sample M from matrix von Mises-Fisher distribution using Gibbs sampling.
    
    Full conditional:
        M | W, Λ, V ~ MF(W @ Λ + prior_M)
    
    Args:
        W: Projection matrix (p, D)
        Lambda: Diagonal concentration matrix (D, D) or vector (D,)
        V: Orthonormal matrix (D, D) - not used in this formulation
        p: Dimension
        prior_M: Prior mean for M (p, D), default: zeros
        M_prev: Previous M sample for Gibbs sampling (p, D)
        mv_sampler: "python" for local Gibbs updates or "rstiefel" for R rstiefel::rmf.matrix.gibbs
        rstiefel_rscol: Optional number of columns to update simultaneously in rstiefel
        
    Returns:
        Sampled M matrix (p, D)
    """
    if prior_M is None:
        prior_M = np.zeros((p, W.shape[1]))  # Default: zeros(p, D)
    
    # For D=1: Lambda is (1,) or scalar, W is (p, 1)
    # M_param = W @ Lambda + prior_M
    # Handle Lambda: if it's (1,), extract scalar; if it's already scalar, use it
    if isinstance(Lambda, np.ndarray):
        if Lambda.ndim == 1 and len(Lambda) == 1:
            lambda_val = Lambda[0]
        elif Lambda.ndim == 0:
            lambda_val = Lambda.item()
        else:
            # Lambda is a vector or matrix
            Lambda_diag = np.diag(Lambda) if Lambda.ndim > 1 else Lambda
            M_param = W @ Lambda_diag + prior_M
            return _sample_matrix_vmf_gibbs(
                M_param,
                M_prev,
                mv_sampler=mv_sampler,
                rstiefel_rscol=rstiefel_rscol,
                python_gibbs_sampler=rmf_matrix_gibbsN,
                direct_sampler=lambda: rmf_matrix(M_param),
            )
    else:
        lambda_val = Lambda
    
    # For D=1 with scalar Lambda: M_param = W * lambda_val + prior_M
    M_param = W * lambda_val + prior_M
    
    # Use Gibbs sampling with previous sample
    # For D=1, reshape M_prev to column vector as per user's code.
    M_prev_reshaped = None if M_prev is None else (M_prev.reshape(-1, 1) if M_prev.ndim == 2 else M_prev)
    return _sample_matrix_vmf_gibbs(
        M_param,
        M_prev_reshaped,
        mv_sampler=mv_sampler,
        rstiefel_rscol=rstiefel_rscol,
        python_gibbs_sampler=rmf_matrix_gibbsN,
        direct_sampler=lambda: rmf_matrix(M_param),
    )


def sample_V(W: np.ndarray, Lambda: np.ndarray, M: np.ndarray, D: int,
             prior_V: Optional[np.ndarray] = None,
             V_prev: Optional[np.ndarray] = None,
             mv_sampler: str = "python",
             rstiefel_rscol: Optional[int] = None) -> np.ndarray:
    """
    Sample V from matrix von Mises-Fisher distribution using Gibbs sampling.
    
    Full conditional:
        V | W, Λ, M ~ MF((W^T @ M) @ Λ + prior_V)
    
    Args:
        W: Projection matrix (p, D)
        Lambda: Diagonal concentration matrix (D, D) or vector (D,)
        M: Current M sample (p, D)
        D: Reduced dimension
        prior_V: Prior mean for V (D, D), default: zeros
        V_prev: Previous V sample for Gibbs sampling (D, D)
        mv_sampler: "python" for local Gibbs updates or "rstiefel" for R rstiefel::rmf.matrix.gibbs
        rstiefel_rscol: Optional number of columns to update simultaneously in rstiefel
        
    Returns:
        Sampled V matrix (D, D)
    """
    if prior_V is None:
        prior_V = np.zeros((D, D))  # Default: zeros(D, D)
    
    # For D=1: Lambda is (1,) or scalar, W is (p, 1), M is (p, 1)
    # V_param = (W^T @ M) @ Lambda + prior_V
    # Handle Lambda: if it's (1,), extract scalar; if it's already scalar, use it
    if isinstance(Lambda, np.ndarray):
        if Lambda.ndim == 1 and len(Lambda) == 1:
            lambda_val = Lambda[0]
        elif Lambda.ndim == 0:
            lambda_val = Lambda.item()
        else:
            # Lambda is a vector or matrix
            Lambda_diag = np.diag(Lambda) if Lambda.ndim > 1 else Lambda
            WTM = W.T @ M  # (D, p) @ (p, D) = (D, D)
            V_param = WTM @ Lambda_diag + prior_V
            return _sample_matrix_vmf_gibbs(
                V_param,
                V_prev,
                mv_sampler=mv_sampler,
                rstiefel_rscol=rstiefel_rscol,
                python_gibbs_sampler=rmf_matrix_gibbs,
                direct_sampler=lambda: rmf_matrix(V_param.T).T,
            )
    else:
        lambda_val = Lambda
    
    # For D=1 with scalar Lambda: V_param = (W^T @ M) * lambda_val + prior_V
    WTM = W.T @ M  # (1, p) @ (p, 1) = (1, 1) for D=1
    V_param = WTM * lambda_val + prior_V
    
    return _sample_matrix_vmf_gibbs(
        V_param,
        V_prev,
        mv_sampler=mv_sampler,
        rstiefel_rscol=rstiefel_rscol,
        python_gibbs_sampler=rmf_matrix_gibbs,
        direct_sampler=lambda: rmf_matrix(V_param.T).T,
    )


def laplace_approximate_0F1(p: int, lambdas: np.ndarray) -> float:
    """
    Laplace approximation for hypergeometric function ₀F₁.
    
    Args:
        p: Parameter
        lambdas: Array of concentration parameters
        
    Returns:
        Approximation of ₀F₁(p/2; Λ)
    """
    lambdas_flat = lambdas.flatten()
    d = len(lambdas_flat)
    
    # Calculate y_hat
    y_hat = (2 * lambdas_flat / p) / (1 + np.sqrt(4 * lambdas_flat**2 / p**2 + 1))
    
    # Calculate R_0,1
    R_01 = 1.0
    for i in range(d):
        for j in range(i, d):
            R_01 *= (1 - y_hat[i]**2 * y_hat[j]**2)
    
    # Calculate product term
    product_term = 1.0
    for i in range(d):
        product_term *= (1 - y_hat[i]**2)**(p / 2) * np.exp(lambdas_flat[i] * y_hat[i])
    
    oF1 = (1 / np.sqrt(R_01)) * product_term
    return oF1


def likelihood_lambda(V: np.ndarray, M: np.ndarray, Lambda: np.ndarray, 
                     W: np.ndarray, p: int) -> float:
    """
    Compute likelihood for Lambda parameter.
    
    Args:
        V, M: Prior matrices
        Lambda: Concentration parameters
        W: Projection matrix
        p: Dimension
        
    Returns:
        Likelihood value
    """
    Lambda_diag = np.diag(Lambda.flatten())
    exponential_term = np.sum(np.diag(V.T @ Lambda_diag @ M.T @ W))
    likelihood_value = np.exp(exponential_term) / laplace_approximate_0F1(p, Lambda)
    return likelihood_value


def sample_Lambda_slice(Lambda_cur: np.ndarray, prior_value: np.ndarray,
                        M: np.ndarray, V: np.ndarray, W: np.ndarray,
                        p: int, max_iter: int = 1000, epsilon: float = 2.0) -> np.ndarray:
    """
    Sample Lambda using elliptical slice sampling.
    
    Args:
        Lambda_cur: Current Lambda value
        prior_value: Prior value (nu)
        M, V, W: Model matrices
        p: Dimension
        max_iter: Maximum iterations
        epsilon: Minimum threshold for Lambda
        
    Returns:
        Sampled Lambda
    """
    cur_log_like = likelihood_lambda(V, M, Lambda_cur, W, p)
    nu = prior_value
    hh = cur_log_like
    
    # Set up angle bracket
    phi = np.random.uniform(0, 2 * np.pi)
    phi_min = phi - 2 * np.pi
    phi_max = phi
    
    # Slice sampling loop
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        
        # Propose Lambda
        Lambda_prop = np.cos(phi) * Lambda_cur + np.sin(phi) * nu
        
        # Check positivity
        if np.all(Lambda_prop > epsilon):
            prop_log_like = likelihood_lambda(V, M, Lambda_prop, W, p)
            
            if prop_log_like > hh:
                    # New point is on the slice, exit loop
                Lambda_cur = Lambda_prop
                break
        
        # Shrink slice
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            phi = np.random.uniform(-epsilon, epsilon)
        
        phi = np.random.uniform(phi_min, phi_max)
    
    # Return current if max iterations reached
    return Lambda_cur


def covar_sepp(Z, d, g):
    """
    Covariance function for 3-layer latent variable sampling.
    
    Args:
        Z: Input matrix (n, D)
        d: Lengthscale parameter (scalar)
        g: Nugget parameter
        
    Returns:
        Covariance matrix (n, n)
    """
    n = Z.shape[0]
    C_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sum_term = np.sum((Z[i, :] - Z[j, :])**2)
            C_matrix[i, j] = np.exp(-sum_term / (2 * d**2))
    return C_matrix + g * np.eye(n)


def log_likelihoodd(g, Z, Y, theta_D, tau2):
    """
    Log-likelihood for 3-layer latent variable sampling.
    
    Args:
        g: Nugget
        Z: Latent variable matrix
        Y: Response (or upper layer latent variable)
        theta_D: Lengthscale
        tau2: Variance parameter
        
    Returns:
        Log-likelihood value
    """
    Sigma = covar_sepp(Z, theta_D, g)
    Sigma_inv = np.linalg.pinv(Sigma)
    n = len(Y)
    sign, log_det = np.linalg.slogdet(Sigma)
    log_lik = -0.5 * (n * np.log(2 * np.pi * tau2) + log_det + Y.T @ Sigma_inv @ Y / tau2)
    return log_lik


def sample_R_3layer_ESS(Y, R_current, Q, g_q, theta_q, theta_r, g_r,
                        tau2_q=1.0, tau2_r=1.0,
                        kernel_type: str = 'isotropic_squared_exponential'):
    """
    Sample latent R for 3-layer model using ESS (D=1).
    
    For 3-layer: R is the innermost latent layer
        Q | R, θ_q, g_q, τ²_q ~ GP(0, τ²_q(C_q + g_q*I))
        R | Z, θ_r, g_r, τ²_r ~ GP(0, τ²_r(C_r + g_r*I))
    
    Args:
        Y: Q values (used for likelihood)
        R_current: Current R values (n, 1)
        Q: Projected inputs Z = XW (n, 1), used for R prior
        g_q: Nugget for Q layer
        theta_q: Lengthscale for Q layer
        theta_r: Lengthscale for R layer
        g_r: Nugget for R layer
        tau2_q: Variance for Q|R likelihood kernel
        tau2_r: Variance for R|Z prior kernel
        kernel_type: Kernel type for covariance functions
        
    Returns:
        Updated R (n, 1)
    """
    N, D = R_current.shape
    D_kernel = 1  # D=1 for this function
    
    # Create kernel instances
    kernel_q = get_kernel_instance(kernel_type, theta_q, g_q, tau2_q, D_kernel)
    kernel_r = get_kernel_instance(kernel_type, theta_r, g_r, tau2_r, D_kernel)
    
    # Compute log-likelihood using kernel
    ll_prev = kernel_q.log_likelihood(Y.flatten(), R_current)
    
    # Sample prior: R ~ N(0, τ²_r(C_r + g_r I)) using kernel, with input Z (=Q arg here)
    cov = kernel_r.compute_covariance(Q, Q)
    R_prior = np.random.multivariate_normal(mean=np.zeros(N), cov=cov).reshape(-1, 1)
    
    # ESS
    a = np.random.uniform(0, 2 * np.pi)
    amin, amax = a - 2 * np.pi, a
    ru = np.random.uniform()
    ll_threshold = ll_prev + np.log(ru)
    
    accept = False
    count = 0
    R_prev = R_current.copy()
    
    while not accept:
        count += 1
        R_current = R_prev * np.cos(a) + R_prior * np.sin(a)
        
        new_logl = kernel_q.log_likelihood(Y.flatten(), R_current)
        
        if new_logl > ll_threshold:
            ll_prev = new_logl
            accept = True
        else:
            if a < 0:
                amin = a
            else:
                amax = a
            a = np.random.uniform(amin, amax)
            
            if count > 1000:
                print(f"Warning: ESS for R reached max iterations (1000)")
                break
    
    return R_current


def sample_Q_3layer_ESS(Q_current, R, Y, g_y, theta_y, g_q, theta_q,
                        tau2_y=1.0, tau2_q=1.0,
                        kernel_type: str = 'isotropic_squared_exponential'):
    """
    Sample latent Q for 3-layer model using ESS (D=1).
    
    For 3-layer:
        Y | Q, θ_y, g_y, τ²_y ~ GP(0, τ²_y(C_y + g_y*I))
        Q | R, θ_q, g_q, τ²_q ~ GP(0, τ²_q(C_q + g_q*I))
    
    Args:
        Q_current: Current Q values (n, 1)
        R: R values (n, 1)
        Y: Response (n,)
        g_y: Nugget for Y layer
        theta_y: Lengthscale for Y layer
        g_q: Nugget for Q layer
        theta_q: Lengthscale for Q layer
        tau2_y: Variance for Y|Q likelihood kernel
        tau2_q: Variance for Q|R prior kernel
        kernel_type: Kernel type for covariance functions
        
    Returns:
        Updated Q (n, 1)
    """
    N, D = Q_current.shape
    D_kernel = 1  # D=1 for this function
    
    # Create kernel instances
    kernel_y = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D_kernel)
    kernel_q = get_kernel_instance(kernel_type, theta_q, g_q, tau2_q, D_kernel)
    
    # Compute log-likelihood using kernel
    ll_prev = kernel_y.log_likelihood(Y, Q_current)
    
    # Sample prior: Q ~ N(0, τ²_q(C_q + g_q I)) using kernel
    cov = kernel_q.compute_covariance(R, R)
    Q_prior = np.random.multivariate_normal(mean=np.zeros(N), cov=cov).reshape(-1, 1)
    
    # ESS
    a = np.random.uniform(0, 2 * np.pi)
    amin, amax = a - 2 * np.pi, a
    ru = np.random.uniform()
    ll_threshold = ll_prev + np.log(ru)
    
    accept = False
    count = 0
    Q_prev = Q_current.copy()
    
    while not accept:
        count += 1
        Q_current = Q_prev * np.cos(a) + Q_prior * np.sin(a)
        
        new_logl = kernel_y.log_likelihood(Y, Q_current)
        
        if new_logl >= ll_threshold:
            ll_prev = new_logl
            accept = True
        else:
            if a < 0:
                amin = a
            else:
                amax = a
            a = np.random.uniform(amin, amax)
            
            if count > 1000:
                print(f"Warning: ESS for Q reached max iterations (1000)")
                break
    
    return Q_current





def covar_isotropic_full(Z, theta_scalar, g):
    """
    Isotropic covariance over full Z matrix using single lengthscale.
    Used in 2-layer ESS.
    """
    n = Z.shape[0]
    C_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sum_term = np.sum((Z[i, :] - Z[j, :])**2)
            C_matrix[i, j] = np.exp(-sum_term / (2 * theta_scalar**2))
    return C_matrix + g * np.eye(n)


def sample_Q_2layer_ESS(Y, Q_current, Z, g_y, theta_y, theta_q, g_q, tau2_y,
                       tau2_q=1.0,
                       kernel_type: str = 'isotropic_squared_exponential'):
    """
    Sample latent Q for 2-layer model using Elliptical Slice Sampling (D=1).
    
    For 2-layer Deep GP:
        Y | Q, θ_y, g_y, τ²_y ~ GP(0, τ²_y(C_y + g_y*I))
        Q | Z, θ_q, g_q, τ²_q ~ GP(0, τ²_q(C_q + g_q*I))
    
    Args:
        Y: Response vector (n,)
        Q_current: Current Q values (n, 1)
        Z: Projected inputs Z = XW (n, 1)
        g_y: Nugget for Y layer
        theta_y: Lengthscale for Y layer
        theta_q: Lengthscale for Q layer (prior)
        g_q: Nugget for Q layer (prior)
        tau2_y: Observation noise for Y layer
        tau2_q: Latent variance for Q layer prior
        kernel_type: Kernel type for covariance functions
        
    Returns:
        Updated Q (n, 1)
    """
    N = len(Y)
    D = 1  # D=1 for this function
    
    # Compute current log-likelihood using kernel
    def log_likelihood_Q(Q):
        # Use kernel instance for Y layer likelihood
        kernel_y = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
        Z_Q = Q.reshape(-1, 1)  # Q as input for Y layer
        return kernel_y.log_likelihood(Y, Z_Q)
    
    ll_prev = log_likelihood_Q(Q_current)
    
    # Sample from prior: Q ~ N(0, tau2_q * (C_q + g_q I))
    # Use kernel for Q layer prior (with theta_q, g_q)
    kernel_q = get_kernel_instance(kernel_type, theta_q, g_q, tau2_q, D)
    cov = kernel_q.compute_covariance(Z, Z)
    Q_prior = np.random.multivariate_normal(mean=np.zeros(N), cov=cov).reshape(-1, 1)
    
    # Elliptical Slice Sampling
    a = np.random.uniform(0, 2 * np.pi)
    amin, amax = a - 2 * np.pi, a
    ru = np.random.uniform()
    ll_threshold = ll_prev + np.log(ru)
    
    accept = False
    count = 0
    Q_prev = Q_current.copy()
    
    while not accept:
        count += 1
        Q_new = Q_prev * np.cos(a) + Q_prior * np.sin(a)
        
        new_logl = log_likelihood_Q(Q_new)
        
        if new_logl > ll_threshold:
            ll_prev = new_logl
            accept = True
            Q_current = Q_new
        else:
            if a < 0:
                amin = a
            else:
                amax = a
            a = np.random.uniform(amin, amax)
            
            if count > 100:
                print(f"Warning: ESS reached max iterations (100), accepting current sample")
                break
    
    return Q_current


if __name__ == "__main__":
    print("="*70)
    print("Parameter Sampler Module for D=1 (Test)")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n, p, D = 50, 10, 1
    
    X = np.random.randn(n, p)
    W_true = np.random.randn(p, D)
    W_true = W_true / np.linalg.norm(W_true)
    
    Z = X @ W_true
    theta_test = 1.0
    C = np.exp(-0.5 * np.sum((Z[:, np.newaxis, :] - Z[np.newaxis, :, :])**2, axis=2) / theta_test)
    Y = np.random.multivariate_normal(np.zeros(n), C + 0.01 * np.eye(n))
    
    print(f"\nTest Data: n={n}, p={p}, D={D}")
    
    # Test sampling functions
    print("\n" + "-"*70)
    print("Testing Parameter Sampling Functions")
    print("-"*70)
    
    # Test tau2 sampling
    tau2_sample = sample_tau2(Y, X, W_true, tau2_curr=1.0, theta_D=1.0, g=0.01)
    print(f"✓ tau2 sampled: {tau2_sample:.6f}")
    
    # Test g sampling
    g_sample = sample_g(Y, X, W_true, g_curr=0.01, theta_D=1.0, tau2=1.0)
    print(f"✓ g sampled: {g_sample:.6f}")
    
    # Test theta_D sampling
    theta_sample = sample_theta_D(Y, X, W_true, theta_D_curr=1.0, tau2=1.0, g=0.01)
    print(f"✓ theta_D sampled: {theta_sample:.6f}")
    
    # Test W sampling
    W_sample = sample_W_HMC_stiefel(
        Y, X, W_true,
        M=1, eps=0.001, T_step=10, use_tf=False, layer=1,
        tau2_y=1.0, theta_D_y=1.0, g_y=0.01
    )
    print(f"✓ W sampled, shape: {W_sample.shape}, norm: {np.linalg.norm(W_sample):.6f}")
    
    print("\n" + "="*70)
    print("All sampling functions tested successfully!")
    print("="*70)
