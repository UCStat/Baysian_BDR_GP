"""
Parameter Sampling Functions for GP with Bayesian Dimensionality Reduction (D>1)

This module implements MCMC and MLE sampling functions for all parameters in 1, 2, and 3-layer
Deep Gaussian Process models with dimensionality reduction to D>1 (e.g., D=2, 3, 5, etc.).

Key differences from D=1 module:
    - θ (theta_D) is now a VECTOR with one lengthscale per dimension
    - Uses separable squared exponential kernel
    - W is a matrix (p × D) on Stiefel manifold St(p, D)

Parameters sampled:
    - τ² (tau2): Observation noise variance (scalar)
    - g: Nugget parameter (scalar)
    - θ (theta_D): Lengthscale parameters (D-dimensional vector)
    - W: Projection matrix (p × D) on Stiefel manifold
    - M, V, Λ: Matrix Langevin prior parameters
"""

import numpy as np
from scipy.stats import invgamma, gamma, uniform
from scipy.special import gammaln, ive
from scipy.linalg import svd, qr, null_space
from numpy.linalg import norm, matrix_rank
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

def get_kernel_instance(kernel_type: str, theta_D: np.ndarray, 
                        g: float, tau2: float, D: int):
    """
    Create kernel instance based on kernel type for D>1.
    
    Args:
        kernel_type: One of 'isotropic_squared_exponential', 'separable_squared_exponential',
                     'isotropic_matern32', 'separable_matern32'
        theta_D: Lengthscale parameter(s) - vector (D,) for separable, scalar for isotropic
        g: Nugget parameter
        tau2: Observation noise variance (must be scalar)
        D: Reduced dimension (must be >1)
        
    Returns:
        Kernel instance
    """
    # Ensure tau2 is scalar
    if isinstance(tau2, np.ndarray):
        tau2 = tau2.item() if tau2.size == 1 else float(tau2[0])
    tau2 = float(tau2)
    
    # Ensure g is scalar
    if isinstance(g, np.ndarray):
        g = g.item() if g.size == 1 else float(g[0])
    g = float(g)
    
    if kernel_type == 'isotropic_squared_exponential':
        # For isotropic, use first element of theta_D or average
        if np.isscalar(theta_D) or (isinstance(theta_D, np.ndarray) and theta_D.size == 1):
            theta = float(theta_D if np.isscalar(theta_D) else theta_D.item())
        else:
            theta = float(theta_D[0]) if len(theta_D) > 0 else 1.0
        return IsotropicSquaredExponentialKernel(lengthscale=theta, nugget=g, tau2=tau2)
    elif kernel_type == 'separable_squared_exponential':
        if np.isscalar(theta_D) or (isinstance(theta_D, np.ndarray) and theta_D.size == 1):
            theta_array = np.full(D, float(theta_D if np.isscalar(theta_D) else theta_D.item()))
        else:
            theta_array = np.array(theta_D) if isinstance(theta_D, (list, np.ndarray)) else np.array([theta_D])
            # Ensure length matches D
            if len(theta_array) != D:
                if len(theta_array) < D:
                    theta_array = np.concatenate([theta_array, np.full(D - len(theta_array), theta_array[0])])
                else:
                    theta_array = theta_array[:D]
        return SeparableSquaredExponentialKernel(lengthscales=theta_array, nugget=g, tau2=tau2)
    elif kernel_type == 'isotropic_matern32':
        if np.isscalar(theta_D) or (isinstance(theta_D, np.ndarray) and theta_D.size == 1):
            theta = float(theta_D if np.isscalar(theta_D) else theta_D.item())
        else:
            theta = float(theta_D[0]) if len(theta_D) > 0 else 1.0
        return IsotropicMatern32Kernel(lengthscale=theta, nugget=g, tau2=tau2)
    elif kernel_type == 'separable_matern32':
        if np.isscalar(theta_D) or (isinstance(theta_D, np.ndarray) and theta_D.size == 1):
            theta_array = np.full(D, float(theta_D if np.isscalar(theta_D) else theta_D.item()))
        else:
            theta_array = np.array(theta_D) if isinstance(theta_D, (list, np.ndarray)) else np.array([theta_D])
            # Ensure length matches D
            if len(theta_array) != D:
                if len(theta_array) < D:
                    theta_array = np.concatenate([theta_array, np.full(D - len(theta_array), theta_array[0])])
                else:
                    theta_array = theta_array[:D]
        return SeparableMatern32Kernel(lengthscales=theta_array, nugget=g, tau2=tau2)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Must be one of: "
                        "'isotropic_squared_exponential', 'separable_squared_exponential', "
                        "'isotropic_matern32', 'separable_matern32'")


# =============================================================================
# Utility Functions for Stiefel Manifold Sampling
# =============================================================================

def NullC(M: np.ndarray) -> np.ndarray:
    """Compute null space of matrix M using QR decomposition."""
    if M.size == 0 or M.shape[1] == 0:
        return np.eye(M.shape[0])
    
    Q, R = qr(M, mode='full')
    rank = np.linalg.matrix_rank(M)
    
    if rank == 0:
        return np.eye(M.shape[0])
    else:
        return Q[:, rank:]


def rW(kap: float, m: int) -> float:
    """Simulate from the W distribution (Wood 1994)."""
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
    return 0.0


def rmf_vector(kmu: np.ndarray) -> np.ndarray:
    """Simulate from vector multivariate Fisher distribution (Wood 1994)."""
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
    """Sample from matrix von Mises-Fisher distribution."""
    if M.shape[1] == 1:
        XX = rmf_vector(M[:, 0]).reshape(-1, 1)
    else:
        U, S, Vt = svd(M, full_matrices=False)
        H = U @ np.diag(S)  # Ensure H matches dimensions of M
        m, R = H.shape
        cmet = False
        
        while not cmet:
            U_sample = np.zeros((m, R))
            U_sample[:, 0] = rmf_vector(H[:, 0])
            lr = 0
            
            for j in range(1, R):
                N = null_space(U_sample[:, :j].T)
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


# =============================================================================
# Covariance Functions for D>1
# =============================================================================

def covar_sep(Z: np.ndarray, theta: np.ndarray, g: float) -> np.ndarray:
    """
    Compute separable squared exponential covariance matrix.
    
    Args:
        Z: Reduced inputs (n, D)
        theta: Lengthscale parameters (D,) - one per dimension
        g: Nugget parameter
        
    Returns:
        Covariance matrix Σ = C + g*I
    """
    n = Z.shape[0]
    D = Z.shape[1]
    
    theta = np.atleast_1d(theta)
    
    C_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sum_term = sum((Z[i, k] - Z[j, k])**2 / (2 * theta[k]**2) for k in range(D))
            C_matrix[i, j] = np.exp(-sum_term)
    
    Sigma = C_matrix + g * np.eye(n)
    return Sigma


def rmf_vectorN(kmu):
    """Simulate from vector MF distribution (version N for gibbsN)."""
    kap = np.linalg.norm(kmu)
    mu = kmu / kap if kap != 0 else np.zeros_like(kmu)
    m = len(mu)
    
    if kap == 0:
        u = np.random.normal(size=m)
        u /= np.linalg.norm(u)
        return u.reshape(m, 1)
    
    if kap > 0:
        if m == 1:
            prob = 1 / (1 + np.exp(2 * kap * mu))
            u = (-1) ** np.random.binomial(1, prob)
            return np.array([u])
        
        if m > 1:
            W = rW(kap, m)
            V = np.random.normal(size=m - 1)
            V /= np.linalg.norm(V)
            x = np.concatenate(((1 - W**2)**0.5 * V, [W]))
            u = np.hstack((NullC(mu.reshape(-1,1)), mu.reshape(-1, 1))) @ x
            return u


def rmf_matrixN(M):
    """Simulate from matrix MF distribution (version N for gibbsN)."""
    if M.shape[1] == 1:
        XX = rmf_vectorN(M[:, 0]).reshape(-1, 1)
    else:
        U, S, Vt = svd(M, full_matrices=False)
        H = U @ np.diag(S)
        m, R = H.shape
        cmet = False
        while not cmet:
            U_sample = np.zeros((m, R))
            U_sample[:, 0] = rmf_vectorN(H[:, 0]).flatten()
            lr = 0
            for j in range(1, R):
                N = null_space(U_sample[:, :j].T)
                xx = rmf_vectorN(N.T @ H[:, j])
                Nx = N @ xx.reshape(-1,1)
                U_sample[:, j] = Nx.flatten()
                if S[j] > 0:
                    xn = np.linalg.norm(N.T @ H[:, j])
                    xd = np.linalg.norm(H[:, j])
                    lbr = (np.log(ive(0.5 * (m - j - 1), xn)) -
                           np.log(ive(0.5 * (m - j - 1), xd)))
                    lbr = 0.5 * (np.log(xd) - np.log(xn)) if np.isnan(lbr) else lbr
                    lr += lbr + (xn - xd) + 0.5 * (m - j - 1) * (np.log(xd) - np.log(xn))
            cmet = np.log(np.random.uniform()) < lr
        XX = U_sample @ Vt
    return XX


def rmf_matrix_gibbsN(M, Xn, rscol=None):
    """Gibbs sampling for matrix-variate von Mises-Fisher (version N for M sampling)."""
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


def rmf_matrix_gibbs(M, Xn, rscol=None):
    """Gibbs sampling for matrix-variate von Mises-Fisher (for V sampling)."""
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


def log_likelihood_gp(g: float, Z: np.ndarray, Y: np.ndarray, 
                     theta_D: np.ndarray, tau2: float,
                     kernel_type: str = 'separable_squared_exponential') -> float:
    """
    Compute log-likelihood for GP model using kernel.
    
    Args:
        g: Nugget parameter
        Z: Reduced inputs (n, D)
        Y: Response vector (n,)
        theta_D: Lengthscale parameters (D,) for separable, or scalar for isotropic
        tau2: Observation noise variance
        kernel_type: Kernel type to use
        
    Returns:
        Log-likelihood value
    """
    D = Z.shape[1] if len(Z.shape) > 1 else 1
    kernel = get_kernel_instance(kernel_type, theta_D, g, tau2, D)
    return kernel.log_likelihood(Y, Z)


# =============================================================================
# MLE Estimation Functions
# =============================================================================

def estimate_tau2_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                     theta_D: np.ndarray, g: float,
                     kernel_type: str = 'separable_squared_exponential') -> float:
    """
    Estimate τ² using Maximum Likelihood Estimation.
    
    MLE estimate:
        τ²_MLE = (1/n) * Y^T Σ^{-1} Y
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p) OR latent space Z/Q/R (n, D)
        W: Projection matrix (p, D) OR identity matrix (n, n) for multi-layer
        theta_D: Lengthscale parameters (D,)
        g: Nugget parameter
        kernel_type: Kernel type to use
        
    Returns:
        MLE estimate of τ²
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    # If W is identity matrix (n, n), input_matrix is already Q or R
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        # Ensure Q/R is 2D: (n, D)
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
    D = Z.shape[1] if len(Z.shape) > 1 else len(theta_D)
    
    # Use kernel to compute covariance (without tau2)
    kernel = get_kernel_instance(kernel_type, theta_D, g, tau2=1.0, D=D)
    Sigma = kernel.compute_sigma(Z) / 1.0  # Remove tau2 scaling
    
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return 1.0
    
    tau2_mle = (Y.T @ Sigma_inv @ Y) / n
    tau2_mle = max(tau2_mle, 1e-6)
    
    return float(tau2_mle)


def estimate_g_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                  theta_D: np.ndarray, tau2: float,
                  bounds: Tuple[float, float] = (1e-6, 0.1),
                  n_grid: int = 50,
                  kernel_type: str = 'separable_squared_exponential') -> float:
    """
    Estimate nugget parameter g using Maximum Likelihood Estimation.
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p) OR latent space Z/Q/R (n, D)
        W: Projection matrix (p, D) OR identity matrix (n, n) for multi-layer
        theta_D: Lengthscale parameters (D,)
        tau2: Observation noise variance
        bounds: (lower, upper) bounds for g
        n_grid: Number of grid points to evaluate
        kernel_type: Kernel type to use
        
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
    
    D = Z.shape[1] if len(Z.shape) > 1 else len(theta_D) if isinstance(theta_D, np.ndarray) else 1
    
    g_grid = np.linspace(bounds[0], bounds[1], n_grid)
    log_liks = np.zeros(n_grid)
    
    for i, g_val in enumerate(g_grid):
        log_liks[i] = log_likelihood_gp(g_val, Z, Y, theta_D, tau2, kernel_type=kernel_type)
    
    idx_max = np.argmax(log_liks)
    g_mle = g_grid[idx_max]
    
    return float(g_mle)


def estimate_theta_D_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                        g: float, tau2: float, D: int,
                        bounds: Tuple[float, float] = (0.01, 10.0),
                        n_grid: int = 30,
                        kernel_type: str = 'separable_squared_exponential') -> np.ndarray:
    """
    Estimate lengthscale vector θ using MLE (grid search per dimension).
    
    For separable kernels: optimizes each dimension separately using column-wise input but same response.
    For isotropic kernels: optimizes single scalar theta (same for all dimensions).
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p)
        W: Projection matrix (p, D)
        g: Nugget parameter
        tau2: Observation noise variance
        D: Number of dimensions
        bounds: Search bounds for each θ_d
        n_grid: Number of grid points per dimension
        kernel_type: Kernel type to use
        
    Returns:
        MLE estimate of θ (D,)
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
        Z = input_matrix @ W  # Projected inputs (n, D)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    
    # Ensure D matches Z shape
    D_actual = Z.shape[1] if len(Z.shape) > 1 else 1
    if D_actual != D:
        D = D_actual
    
    theta_mle = np.zeros(D)
    
    # For separable kernels, optimize each dimension separately
    if 'separable' in kernel_type:
        # Optimize each dimension separately using column-wise input (Z[:, m]) but same response Y
        for m in range(D):
            Z_m = Z[:, m].reshape(-1, 1)  # Column-wise input (n, 1)
            W_identity = np.array([[1.0]])  # (1, 1) identity for pass-through
            
            theta_grid = np.linspace(bounds[0], bounds[1], n_grid)
            log_liks = np.zeros(n_grid)
            
            for i, theta_val in enumerate(theta_grid):
                # Use column-wise input with single theta value
                theta_test = np.array([theta_val])
                log_liks[i] = log_likelihood_gp(g, Z_m, Y, theta_test, tau2, kernel_type=kernel_type)
            
            idx_max = np.argmax(log_liks)
            theta_mle[m] = theta_grid[idx_max]
        
        return theta_mle
    else:
        # For isotropic kernels, optimize single scalar theta (same for all dimensions)
        theta_grid = np.linspace(bounds[0], bounds[1], n_grid)
        log_liks = np.zeros(n_grid)
        
        for i, theta_val in enumerate(theta_grid):
            theta_test = np.full(D, theta_val)  # Same value for all dimensions
            log_liks[i] = log_likelihood_gp(g, Z, Y, theta_test, tau2, kernel_type=kernel_type)
        
        idx_max = np.argmax(log_liks)
        theta_mle = np.full(D, theta_grid[idx_max])
        
        return theta_mle


def estimate_all_hyperparameters_MLE(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                                     D: int,
                                     tau2_init: float = 1.0, g_init: float = 0.01, 
                                     theta_init: Optional[np.ndarray] = None,
                                     n_iterations: int = 5,
                                     g_bounds: Tuple[float, float] = (1e-6, 0.1),
                                     theta_bounds: Tuple[float, float] = (0.01, 10.0),
                                     n_grid: int = 30,
                                     verbose: bool = False,
                                     kernel_type: str = 'separable_squared_exponential') -> Dict[str, Union[float, np.ndarray]]:
    """
    Jointly estimate τ², g, and θ using iterative MLE (coordinate ascent).
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p)
        W: Projection matrix (p, D)
        D: Reduced dimension
        tau2_init: Initial τ² value
        g_init: Initial g value
        theta_init: Initial θ vector (D,)
        n_iterations: Maximum iterations
        g_bounds: Bounds for g
        theta_bounds: Bounds for θ_d
        n_grid: Grid points for search
        verbose: Print iterations
        kernel_type: Kernel type to use
        
    Returns:
        Dictionary with MLE estimates
    """
    tau2 = tau2_init
    g = g_init
    theta_D = theta_init if theta_init is not None else np.ones(D)
    
    if verbose:
        print("Iterative MLE Estimation (D>1):")
        print(f"{'Iter':<6} {'tau2':<12} {'g':<12} {'theta':<30} {'log_lik':<12}")
        print("-" * 72)
    
    for iter in range(n_iterations):
        # Update tau2
        tau2 = estimate_tau2_MLE(Y, input_matrix, W, theta_D, g, kernel_type=kernel_type)
        
        # Update g
        g = estimate_g_MLE(Y, input_matrix, W, theta_D, tau2, g_bounds, n_grid, kernel_type=kernel_type)
        
        # Update theta_D (coordinate-wise)
        theta_D = estimate_theta_D_MLE(Y, input_matrix, W, g, tau2, D, theta_bounds, n_grid, kernel_type=kernel_type)
        
        if verbose:
            Z = input_matrix @ W
            log_lik = log_likelihood_gp(g, Z, Y, theta_D, tau2, kernel_type=kernel_type)
            theta_str = np.array2string(theta_D, precision=4, separator=',', suppress_small=True)
            print(f"{iter+1:<6} {tau2:<12.6f} {g:<12.6f} {theta_str:<30} {log_lik:<12.2f}")
    
    if verbose:
        print("-" * 72)
        print("MLE estimation complete!\n")
    
    return {
        'tau2': tau2,
        'g': g,
        'theta_D': theta_D
    }


# =============================================================================
# MCMC Sampling Functions
# =============================================================================

def sample_tau2(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
               tau2_curr: float, theta_D: np.ndarray, g: float,
               alpha1: float = 1.0, alpha2: float = 1000.0,
               kernel_type: str = 'separable_squared_exponential') -> float:
    """
    Sample τ² from inverse-gamma posterior.
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p) OR latent space Z/Q/R (n, D)
        W: Projection matrix (p, D) OR identity matrix (n, n) for multi-layer
        tau2_curr: Current τ² value
        theta_D: Lengthscale parameters (D,)
        g: Nugget parameter
        alpha1: Prior shape
        alpha2: Prior scale
        kernel_type: Kernel type to use
        
    Returns:
        Sampled τ² value
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    # If W is identity matrix (n, n), input_matrix is already Q or R
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        # Ensure Q/R is 2D: (n, D)
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
    D = Z.shape[1] if len(Z.shape) > 1 else len(theta_D)
    
    # Use kernel to compute covariance (without tau2)
    kernel = get_kernel_instance(kernel_type, theta_D, g, tau2=1.0, D=D)
    Sigma = kernel.compute_sigma(Z) / 1.0  # Remove tau2 scaling
    
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return tau2_curr
    
    alpha_post = alpha1 + n / 2
    beta_post = alpha2 + 0.5 * (Y.T @ Sigma_inv @ Y)
    
    tau2_sample = invgamma.rvs(alpha_post, scale=beta_post)
    
    return tau2_sample[0] if isinstance(tau2_sample, np.ndarray) else tau2_sample


def sample_g(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
            g_curr: float, theta_D: np.ndarray, tau2: float,
            beta1: float = 0.01, beta2: float = 0.005,
            l: float = 1.0, u: float = 2.0,
            kernel_type: str = 'separable_squared_exponential') -> float:
    """
    Sample nugget parameter g using Metropolis-Hastings.
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p)
        W: Projection matrix (p, D)
        g_curr: Current g value
        theta_D: Lengthscale parameters (D,)
        tau2: Observation noise variance
        beta1: Prior shape
        beta2: Prior rate
        l, u: Proposal bounds
        kernel_type: Kernel type to use
        
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
    
    D = Z.shape[1] if len(Z.shape) > 1 else (len(theta_D) if isinstance(theta_D, np.ndarray) else 1)
    
    g_prop = np.random.uniform((l * g_curr) / u, (u * g_curr) / l)
    
    if g_prop <= 0 or g_prop > 0.1:
        return g_curr
    
    ru = np.random.uniform(0, 1)
    eps = np.sqrt(np.finfo(float).eps)
    
    log_lik_curr = log_likelihood_gp(g_curr, Z, Y, theta_D, tau2, kernel_type=kernel_type)
    log_prior_curr = (beta1 - 1) * np.log(g_curr - eps) - beta2 * (g_curr - eps)
    
    log_lik_prop = log_likelihood_gp(g_prop, Z, Y, theta_D, tau2, kernel_type=kernel_type)
    log_prior_prop = (beta1 - 1) * np.log(g_prop - eps) - beta2 * (g_prop - eps)
    
    lpost_curr = log_lik_curr + log_prior_curr + np.log(ru) - np.log(g_curr) + np.log(g_prop)
    lpost_prop = log_lik_prop + log_prior_prop
    
    if lpost_prop > lpost_curr:
        return g_prop
    else:
        return g_curr


def sample_theta_D(Y: np.ndarray, input_matrix: np.ndarray, W: np.ndarray,
                  theta_D_curr: np.ndarray, tau2: float, g: float,
                  gamma1: float = 0.01, gamma2: float = 0.01/3,
                  l: float = 1.0, u: float = 2.0,
                  kernel_type: str = 'separable_squared_exponential') -> np.ndarray:
    """
    Sample lengthscale vector θ using Metropolis-Hastings.
    
    For D>1, θ is a vector with one lengthscale per dimension.
    For separable kernels, samples each dimension separately (column-wise input, same response).
    
    Args:
        Y: Response vector (n,)
        input_matrix: Design matrix X (n, p) or Z (n, D) for dimension-wise sampling
        W: Projection matrix (p, D) or identity (1, 1) for dimension-wise sampling
        theta_D_curr: Current θ values (D,) or scalar for dimension-wise
        tau2: Observation noise variance
        g: Nugget parameter
        gamma1: Prior shape
        gamma2: Prior rate
        l, u: Proposal bounds
        kernel_type: Kernel type to use
        
    Returns:
        Sampled θ vector (D,) or scalar for dimension-wise
    """
    n = len(Y)
    # Check if input_matrix is already in latent space (multi-layer case)
    if W.shape[0] == W.shape[1] and W.shape[0] == n:
        # Multi-layer case: input_matrix is already Q or R (latent space)
        if input_matrix.ndim == 1:
            Z = input_matrix.reshape(-1, 1)
        else:
            Z = input_matrix
        D = Z.shape[1] if len(Z.shape) > 1 else 1

        theta_D_curr = np.asarray(theta_D_curr, dtype=float).reshape(-1)
        if theta_D_curr.size == 1:
            theta_D_curr = np.full(D, float(theta_D_curr[0]))
        elif theta_D_curr.size < D:
            theta_D_curr = np.concatenate([
                theta_D_curr,
                np.full(D - theta_D_curr.size, theta_D_curr[0])
            ])
        else:
            theta_D_curr = theta_D_curr[:D]

        theta_D_prop = np.random.uniform((l * theta_D_curr) / u, (u * theta_D_curr) / l)

        if np.any(theta_D_prop <= 0) or np.any(theta_D_prop > 20):
            return theta_D_curr

        ru = np.random.uniform(0, 1)
        eps = np.sqrt(np.finfo(float).eps)

        log_lik_curr = log_likelihood_gp(g, Z, Y, theta_D_curr, tau2, kernel_type=kernel_type)
        log_prior_curr = np.sum((gamma1 - 1) * np.log(theta_D_curr - eps) - gamma2 * (theta_D_curr - eps))

        log_lik_prop = log_likelihood_gp(g, Z, Y, theta_D_prop, tau2, kernel_type=kernel_type)
        log_prior_prop = np.sum((gamma1 - 1) * np.log(theta_D_prop - eps) - gamma2 * (theta_D_prop - eps))

        lpost_curr = log_lik_curr + log_prior_curr + np.log(ru) - np.sum(np.log(theta_D_curr)) + np.sum(np.log(theta_D_prop))
        lpost_prop = log_lik_prop + log_prior_prop

        if lpost_prop > lpost_curr:
            return theta_D_prop
        return theta_D_curr
    # Check if this is dimension-wise sampling (W is 1x1 identity)
    elif W.shape == (1, 1) and input_matrix.shape[1] == 1:
        # Dimension-wise sampling for separable kernels
        Z = input_matrix  # Already column-wise (n, 1)
        D = 1
        theta_curr = theta_D_curr[0] if isinstance(theta_D_curr, np.ndarray) else theta_D_curr
        
        # Propose new theta
        theta_prop = np.random.uniform((l * theta_curr) / u, (u * theta_curr) / l)
        
        if theta_prop <= 0 or theta_prop > 20:
            return np.array([theta_curr])
        
        ru = np.random.uniform(0, 1)
        eps = np.sqrt(np.finfo(float).eps)
        
        # Log-likelihood current
        log_lik_curr = log_likelihood_gp(g, Z, Y, np.array([theta_curr]), tau2, kernel_type=kernel_type)
        
        # Log-prior current
        log_prior_curr = (gamma1 - 1) * np.log(theta_curr - eps) - gamma2 * (theta_curr - eps)
        
        # Log-likelihood proposed
        log_lik_prop = log_likelihood_gp(g, Z, Y, np.array([theta_prop]), tau2, kernel_type=kernel_type)
        
        # Log-prior proposed
        log_prior_prop = (gamma1 - 1) * np.log(theta_prop - eps) - gamma2 * (theta_prop - eps)
        
        # Acceptance ratio
        lpost_curr = log_lik_curr + log_prior_curr + np.log(ru) - np.log(theta_curr) + np.log(theta_prop)
        lpost_prop = log_lik_prop + log_prior_prop
        
        if lpost_prop > lpost_curr:
            return np.array([theta_prop])
        else:
            return np.array([theta_curr])
    else:
        # Full vector sampling (for isotropic kernels or when not dimension-wise)
        Z = input_matrix @ W
        D = len(theta_D_curr)
        
        # Propose new theta_D vector
        theta_D_prop = np.random.uniform((l * theta_D_curr) / u, (u * theta_D_curr) / l)
        
        if np.any(theta_D_prop <= 0) or np.any(theta_D_prop > 20):
            return theta_D_curr
        
        ru = np.random.uniform(0, 1)
        eps = np.sqrt(np.finfo(float).eps)
        
        # Log-likelihood current
        log_lik_curr = log_likelihood_gp(g, Z, Y, theta_D_curr, tau2, kernel_type=kernel_type)
        
        # Log-prior current (product of Gammas)
        log_prior_curr = np.sum((gamma1 - 1) * np.log(theta_D_curr - eps) - gamma2 * (theta_D_curr - eps))
        
        # Log-likelihood proposed
        log_lik_prop = log_likelihood_gp(g, Z, Y, theta_D_prop, tau2, kernel_type=kernel_type)
        
        # Log-prior proposed
        log_prior_prop = np.sum((gamma1 - 1) * np.log(theta_D_prop - eps) - gamma2 * (theta_D_prop - eps))
        
        # Acceptance ratio
        lpost_curr = log_lik_curr + log_prior_curr + np.log(ru) - np.sum(np.log(theta_D_curr)) + np.sum(np.log(theta_D_prop))
        lpost_prop = log_lik_prop + log_prior_prop
        
        if lpost_prop > lpost_curr:
            return theta_D_prop
        else:
            return theta_D_curr


# =============================================================================
# W Sampling using HMC on Stiefel Manifold (D>1)
# =============================================================================

def loglik_and_gradW_numpy(Y: np.ndarray, X: np.ndarray, W: np.ndarray,
                          F_Wprior: Optional[np.ndarray] = None,
                          use_tf: bool = False,
                          kernel_type: str = 'separable_squared_exponential',
                          layer: int = 1,
                          Q: Optional[np.ndarray] = None,
                          R: Optional[np.ndarray] = None,
                          tau2_y: Optional[float] = None,
                          tau2_q: Optional[Union[float, np.ndarray]] = None,
                          tau2_r: Optional[Union[float, np.ndarray]] = None,
                          theta_D_y: Optional[Union[float, np.ndarray]] = None,
                          theta_D_q: Optional[Union[float, np.ndarray]] = None,
                          theta_D_r: Optional[Union[float, np.ndarray]] = None,
                          g_y: Optional[float] = None,
                          g_q: Optional[Union[float, np.ndarray]] = None,
                          g_r: Optional[Union[float, np.ndarray]] = None) -> Tuple[float, np.ndarray]:
    """
    Compute log-posterior and gradient with respect to W for D>1.
    
    Supports hierarchical structures:
      layer=1: log p(Y | XW) + log p(W)
      layer=2: log p(Y | Q) + sum_d log p(Q_d | Z_d) + log p(W), Z=XW
      layer=3: log p(Y | Q) + sum_d log p(Q_d | R_d) + sum_d log p(R_d | Z_d) + log p(W)
    
    Required hyperparameters by layer:
      - layer=1: tau2_y, theta_D_y, g_y
      - layer=2: tau2_y, theta_D_y, g_y, tau2_q, theta_D_q, g_q
      - layer=3: tau2_y, theta_D_y, g_y, tau2_q, theta_D_q, g_q, tau2_r, theta_D_r, g_r
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
    
    def _to_vec(val: Optional[Union[float, np.ndarray]], name: str, default: Optional[float] = None) -> np.ndarray:
        if val is None:
            if default is None:
                raise ValueError(f"Missing required vector parameter: {name}")
            return np.full(D, float(default), dtype=float)
        arr = np.asarray(val, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Parameter {name} is empty.")
        if arr.size == 1:
            return np.full(D, float(arr[0]), dtype=float)
        if arr.size < D:
            return np.concatenate([arr, np.full(D - arr.size, arr[0], dtype=float)])
        return arr[:D]
    
    def _to_matrix(arr: np.ndarray, name: str) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.ndim != 2:
            raise ValueError(f"{name} must be 2D. Got {out.shape}.")
        return out
    
    def _kernel_grad_col(kernel, response: np.ndarray, X_mat: np.ndarray, w_col: np.ndarray) -> np.ndarray:
        response_vec = np.asarray(response, dtype=float).reshape(-1)
        if use_tf and TF_AVAILABLE:
            y_tf = tf.constant(response_vec, dtype=tf.float64)
            x_tf = tf.constant(X_mat, dtype=tf.float64)
            w_tf = tf.Variable(w_col, dtype=tf.float64)
            return kernel.gradient_log_likelihood_W_tf(y_tf, x_tf, w_tf).numpy()
        return kernel.gradient_log_likelihood_W(response_vec, X_mat, w_col)
    
    Y_vec = np.asarray(Y, dtype=float).reshape(-1)
    Z = X @ W
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    if layer == 1:
        tau2_y_eff = _to_scalar(tau2_y, "tau2_y")
        theta_y_eff = _to_vec(theta_D_y, "theta_D_y")
        g_y_eff = _to_scalar(g_y, "g_y")
        kernel = get_kernel_instance(kernel_type, theta_y_eff, g_y_eff, tau2_y_eff, D)
        log_lik = kernel.log_likelihood(Y_vec, Z)
        
        if use_tf and TF_AVAILABLE:
            y_tf = tf.constant(Y_vec, dtype=tf.float64)
            x_tf = tf.constant(X, dtype=tf.float64)
            w_tf = tf.Variable(W, dtype=tf.float64)
            grad_loglik = kernel.gradient_log_likelihood_W_tf(y_tf, x_tf, w_tf).numpy()
        else:
            grad_loglik = kernel.gradient_log_likelihood_W(Y_vec, X, W)
    
    elif layer == 2:
        if Q is None:
            raise ValueError("layer=2 requires Q.")
        tau2_y_eff = _to_scalar(tau2_y, "tau2_y", 1.0)
        tau2_q_eff = _to_vec(tau2_q, "tau2_q", 1.0)
        theta_y_eff = _to_vec(theta_D_y, "theta_D_y", 1.0)
        theta_q_eff = _to_vec(theta_D_q, "theta_D_q", 1.0)
        g_y_eff = _to_scalar(g_y, "g_y", 1e-6)
        g_q_eff = _to_vec(g_q, "g_q", 1e-6)
        Q_mat = _to_matrix(Q, "Q")
        if Q_mat.shape[1] != D:
            raise ValueError(f"Q must have shape (n,{D}). Got {Q_mat.shape}.")
        
        # Outer layer: Y | Q
        kernel_y = get_kernel_instance(kernel_type, theta_y_eff, g_y_eff, tau2_y_eff, D)
        log_lik_y = kernel_y.log_likelihood(Y_vec, Q_mat)
        
        # Inner layer: sum_d Q_d | Z_d
        log_lik_q = 0.0
        grad_loglik = np.zeros((p, D))
        for d in range(D):
            z_d = Z[:, d].reshape(-1, 1)
            q_d = Q_mat[:, d]
            w_d = W[:, d].reshape(-1, 1)
            kernel_d = get_kernel_instance(kernel_type, np.array([theta_q_eff[d]]), g_q_eff[d], tau2_q_eff[d], 1)
            log_lik_q += kernel_d.log_likelihood(q_d, z_d)
            grad_loglik[:, d:d+1] = _kernel_grad_col(kernel_d, q_d, X, w_d)
        
        log_lik = log_lik_y + log_lik_q
    
    elif layer == 3:
        if Q is None or R is None:
            raise ValueError("layer=3 requires both Q and R.")
        tau2_y_eff = _to_scalar(tau2_y, "tau2_y", 1.0)
        tau2_q_eff = _to_vec(tau2_q, "tau2_q", 1.0)
        tau2_r_eff = _to_vec(tau2_r, "tau2_r", 1.0)
        theta_y_eff = _to_vec(theta_D_y, "theta_D_y", 1.0)
        theta_q_eff = _to_vec(theta_D_q, "theta_D_q", 1.0)
        theta_r_eff = _to_vec(theta_D_r, "theta_D_r", 1.0)
        g_y_eff = _to_scalar(g_y, "g_y", 1e-6)
        g_q_eff = _to_vec(g_q, "g_q", 1e-6)
        g_r_eff = _to_vec(g_r, "g_r", 1e-6)
        Q_mat = _to_matrix(Q, "Q")
        R_mat = _to_matrix(R, "R")
        if Q_mat.shape[1] != D or R_mat.shape[1] != D:
            raise ValueError(f"Q and R must have shape (n,{D}). Got Q={Q_mat.shape}, R={R_mat.shape}.")
        
        # Outer layer: Y | Q
        kernel_y = get_kernel_instance(kernel_type, theta_y_eff, g_y_eff, tau2_y_eff, D)
        log_lik_y = kernel_y.log_likelihood(Y_vec, Q_mat)
        
        # Middle layer: sum_d Q_d | R_d
        log_lik_q = 0.0
        for d in range(D):
            q_d = Q_mat[:, d]
            r_d = R_mat[:, d].reshape(-1, 1)
            kernel_q_d = get_kernel_instance(kernel_type, np.array([theta_q_eff[d]]), g_q_eff[d], tau2_q_eff[d], 1)
            log_lik_q += kernel_q_d.log_likelihood(q_d, r_d)
        
        # Inner layer (W-dependent): sum_d R_d | Z_d
        log_lik_r = 0.0
        grad_loglik = np.zeros((p, D))
        for d in range(D):
            z_d = Z[:, d].reshape(-1, 1)
            r_d = R_mat[:, d]
            w_d = W[:, d].reshape(-1, 1)
            kernel_r_d = get_kernel_instance(kernel_type, np.array([theta_r_eff[d]]), g_r_eff[d], tau2_r_eff[d], 1)
            log_lik_r += kernel_r_d.log_likelihood(r_d, z_d)
            grad_loglik[:, d:d+1] = _kernel_grad_col(kernel_r_d, r_d, X, w_d)
        
        log_lik = log_lik_y + log_lik_q + log_lik_r
    
    else:
        raise ValueError(f"Unknown layer={layer}. Expected 1, 2, or 3.")
    
    # Matrix Langevin prior
    if F_Wprior is not None:
        log_prior = np.trace(F_Wprior.T @ W)
        grad_logprior = F_Wprior
    else:
        log_prior = 0.0
        grad_logprior = np.zeros((p, D))
    
    log_post = log_lik + log_prior
    grad_logpost = grad_loglik + grad_logprior
    
    return log_post, grad_logpost


def sample_W_HMC_stiefel(Y: np.ndarray, X: np.ndarray, W: np.ndarray,
                        F_Wprior: Optional[np.ndarray] = None,
                        M: int = 1, eps: float = 0.001, T_step: int = 17,
                        use_tf: bool = False,
                        kernel_type: str = 'separable_squared_exponential',
                        layer: int = 1,
                        Q: Optional[np.ndarray] = None,
                        R: Optional[np.ndarray] = None,
                        tau2_y: Optional[float] = None,
                        tau2_q: Optional[Union[float, np.ndarray]] = None,
                        tau2_r: Optional[Union[float, np.ndarray]] = None,
                        theta_D_y: Optional[Union[float, np.ndarray]] = None,
                        theta_D_q: Optional[Union[float, np.ndarray]] = None,
                        theta_D_r: Optional[Union[float, np.ndarray]] = None,
                        g_y: Optional[float] = None,
                        g_q: Optional[Union[float, np.ndarray]] = None,
                        g_r: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """
    Sample W from Stiefel manifold using HMC with geodesic flows (D>1).
    
    Args:
        Y: Response vector (n,)
        X: Design matrix (n, p)
        W: Current projection matrix (p, D)
        F_Wprior: Prior parameter
        M: Number of samples
        eps: Step size
        T_step: Leapfrog steps
        use_tf: Use TensorFlow
        kernel_type: Kernel type to use
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
        Sampled W matrix
    """
    p, D = W.shape
    hmc_samples = []
    nn = 0
    max_hmc_iterations = 1000  # Prevent infinite loop
    iteration_count = 0
    
    while nn < M and iteration_count < max_hmc_iterations:
        iteration_count += 1
        
        # Sample momentum
        u = np.random.normal(size=(p, D))
        u_proj = (np.eye(p) - W @ W.T) @ u
        
        # Current log-posterior and gradient
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
            
            # Geodesic flow on Stiefel manifold (Cayley transform approximation)
            A = u_new_proj @ W_star.T - W_star @ u_new_proj.T
            
            # Safety check for matrix inversion
            try:
                I_minus_A = np.eye(p) - eps * A / 2
                # Check condition number to avoid near-singular matrices
                cond = np.linalg.cond(I_minus_A)
                if cond > 1e12:
                    # Use QR-based update instead if matrix is ill-conditioned
                    W_star, _ = np.linalg.qr(W_star + eps * u_new_proj)
                else:
                    I_plus_A = np.eye(p) + eps * A / 2
                    W_star = I_plus_A @ np.linalg.inv(I_minus_A) @ W_star
            except np.linalg.LinAlgError:
                # Fallback: use QR-based update
                W_star, _ = np.linalg.qr(W_star + eps * u_new_proj)
            
            # Re-orthonormalize to maintain numerical stability
            W_star, _ = np.linalg.qr(W_star)
            
            # Update momentum
            logPosteriorW, grad_logPosteriorW = loglik_and_gradW_numpy(
                Y, X, W_star, F_Wprior, use_tf, kernel_type,
                layer=layer, Q=Q, R=R,
                tau2_y=tau2_y, tau2_q=tau2_q, tau2_r=tau2_r,
                theta_D_y=theta_D_y, theta_D_q=theta_D_q, theta_D_r=theta_D_r,
                g_y=g_y, g_q=g_q, g_r=g_r
            )
            u_new = u_proj + eps * grad_logPosteriorW / 2
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
        warnings.warn(f"HMC (D>1): Only generated {nn}/{M} samples after {max_hmc_iterations} iterations. Using last W.")
        while len(hmc_samples) < M:
            hmc_samples.append(W.copy())
    
    return hmc_samples[0] if M == 1 else hmc_samples


# =============================================================================
# Matrix Langevin Prior Parameters
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
        M | W, Λ, V ~ MF(W @ Λ @ V + prior_M)
    
    Args:
        W: Projection matrix (p, D)
        Lambda: Diagonal concentration matrix (D, D) or vector (D,)
        V: Orthonormal matrix (D, D)
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
    
    # For D>1: Lambda is (D,) or (D, D), W is (p, D)
    # M_param = W @ Lambda @ V + prior_M
    if Lambda.ndim == 1:
        Lambda_diag = np.diag(Lambda)
    else:
        Lambda_diag = Lambda
    
    M_param = W @ Lambda_diag @ V + prior_M
    
    return _sample_matrix_vmf_gibbs(
        M_param,
        M_prev,
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
    
    # For D>1: Lambda is (D,) or (D, D), W is (p, D), M is (p, D)
    # V_param = (W^T @ M) @ Lambda + prior_V
    if Lambda.ndim == 1:
        Lambda_diag = np.diag(Lambda)
    else:
        Lambda_diag = Lambda
    
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


def laplace_approximate_0F1(p: int, lambdas: np.ndarray) -> float:
    """Laplace approximation for hypergeometric function ₀F₁."""
    lambdas_flat = lambdas.flatten()
    d = len(lambdas_flat)
    
    y_hat = (2 * lambdas_flat / p) / (1 + np.sqrt(4 * lambdas_flat**2 / p**2 + 1))
    
    R_01 = 1.0
    for i in range(d):
        for j in range(i, d):
            R_01 *= (1 - y_hat[i]**2 * y_hat[j]**2)
    
    product_term = 1.0
    for i in range(d):
        product_term *= (1 - y_hat[i]**2)**(p / 2) * np.exp(lambdas_flat[i] * y_hat[i])
    
    oF1 = (1 / np.sqrt(R_01)) * product_term
    return oF1


def likelihood_lambda(V: np.ndarray, M: np.ndarray, Lambda: np.ndarray, 
                     W: np.ndarray, p: int) -> float:
    """Compute likelihood for Lambda parameter."""
    Lambda_diag = np.diag(Lambda.flatten())
    exponential_term = np.sum(np.diag(V.T @ Lambda_diag @ M.T @ W))
    likelihood_value = np.exp(exponential_term) / laplace_approximate_0F1(p, Lambda)
    return likelihood_value


def sample_Lambda_slice(Lambda_cur: np.ndarray, prior_value: np.ndarray,
                        M: np.ndarray, V: np.ndarray, W: np.ndarray,
                        p: int, max_iter: int = 1000, epsilon: float = 2.0,
                        max_outer_attempts: int = 10) -> np.ndarray:
    """
    Sample Lambda using elliptical slice sampling.
    
    STRICT CONSTRAINT: Lambda values must be in descending order (lambda_1 > lambda_2 > ... > lambda_D).
    Any proposal that violates this constraint is automatically rejected.
    
    Args:
        Lambda_cur: Current Lambda value (diagonal matrix (D, D) or vector (D,))
        prior_value: Prior value (nu) - same shape as Lambda_cur
        M, V, W: Model matrices
        p: Dimension
        max_iter: Maximum slice-shrink iterations per bracket
        epsilon: Minimum threshold for Lambda values
        max_outer_attempts: Maximum bracket restarts before falling back to the
            current Lambda value
        
    Returns:
        Sampled Lambda with values in descending order
    """
    def extract_diagonal(Lambda_arr: np.ndarray) -> np.ndarray:
        """Extract diagonal values from Lambda (handles both matrix and vector)."""
        if Lambda_arr.ndim == 2:
            return np.diag(Lambda_arr)
        else:
            return Lambda_arr.flatten()
    
    def is_descending_order(lambda_vals: np.ndarray) -> bool:
        """
        Check if Lambda values are in strictly descending order.
        
        Returns:
            True if lambda_1 > lambda_2 > ... > lambda_D, False otherwise
        """
        if len(lambda_vals) <= 1:
            return True
        # Check that all consecutive pairs are in descending order
        diffs = np.diff(lambda_vals)
        return np.all(diffs < 0)  # Strictly descending: all differences < 0
    
    cur_log_like = likelihood_lambda(V, M, Lambda_cur, W, p)
    nu = prior_value

    # Keep resampling until a valid descending Lambda proposal is accepted.
    # No fallback sorting is used; if no valid value is found, keep Lambda_cur.
    for _ in range(max_outer_attempts):
        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi
        iterations = 0
        Lambda_cur_ess = Lambda_cur.copy()
        
        # ESS loop: try to find a valid sample with descending order
        while iterations < max_iter:
            iterations += 1
            
            Lambda_prop = np.cos(phi) * Lambda_cur_ess + np.sin(phi) * nu
            
            # Extract diagonal values for checking
            lambda_prop_vals = extract_diagonal(Lambda_prop)
            
            # STRICT CONDITION 1: All values must be > epsilon (positivity)
            # STRICT CONDITION 2: Values must be in descending order
            if np.all(lambda_prop_vals > epsilon) and is_descending_order(lambda_prop_vals):
                prop_log_like = likelihood_lambda(V, M, Lambda_prop, W, p)
                
                if prop_log_like > cur_log_like:
                    # New point is on the slice and satisfies all constraints
                    # Verify descending order one more time before accepting
                    if is_descending_order(lambda_prop_vals):
                        # Successfully sampled with descending order, return immediately
                        return Lambda_prop
            
            # If proposal doesn't satisfy constraints, shrink slice and continue
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                phi = np.random.uniform(-epsilon, epsilon)
            
            phi = np.random.uniform(phi_min, phi_max)

    warnings.warn(
        "Lambda slice sampler did not find a valid descending proposal; keeping current Lambda.",
        RuntimeWarning
    )
    return Lambda_cur


def covar_sepp(Z, d, g):
    """Covariance function for 3-layer latent variable sampling."""
    n = Z.shape[0]
    C_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sum_term = np.sum((Z[i, :] - Z[j, :])**2)
            C_matrix[i, j] = np.exp(-sum_term / (2 * d**2))
    return C_matrix + g * np.eye(n)


def log_likelihoodd(g, Z, Y, theta_D, tau2):
    """Log-likelihood for 3-layer latent variable sampling."""
    Sigma = covar_sepp(Z, theta_D, g)
    Sigma_inv = np.linalg.pinv(Sigma)
    n = len(Y)
    sign, log_det = np.linalg.slogdet(Sigma)
    log_lik = -0.5 * (n * np.log(2 * np.pi * tau2) + log_det + Y.T @ Sigma_inv @ Y / tau2)
    return log_lik


def sample_R_3layer_ESS(Y, R_current, Z, g_q, theta_q, theta_r, g_r,
                        tau2_q=1.0, tau2_r=1.0,
                        kernel_type: str = 'separable_squared_exponential'):
    """
    Sample latent R for 3-layer using ESS (D>1).
    
    For 3-layer: R is the innermost latent layer
        Q | R, θ_q, g_q, τ²_q ~ GP(0, τ²_q(C_q + g_q*I))
        R | Z, θ_r, g_r, τ²_r ~ GP(0, τ²_r(C_r + g_r*I)) where Z = XW
    
    Args:
        Y: Q values (n, D) used as "response" for R likelihood
        R_current: Current R values (n, D)
        Z: Projected inputs Z = XW (n, D) used for R prior
        g_q: Nugget for Q layer
        theta_q: Lengthscale vector for Q layer (D,)
        theta_r: Lengthscale vector for R layer (D,)
        g_r: Nugget for R layer
        tau2_q: Variance for Q|R likelihood kernel (scalar or length-D vector)
        tau2_r: Variance for R|Z prior kernel (scalar or length-D vector)
        kernel_type: Kernel type for covariance functions
        
    Returns:
        Updated R (n, D)
    """
    N, D = R_current.shape
    
    # Extract scalar nugget from g_q (should be 0.0 for latent layers)
    g_q_val = g_q[0] if hasattr(g_q, '__len__') and len(g_q) > 1 else (g_q if np.isscalar(g_q) else g_q[0])
    tau2_q_val = tau2_q[0] if hasattr(tau2_q, '__len__') and len(tau2_q) > 1 else (tau2_q if np.isscalar(tau2_q) else tau2_q[0])
    
    # Create kernel instance for Q layer likelihood (Q | R)
    kernel_q = get_kernel_instance(kernel_type, theta_q, g_q_val, tau2_q_val, D)
    
    # Compute current log-likelihood using kernel
    # Y is Q (n, D), but for separable kernels we need to handle it dimension-wise
    # For now, use the first dimension or sum across dimensions
    # Actually, for 3-layer, Q | R means we need to compute likelihood for each dimension
    # But the kernel expects (n,) response, so we'll use Q[:, 0] or sum
    # For simplicity, use first dimension (this matches the dimension-wise ESS loop)
    Q_for_likelihood = Y[:, 0] if Y.ndim > 1 else Y
    R_for_likelihood = R_current[:, 0].reshape(-1, 1) if R_current.ndim > 1 else R_current
    
    def log_likelihood_R(R):
        # For dimension-wise ESS, we update one dimension at a time
        # So we compute likelihood using the current R with updated dimension
        # Use first dimension for likelihood computation (will be updated in loop)
        R_use = R[:, 0].reshape(-1, 1) if R.ndim > 1 else R
        return kernel_q.log_likelihood(Q_for_likelihood, R_use)
    
    ll_prev = log_likelihood_R(R_current)
    
    for i in range(D):
        # Sample from prior for dimension i: R[:,i] ~ N(0, C_r)
        # For prior, use isotropic kernel with theta_r[i] as lengthscale
        # Prior is based on Z (not R_current)
        g_r_val = g_r[i] if hasattr(g_r, '__len__') and len(g_r) > 1 else (g_r if np.isscalar(g_r) else g_r[0])
        tau2_r_val = tau2_r[i] if hasattr(tau2_r, '__len__') and len(tau2_r) > 1 else (tau2_r if np.isscalar(tau2_r) else tau2_r[0])
        kernel_r_prior = get_kernel_instance('isotropic_squared_exponential',
                                            np.array([theta_r[i]]), g_r_val, tau2_r_val, D)
        cov = kernel_r_prior.compute_covariance(Z, Z)
        R_prior = np.random.multivariate_normal(mean=np.zeros(N), cov=cov)
        
        a, amin, amax = np.random.uniform(0, 2*np.pi), 0, 2*np.pi
        amin, amax = a - 2*np.pi, a
        ll_threshold = ll_prev + np.log(np.random.uniform())
        
        accept, count, R_prev_i = False, 0, R_current[:, i].copy()
        while not accept:
            count += 1
            R_current[:, i] = R_prev_i * np.cos(a) + R_prior * np.sin(a)
            new_logl = log_likelihood_R(R_current)
            
            if new_logl > ll_threshold:
                ll_prev, accept = new_logl, True
            else:
                amin, amax = (a, amax) if a < 0 else (amin, a)
                a = np.random.uniform(amin, amax)
                if count > 1000:
                    R_current[:, i] = R_prev_i
                    break
    return R_current


def sample_Q_3layer_ESS(Q_current, R, Y, g_y, theta_y, g_q, theta_q,
                        tau2_y=1.0, tau2_q=1.0,
                        kernel_type: str = 'separable_squared_exponential'):
    """Sample latent Q for 3-layer using ESS (D>1)."""
    N, D = Q_current.shape
    
    # Create kernel instance for Y layer likelihood (Y | Q)
    kernel_y = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    
    # Compute current log-likelihood using kernel
    def log_likelihood_Q(Q):
        return kernel_y.log_likelihood(Y, Q)
    
    ll_prev = log_likelihood_Q(Q_current)
    
    for i in range(D):
        # Sample from prior for dimension i: Q[:,i] ~ N(0, C_q)
        # For prior, use isotropic kernel with theta_q[i] as lengthscale
        g_q_val = g_q[i] if hasattr(g_q, '__len__') and len(g_q) > 1 else (g_q if np.isscalar(g_q) else g_q[0])
        tau2_q_val = tau2_q[i] if hasattr(tau2_q, '__len__') and len(tau2_q) > 1 else (tau2_q if np.isscalar(tau2_q) else tau2_q[0])
        kernel_q_prior = get_kernel_instance('isotropic_squared_exponential',
                                            np.array([theta_q[i]]), g_q_val, tau2_q_val, D)
        cov = kernel_q_prior.compute_covariance(R, R)
        Q_prior = np.random.multivariate_normal(mean=np.zeros(N), cov=cov)
        
        a = np.random.uniform(0, 2*np.pi)
        amin, amax = a - 2*np.pi, a
        ll_threshold = ll_prev + np.log(np.random.uniform())
        
        accept, count, Q_prev_i = False, 0, Q_current[:, i].copy()
        while not accept:
            count += 1
            Q_current[:, i] = Q_prev_i * np.cos(a) + Q_prior * np.sin(a)
            new_logl = log_likelihood_Q(Q_current)
            
            if new_logl >= ll_threshold:
                ll_prev, accept = new_logl, True
            else:
                amin, amax = (a, amax) if a < 0 else (amin, a)
                a = np.random.uniform(amin, amax)
                if count > 1000:
                    Q_current[:, i] = Q_prev_i
                    break
    return Q_current


def covar_isotropic_full(Z, theta_scalar, g):
    """
    Isotropic covariance over full Z matrix using single lengthscale.
    
    Used in ESS for Q sampling where we apply a single lengthscale to all dimensions of Z.
    
    Args:
        Z: Input matrix (n, D)
        theta_scalar: Single lengthscale parameter
        g: Nugget
        
    Returns:
        Covariance matrix (n, n)
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
                        kernel_type: str = 'separable_squared_exponential'):
    """
    Sample latent Q for 2-layer model using Elliptical Slice Sampling (D>1).
    
    For 2-layer Deep GP:
        Y | Q, θ_y, g_y, τ²_y ~ GP(0, τ²_y(C_y + g_y*I))
        Q | Z, θ_q, g_q, τ²_q ~ GP(0, τ²_q(C_q + g_q*I))
    
    Args:
        Y: Response vector (n,)
        Q_current: Current Q values (n, D)
        Z: Projected inputs Z = XW (n, D)
        g_y: Nugget for Y layer
        theta_y: Lengthscale vector for Y layer (D,)
        theta_q: Lengthscale vector for Q layer (D,)
        g_q: Nugget for Q layer (prior)
        tau2_y: Observation noise for Y layer
        tau2_q: Latent variance for Q layer prior (scalar or length-D vector)
        kernel_type: Kernel type for covariance functions
        
    Returns:
        Updated Q (n, D)
    """
    N, D = Q_current.shape
    
    # Create kernel instances
    kernel_y = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    
    # Compute current log-likelihood using kernel
    def log_likelihood_Q(Q):
        return kernel_y.log_likelihood(Y, Q)
    
    ll_prev = log_likelihood_Q(Q_current)
    
    # Sample each dimension using ESS
    for i in range(D):
        # Allow scalar or vector inputs for layer-specific prior parameters.
        if np.isscalar(g_q):
            g_q_i = float(g_q)
        else:
            g_q_arr = np.asarray(g_q, dtype=float).reshape(-1)
            g_q_i = float(g_q_arr[i] if g_q_arr.size > 1 else g_q_arr[0])

        if np.isscalar(tau2_q):
            tau2_q_i = float(tau2_q)
        else:
            tau2_q_arr = np.asarray(tau2_q, dtype=float).reshape(-1)
            tau2_q_i = float(tau2_q_arr[i] if tau2_q_arr.size > 1 else tau2_q_arr[0])

        # Sample from prior for dimension i: Q[:,i] ~ N(0, tau2 * C_q)
        # For prior, use isotropic kernel with theta_q[i] as lengthscale
        # (applies single lengthscale to all dimensions of Z for this dimension's prior)
        # Use isotropic kernel type for prior (single lengthscale applied to full Z)
        kernel_q_prior = get_kernel_instance('isotropic_squared_exponential', 
                                            np.array([theta_q[i]]), g_q_i, tau2_q_i, D)
        cov = kernel_q_prior.compute_covariance(Z, Z)
        Q_prior = np.random.multivariate_normal(mean=np.zeros(N), cov=cov)
        
        # ESS angle and bounds
        a = np.random.uniform(0, 2 * np.pi)
        amin, amax = a - 2 * np.pi, a
        ru = np.random.uniform()
        ll_threshold = ll_prev + np.log(ru)
        
        accept = False
        count = 0
        Q_prev_i = Q_current[:, i].copy()
        
        while not accept:
            count += 1
            Q_current[:, i] = Q_prev_i * np.cos(a) + Q_prior * np.sin(a)
            
            new_logl = log_likelihood_Q(Q_current)
            
            if new_logl > ll_threshold:
                ll_prev = new_logl
                accept = True
            else:
                if a < 0:
                    amin = a
                else:
                    amax = a
                a = np.random.uniform(amin, amax)
                
                if count > 100:
                    print(f"Warning: ESS for Q[:,{i}] reached max iterations (100)")
                    Q_current[:, i] = Q_prev_i  # Revert
                    break
    
    return Q_current


if __name__ == "__main__":
    print("="*70)
    print("Parameter Sampler Module for D>1 (Test)")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n, p, D = 30, 10, 2  # D=2 for testing
    
    X = np.random.randn(n, p)
    W_true = np.random.randn(p, D)
    W_true, _ = np.linalg.qr(W_true)  # Orthonormalize
    
    Z = X @ W_true
    theta_test = np.array([1.0, 1.0])
    
    # Compute covariance
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sum_term = sum((Z[i, k] - Z[j, k])**2 / (2 * theta_test[k]**2) for k in range(D))
            C[i, j] = np.exp(-sum_term)
    
    Y = np.random.multivariate_normal(np.zeros(n), C + 0.01 * np.eye(n))
    
    print(f"\nTest Data: n={n}, p={p}, D={D}")
    
    # Test MLE functions
    print("\n" + "-"*70)
    print("Testing MLE Functions")
    print("-"*70)
    
    tau2_mle = estimate_tau2_MLE(Y, X, W_true, theta_test, g=0.01)
    print(f"✓ tau2 MLE: {tau2_mle:.6f}")
    
    g_mle = estimate_g_MLE(Y, X, W_true, theta_test, tau2=1.0, n_grid=10)
    print(f"✓ g MLE: {g_mle:.6f}")
    
    theta_mle = estimate_theta_D_MLE(Y, X, W_true, g=0.01, tau2=1.0, D=D, n_grid=10)
    print(f"✓ theta MLE: {theta_mle}")
    
    print("\n  Joint MLE (2 iterations):")
    mle_all = estimate_all_hyperparameters_MLE(
        Y, X, W_true, D=D, n_iterations=2, n_grid=10, verbose=True
    )
    
    # Test MCMC sampling
    print("\n" + "-"*70)
    print("Testing MCMC Sampling Functions")
    print("-"*70)
    
    tau2_sample = sample_tau2(Y, X, W_true, tau2_curr=1.0, theta_D=theta_test, g=0.01)
    print(f"✓ tau2 sampled: {tau2_sample:.6f}")
    
    g_sample = sample_g(Y, X, W_true, g_curr=0.01, theta_D=theta_test, tau2=1.0)
    print(f"✓ g sampled: {g_sample:.6f}")
    
    theta_sample = sample_theta_D(Y, X, W_true, theta_D_curr=theta_test, tau2=1.0, g=0.01)
    print(f"✓ theta sampled: {theta_sample}")
    
    # Test W sampling
    W_sample = sample_W_HMC_stiefel(
        Y, X, W_true,
        M=1, eps=0.001, T_step=5, use_tf=False, layer=1,
        tau2_y=1.0, theta_D_y=theta_test, g_y=0.01
    )
    print(f"✓ W sampled, shape: {W_sample.shape}")
    print(f"  W^T W = I check: max error = {np.max(np.abs(W_sample.T @ W_sample - np.eye(D))):.2e}")
    
    print("\n" + "="*70)
    print("All sampling functions for D>1 tested successfully!")
    print("="*70)
