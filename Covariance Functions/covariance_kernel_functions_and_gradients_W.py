"""
Covariance Kernel Functions and Gradients for GP with Dimensionality Reduction

This module implements various covariance kernels for Gaussian Process models
with Bayesian dimensionality reduction, including:
    1. Isotropic Squared Exponential (SE) kernel
    2. Separable Squared Exponential kernel
    3. Separable Matérn-3/2 kernel
    4. Isotropic Matérn-3/2 kernel

For each kernel, the following are implemented:
    - Covariance function C(Z, Z')
    - Gradient of C with respect to W
    - Log-likelihood function
    - Gradient of log-likelihood with respect to W

Both NumPy and TensorFlow implementations are provided where applicable.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings

# Optional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    class _TensorFlowStub:
        Tensor = Any
        Variable = Any
    tf = _TensorFlowStub()  # type: ignore[assignment]
    warnings.warn("TensorFlow not available. TensorFlow-based methods will not work.")


# =============================================================================
# Base Kernel Class
# =============================================================================

class BaseKernel:
    """Base class for GP covariance kernels with dimensionality reduction."""
    
    def __init__(self, lengthscales: np.ndarray, nugget: float = 1e-6, tau2: float = 1.0):
        """
        Initialize kernel.
        
        Args:
            lengthscales: Lengthscale parameters (scalar or vector)
            nugget: Nugget parameter for numerical stability (g in equations)
            tau2: Observation noise variance (τ²)
        """
        self.lengthscales = np.atleast_1d(lengthscales)
        self.nugget = nugget
        self.tau2 = tau2
    
    def compute_covariance(self, Z: np.ndarray, Z_prime: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute covariance matrix. To be implemented by subclasses."""
        raise NotImplementedError
    
    def compute_sigma(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute Σ_y = τ²(C_y + g*I_n).
        
        Args:
            Z: Reduced dimensional inputs (n, D)
            
        Returns:
            Sigma matrix (n, n)
        """
        C = self.compute_covariance(Z)
        n = C.shape[0]
        Sigma = self.tau2 * (C + self.nugget * np.eye(n))
        return Sigma
    
    def log_likelihood(self, Y: np.ndarray, Z: np.ndarray) -> float:
        """
        Compute log-likelihood: log p(Y | Z, θ).
        
        log L = -0.5 * [log|Σ| + Y^T Σ^{-1} Y + n*log(2π)]
        
        Args:
            Y: Response vector (n,)
            Z: Reduced inputs (n, D)
            
        Returns:
            Log-likelihood value
        """
        Y = Y.reshape(-1, 1)
        n = len(Y)
        
        Sigma = self.compute_sigma(Z)
        
        # Compute log determinant using Cholesky decomposition
        try:
            L = np.linalg.cholesky(Sigma)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            # Fallback to direct computation
            sign, log_det = np.linalg.slogdet(Sigma)
            if sign <= 0:
                return -np.inf
        
        # Solve Σ^{-1} Y using Cholesky
        alpha = np.linalg.solve(Sigma, Y)
        
        # Compute log-likelihood
        quad_form = Y.T @ alpha
        log_lik = -0.5 * (log_det + quad_form[0, 0] + n * np.log(2 * np.pi))
        
        return log_lik
    
    def gradient_log_likelihood_W(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log-likelihood with respect to W.
        To be implemented by subclasses.
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            
        Returns:
            Gradient matrix (p, D)
        """
        raise NotImplementedError


# =============================================================================
# 1. Isotropic Squared Exponential Kernel
# =============================================================================

class IsotropicSquaredExponentialKernel(BaseKernel):
    """
    Isotropic Squared Exponential (RBF) Kernel.
    
    Mathematical form:
        C_{ij} = exp(-||z_i - z_j||² / (2θ))
    
    where z_i = W^T x_i and θ is the lengthscale parameter.
    """
    
    def __init__(self, lengthscale: float, nugget: float = 1e-6, tau2: float = 1.0):
        """
        Initialize isotropic SE kernel.
        
        Args:
            lengthscale: Single lengthscale parameter (θ_y)
            nugget: Nugget parameter (g)
            tau2: Observation noise variance (τ²)
        """
        super().__init__(lengthscales=np.array([lengthscale]), nugget=nugget, tau2=tau2)
        self.theta = lengthscale
    
    def compute_covariance(self, Z: np.ndarray, Z_prime: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute isotropic SE covariance matrix.
        
        Args:
            Z: Input points (n, D)
            Z_prime: Second set of points (m, D). If None, uses Z.
            
        Returns:
            Covariance matrix (n, m)
        """
        if Z_prime is None:
            Z_prime = Z
        
        # Compute squared distances: ||z_i - z_j||²
        # Using broadcasting: (n, 1, D) - (1, m, D) = (n, m, D)
        diff = Z[:, np.newaxis, :] - Z_prime[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        
        # Compute covariance
        C = np.exp(-sq_dist / (2.0 * self.theta))
        
        return C
    
    def compute_D_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute D matrix where D_{ij} = 0.5 * ||x_i - x_j||².
        
        Args:
            X: Input matrix (n, p)
            
        Returns:
            D matrix (n, n)
        """
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D = 0.5 * np.sum(diff**2, axis=2)
        return D
    
    def gradient_covariance_W(self, X: np.ndarray, W: np.ndarray, 
                             C: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradient of covariance matrix with respect to W.
        
        Using equation (2.3):
            ∂C/∂W = -(1/θ) * (C ⊙ D) * X^T * (XW)
        
        where ⊙ is Hadamard product and D_{ij} = 0.5||x_i - x_j||².
        
        Args:
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            C: Precomputed covariance matrix (n, n). If None, computed.
            
        Returns:
            Gradient tensor (p, D)
        """
        Z = X @ W
        
        if C is None:
            C = self.compute_covariance(Z)
        
        # Compute D matrix
        D = self.compute_D_matrix(X)
        
        # Hadamard product: C ⊙ D
        C_hadamard_D = C * D
        
        # Gradient: -(1/θ) * (C ⊙ D) * X^T * (XW)
        grad_W = -(1.0 / self.theta) * (X.T @ C_hadamard_D @ X @ W)
        
        return grad_W
    
    def gradient_log_likelihood_W(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log-likelihood with respect to W.
        
        Using chain rule:
            ∂ℓ/∂W = tr(H * ∂C/∂W)
        
        where H = (1/τ²)αα^T - Σ^{-1} and α = Σ^{-1}Y.
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            
        Returns:
            Gradient matrix (p, D)
        """
        Y = Y.reshape(-1, 1)
        n = len(Y)
        
        Z = X @ W
        Sigma = self.compute_sigma(Z)
        C = self.compute_covariance(Z)
        
        # Compute α = Σ^{-1}Y
        alpha = np.linalg.solve(Sigma, Y)
        
        # Compute H = (1/τ²)αα^T - Σ^{-1}
        Sigma_inv = np.linalg.inv(Sigma)
        H = (1.0 / self.tau2) * (alpha @ alpha.T) - Sigma_inv
        
        # Compute D matrix
        D = self.compute_D_matrix(X)
        
        # Compute weighted matrix: H ⊙ C ⊙ D
        weighted = H * C * D
        
        # Gradient
        grad_W = -(1.0 / self.theta) * (X.T @ weighted @ X @ W)
        
        return grad_W
    
    def gradient_log_likelihood_W_tf(self, Y: tf.Tensor, X: tf.Tensor, 
                                     W: tf.Variable) -> tf.Tensor:
        """
        TensorFlow automatic differentiation for gradient computation.
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D) as TensorFlow Variable
            
        Returns:
            Gradient tensor (p, D)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        with tf.GradientTape() as tape:
            tape.watch(W)
            
            # Compute Z = XW
            Z = tf.matmul(X, W)
            
            # Compute covariance
            diff = tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0)
            sq_dist = tf.reduce_sum(diff**2, axis=2)
            theta_tf = tf.cast(self.theta, dtype=sq_dist.dtype)
            C = tf.exp(-sq_dist / (2.0 * theta_tf))
            
            # Compute Σ
            n = tf.shape(C)[0]
            tau2_tf = tf.cast(self.tau2, dtype=C.dtype)
            nugget_tf = tf.cast(self.nugget, dtype=C.dtype)
            Sigma = tau2_tf * (C + nugget_tf * tf.eye(n, dtype=C.dtype))
            
            # Compute log-likelihood
            Y_reshaped = tf.reshape(Y, [-1, 1])
            L = tf.linalg.cholesky(Sigma)
            log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
            alpha = tf.linalg.cholesky_solve(L, Y_reshaped)
            quad_form = tf.matmul(Y_reshaped, alpha, transpose_a=True)
            n_float = tf.cast(n, dtype=Y.dtype)
            pi_val = tf.cast(np.pi, dtype=Y.dtype)
            log_lik = -0.5 * (log_det + quad_form[0, 0] + n_float * tf.math.log(2.0 * pi_val))
        
        # Compute gradient
        grad_W = tape.gradient(log_lik, W)
        
        return grad_W


# =============================================================================
# 2. Separable Squared Exponential Kernel
# =============================================================================

class SeparableSquaredExponentialKernel(BaseKernel):
    """
    Separable Squared Exponential Kernel with dimension-specific lengthscales.
    
    Mathematical form:
        C_{ij} = exp(-∑_{ℓ=1}^D (z_{iℓ} - z_{jℓ})² / (2θ_ℓ²))
    
    where each dimension has its own lengthscale θ_ℓ.
    """
    
    def __init__(self, lengthscales: np.ndarray, nugget: float = 1e-6, tau2: float = 1.0):
        """
        Initialize separable SE kernel.
        
        Args:
            lengthscales: Lengthscale for each dimension (D,)
            nugget: Nugget parameter (g)
            tau2: Observation noise variance (τ²)
        """
        super().__init__(lengthscales=lengthscales, nugget=nugget, tau2=tau2)
        self.D = len(lengthscales)
    
    def compute_covariance(self, Z: np.ndarray, Z_prime: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute separable SE covariance matrix.
        
        Args:
            Z: Input points (n, D)
            Z_prime: Second set of points (m, D). If None, uses Z.
            
        Returns:
            Covariance matrix (n, m)
        """
        if Z_prime is None:
            Z_prime = Z
        
        n, D = Z.shape
        m = Z_prime.shape[0]
        
        # Compute weighted squared distances per dimension
        diff = Z[:, np.newaxis, :] - Z_prime[np.newaxis, :, :]  # (n, m, D)
        
        # Scale by lengthscales: (z_{iℓ} - z_{jℓ})² / (2θ_ℓ²)
        scaled_sq_diff = (diff**2) / (2.0 * self.lengthscales**2)
        
        # Sum over dimensions and exponentiate
        C = np.exp(-np.sum(scaled_sq_diff, axis=2))
        
        return C
    
    def gradient_log_likelihood_W(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log-likelihood with respect to W.
        
        Using the componentwise formula:
            ∂ℓ/∂W_{pℓ} = -(1/(2θ_ℓ²)) * ∑_{i,j} H_{ij} * C_{ij} * (Z_{iℓ} - Z_{jℓ}) * (X_{ip} - X_{jp})
        
        where H = (1/τ²)αα^T - Σ^{-1}.
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            
        Returns:
            Gradient matrix (p, D)
        """
        Y = Y.reshape(-1, 1)
        n, p = X.shape
        D = W.shape[1]
        
        Z = X @ W  # (n, D)
        Sigma = self.compute_sigma(Z)
        C = self.compute_covariance(Z)
        
        # Compute α = Σ^{-1}Y
        alpha = np.linalg.solve(Sigma, Y)
        
        # Compute H = (1/τ²)αα^T - Σ^{-1}
        Sigma_inv = np.linalg.inv(Sigma)
        H = (1.0 / self.tau2) * (alpha @ alpha.T) - Sigma_inv
        
        # Initialize gradient
        grad_W = np.zeros((p, D))
        
        # Compute gradient for each element
        for ell in range(D):
            theta_ell_sq = self.lengthscales[ell]**2
            
            # Compute Z differences for dimension ell: (n, n)
            Z_diff = Z[:, ell:ell+1] - Z[:, ell:ell+1].T  # (n, n)
            
            # Compute X differences: (n, n, p)
            X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, p)
            
            # Weighted matrix: H ⊙ C ⊙ Z_diff
            weighted = H * C * Z_diff  # (n, n)
            
            # Sum over i,j for each p
            for p_idx in range(p):
                grad_W[p_idx, ell] = -(1.0 / (2.0 * theta_ell_sq)) * np.sum(
                    weighted * X_diff[:, :, p_idx]
                )
        
        return grad_W
    
    def gradient_log_likelihood_W_vectorized(self, Y: np.ndarray, X: np.ndarray, 
                                            W: np.ndarray) -> np.ndarray:
        """
        Vectorized version of gradient computation (more efficient).
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            
        Returns:
            Gradient matrix (p, D)
        """
        Y = Y.reshape(-1, 1)
        n, p = X.shape
        D = W.shape[1]
        
        Z = X @ W
        Sigma = self.compute_sigma(Z)
        C = self.compute_covariance(Z)
        
        # Compute α and H
        alpha = np.linalg.solve(Sigma, Y)
        Sigma_inv = np.linalg.inv(Sigma)
        H = (1.0 / self.tau2) * (alpha @ alpha.T) - Sigma_inv
        
        # Compute H ⊙ C
        HC = H * C  # (n, n)
        
        # Initialize gradient
        grad_W = np.zeros((p, D))
        
        # Compute X differences for efficiency
        X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, p)
        
        # Vectorized computation per dimension
        for ell in range(D):
            theta_ell_sq = self.lengthscales[ell]**2
            
            # Z differences for dimension ell
            Z_diff = Z[:, ell:ell+1] - Z[:, ell:ell+1].T  # (n, n)
            
            # Weighted matrix for this dimension
            weighted = HC * Z_diff  # (n, n)
            
            # Gradient for dimension ell using proper formula
            for k in range(p):
                grad_W[k, ell] = -(1.0 / (2.0 * theta_ell_sq)) * np.sum(
                    weighted * X_diff[:, :, k]
                )
        
        return grad_W
    
    def gradient_log_likelihood_W_tf(self, Y: tf.Tensor, X: tf.Tensor, 
                                     W: tf.Variable) -> tf.Tensor:
        """
        TensorFlow automatic differentiation version.
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D) as TensorFlow Variable
            
        Returns:
            Gradient tensor (p, D)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        with tf.GradientTape() as tape:
            tape.watch(W)
            
            # Compute Z = XW
            Z = tf.matmul(X, W)
            
            # Compute covariance
            diff = tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0)  # (n, m, D)
            lengthscales_tf = tf.cast(self.lengthscales, dtype=diff.dtype)
            scaled_sq_diff = (diff**2) / (2.0 * lengthscales_tf**2)
            C = tf.exp(-tf.reduce_sum(scaled_sq_diff, axis=2))
            
            # Compute Σ
            n = tf.shape(C)[0]
            tau2_tf = tf.cast(self.tau2, dtype=C.dtype)
            nugget_tf = tf.cast(self.nugget, dtype=C.dtype)
            Sigma = tau2_tf * (C + nugget_tf * tf.eye(n, dtype=C.dtype))
            
            # Compute log-likelihood
            Y_reshaped = tf.reshape(Y, [-1, 1])
            L = tf.linalg.cholesky(Sigma)
            log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
            alpha = tf.linalg.cholesky_solve(L, Y_reshaped)
            quad_form = tf.matmul(Y_reshaped, alpha, transpose_a=True)
            n_float = tf.cast(n, dtype=Y.dtype)
            pi_val = tf.cast(np.pi, dtype=Y.dtype)
            log_lik = -0.5 * (log_det + quad_form[0, 0] + n_float * tf.math.log(2.0 * pi_val))
        
        # Compute gradient
        grad_W = tape.gradient(log_lik, W)
        
        return grad_W


# =============================================================================
# 3. Separable Matérn-3/2 Kernel
# =============================================================================

class SeparableMatern32Kernel(BaseKernel):
    """
    Separable Matérn-3/2 Kernel with dimension-specific lengthscales.
    
    Mathematical form:
        R = ∑_{ℓ=1}^D (z_{iℓ} - z_{jℓ})² / θ_ℓ²
        C_{ij} = (1 + √3*R) * exp(-√3*R)
    """
    
    def __init__(self, lengthscales: np.ndarray, nugget: float = 1e-6, tau2: float = 1.0):
        """
        Initialize separable Matérn-3/2 kernel.
        
        Args:
            lengthscales: Lengthscale for each dimension (D,)
            nugget: Nugget parameter (g)
            tau2: Observation noise variance (τ²)
        """
        super().__init__(lengthscales=lengthscales, nugget=nugget, tau2=tau2)
        self.D = len(lengthscales)
    
    def compute_covariance(self, Z: np.ndarray, Z_prime: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Matérn-3/2 covariance matrix.
        
        Args:
            Z: Input points (n, D)
            Z_prime: Second set of points (m, D). If None, uses Z.
            
        Returns:
            Covariance matrix (n, m)
        """
        if Z_prime is None:
            Z_prime = Z
        
        # Compute scaled differences
        diff = Z[:, np.newaxis, :] - Z_prime[np.newaxis, :, :]  # (n, m, D)
        
        # R = ∑_{ℓ} (z_{iℓ} - z_{jℓ})² / θ_ℓ²
        R = np.sum((diff**2) / (self.lengthscales**2), axis=2)
        
        # Add small epsilon to avoid division by zero
        R = np.maximum(R, 1e-12)
        
        sqrt3R = np.sqrt(3.0 * R)
        
        # C = (1 + √3*R) * exp(-√3*R)
        C = (1.0 + sqrt3R) * np.exp(-sqrt3R)
        
        return C
    
    def gradient_covariance_W_componentwise(self, X: np.ndarray, W: np.ndarray,
                                           Z: Optional[np.ndarray] = None,
                                           C: Optional[np.ndarray] = None,
                                           R: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradient using componentwise formula:
            ∂C/∂W_{kj} = -(6R/θ_j²) * exp(-√3*R) * ∑_i (Z_{ij} - Z'_{ij})(X_{ik} - X'_{ik})
        
        For self-covariance (Z' = Z), this simplifies.
        
        Args:
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            Z: Precomputed Z = XW (n, D)
            C: Precomputed covariance (n, n)
            R: Precomputed R matrix (n, n)
            
        Returns:
            Gradient matrix (p, D)
        """
        if Z is None:
            Z = X @ W
        
        n, p = X.shape
        D = W.shape[1]
        
        # Compute R if not provided
        if R is None:
            diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]
            R = np.sum((diff**2) / (self.lengthscales**2), axis=2)
            R = np.maximum(R, 1e-12)
        
        # Compute factor: -6R * exp(-√3*R)
        sqrt3R = np.sqrt(3.0 * R)
        factor = -6.0 * R * np.exp(-sqrt3R)  # (n, n)
        
        # Initialize gradient
        grad_W = np.zeros((p, D))
        
        # Compute gradient for each dimension
        for j in range(D):
            theta_j_sq = self.lengthscales[j]**2
            
            # Z differences for dimension j
            Z_diff = Z[:, j:j+1] - Z[:, j:j+1].T  # (n, n)
            
            # X differences
            X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, p)
            
            # Weighted matrix
            weighted = (factor / theta_j_sq) * Z_diff  # (n, n)
            
            # Sum over i,j for each k
            for k in range(p):
                grad_W[k, j] = np.sum(weighted * X_diff[:, :, k])
        
        return grad_W
    
    def gradient_log_likelihood_W(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log-likelihood with respect to W.
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            
        Returns:
            Gradient matrix (p, D)
        """
        Y = Y.reshape(-1, 1)
        n, p = X.shape
        D = W.shape[1]
        
        Z = X @ W
        Sigma = self.compute_sigma(Z)
        
        # Compute α and H
        alpha = np.linalg.solve(Sigma, Y)
        Sigma_inv = np.linalg.inv(Sigma)
        H = (1.0 / self.tau2) * (alpha @ alpha.T) - Sigma_inv
        
        # Compute R
        diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]
        R = np.sum((diff**2) / (self.lengthscales**2), axis=2)
        R = np.maximum(R, 1e-12)
        
        # Compute dC/dR = -3R * exp(-√3*R)
        sqrt3R = np.sqrt(3.0 * R)
        dC_dR = -3.0 * R * np.exp(-sqrt3R)  # (n, n)
        
        # Initialize gradient
        grad_W = np.zeros((p, D))
        
        # Compute gradient for each dimension
        for j in range(D):
            theta_j_sq = self.lengthscales[j]**2
            
            # Z differences for dimension j
            Z_diff = Z[:, j:j+1] - Z[:, j:j+1].T  # (n, n)
            
            # dR/dZ_{ij} = 2*(Z_{ij} - Z'_{ij}) / θ_j²
            dR_dZ = 2.0 * Z_diff / theta_j_sq  # (n, n)
            
            # dC/dW contribution: H ⊙ dC/dR ⊙ dR/dZ
            weighted = H * dC_dR * dR_dZ  # (n, n)
            
            # Chain rule through Z = XW
            grad_W[:, j] = np.diag(X.T @ weighted @ X)
        
        return grad_W
    
    def gradient_log_likelihood_W_tf(self, Y: tf.Tensor, X: tf.Tensor, 
                                     W: tf.Variable) -> tf.Tensor:
        """TensorFlow automatic differentiation version."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        with tf.GradientTape() as tape:
            tape.watch(W)
            
            # Compute Z = XW
            Z = tf.matmul(X, W)
            
            # Compute R
            diff = tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0)
            lengthscales_tf = tf.cast(self.lengthscales, dtype=diff.dtype)
            R = tf.reduce_sum((diff**2) / (lengthscales_tf**2), axis=2)
            R = tf.maximum(R, 1e-12)
            
            # Compute covariance
            sqrt3R = tf.sqrt(3.0 * R)
            C = (1.0 + sqrt3R) * tf.exp(-sqrt3R)
            
            # Compute Σ
            n = tf.shape(C)[0]
            tau2_tf = tf.cast(self.tau2, dtype=C.dtype)
            nugget_tf = tf.cast(self.nugget, dtype=C.dtype)
            Sigma = tau2_tf * (C + nugget_tf * tf.eye(n, dtype=C.dtype))
            
            # Compute log-likelihood
            Y_reshaped = tf.reshape(Y, [-1, 1])
            L = tf.linalg.cholesky(Sigma)
            log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
            alpha = tf.linalg.cholesky_solve(L, Y_reshaped)
            quad_form = tf.matmul(Y_reshaped, alpha, transpose_a=True)
            n_float = tf.cast(n, dtype=Y.dtype)
            pi_val = tf.cast(np.pi, dtype=Y.dtype)
            log_lik = -0.5 * (log_det + quad_form[0, 0] + n_float * tf.math.log(2.0 * pi_val))
        
        grad_W = tape.gradient(log_lik, W)
        return grad_W


# =============================================================================
# 4. Isotropic Matérn-3/2 Kernel
# =============================================================================

class IsotropicMatern32Kernel(BaseKernel):
    """
    Isotropic Matérn-3/2 Kernel with single lengthscale.
    
    Mathematical form:
        r = ||z_i - z_j|| / θ
        C_{ij} = (1 + √3*r) * exp(-√3*r)
    """
    
    def __init__(self, lengthscale: float, nugget: float = 1e-6, tau2: float = 1.0):
        """
        Initialize isotropic Matérn-3/2 kernel.
        
        Args:
            lengthscale: Single lengthscale parameter (θ)
            nugget: Nugget parameter (g)
            tau2: Observation noise variance (τ²)
        """
        super().__init__(lengthscales=np.array([lengthscale]), nugget=nugget, tau2=tau2)
        self.theta = lengthscale
    
    def compute_covariance(self, Z: np.ndarray, Z_prime: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute isotropic Matérn-3/2 covariance matrix.
        
        Args:
            Z: Input points (n, D)
            Z_prime: Second set of points (m, D). If None, uses Z.
            
        Returns:
            Covariance matrix (n, m)
        """
        if Z_prime is None:
            Z_prime = Z
        
        # Compute Euclidean distances
        diff = Z[:, np.newaxis, :] - Z_prime[np.newaxis, :, :]  # (n, m, D)
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)  # (n, m)
        
        # Scaled distance: r = ||z_i - z_j|| / θ
        r = dist / self.theta
        sqrt3r = np.sqrt(3.0) * r
        
        # C = (1 + √3*r) * exp(-√3*r)
        C = (1.0 + sqrt3r) * np.exp(-sqrt3r)
        
        return C
    
    def gradient_covariance_W(self, X: np.ndarray, W: np.ndarray,
                             Z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradient of covariance with respect to W.
        
        For isotropic Matérn-3/2:
            ∂C/∂W = -(3/θ²) * C * r * exp(-√3*r) * (ΔX)^T * ΔZ
        
        where r = ||z_i - z_j|| / θ.
        
        Args:
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            Z: Precomputed Z = XW (n, D)
            
        Returns:
            Gradient matrix (p, D)
        """
        if Z is None:
            Z = X @ W
        
        n, p = X.shape
        D = W.shape[1]
        
        # Compute distances
        Z_diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]  # (n, n, D)
        dist = np.sqrt(np.sum(Z_diff**2, axis=2) + 1e-12)  # (n, n)
        
        # Scaled distance
        r = dist / self.theta
        sqrt3r = np.sqrt(3.0) * r
        
        # Compute covariance
        C = (1.0 + sqrt3r) * np.exp(-sqrt3r)
        
        # Compute factor: -(3/θ²) * r * exp(-√3*r)
        factor = -(3.0 / (self.theta**2)) * r * np.exp(-sqrt3r)  # (n, n)
        
        # Weighted covariance
        weighted = C * factor  # (n, n)
        
        # X differences
        X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, p)
        
        # Gradient computation
        grad_W = np.zeros((p, D))
        for d in range(D):
            for k in range(p):
                # Sum over pairs (i,j)
                grad_W[k, d] = np.sum(weighted * X_diff[:, :, k] * Z_diff[:, :, d])
        
        return grad_W
    
    def gradient_log_likelihood_W(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log-likelihood with respect to W.
        
        Args:
            Y: Response vector (n,)
            X: Input matrix (n, p)
            W: Projection matrix (p, D)
            
        Returns:
            Gradient matrix (p, D)
        """
        Y = Y.reshape(-1, 1)
        n, p = X.shape
        D = W.shape[1]
        
        Z = X @ W
        Sigma = self.compute_sigma(Z)
        
        # Compute α and H
        alpha = np.linalg.solve(Sigma, Y)
        Sigma_inv = np.linalg.inv(Sigma)
        H = (1.0 / self.tau2) * (alpha @ alpha.T) - Sigma_inv
        
        # Compute distances
        Z_diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]  # (n, n, D)
        dist = np.sqrt(np.sum(Z_diff**2, axis=2) + 1e-12)  # (n, n)
        
        # Scaled distance
        r = dist / self.theta
        sqrt3r = np.sqrt(3.0) * r
        
        # dC/dr = -3r * exp(-√3*r)
        dC_dr = -3.0 * r * np.exp(-sqrt3r)  # (n, n)
        
        # dr/dZ_{id} = (Z_{id} - Z_{jd}) / (θ * ||z_i - z_j||)
        # Avoid division by zero
        safe_dist = np.maximum(dist, 1e-12)
        dr_dZ = Z_diff / (self.theta * safe_dist[:, :, np.newaxis])  # (n, n, D)
        
        # X differences
        X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, p)
        
        # Initialize gradient
        grad_W = np.zeros((p, D))
        
        # Compute gradient
        for d in range(D):
            # H ⊙ dC/dr ⊙ dr/dZ
            weighted = H * dC_dr * dr_dZ[:, :, d]  # (n, n)
            
            for k in range(p):
                grad_W[k, d] = np.sum(weighted * X_diff[:, :, k])
        
        return grad_W
    
    def gradient_log_likelihood_W_tf(self, Y: tf.Tensor, X: tf.Tensor, 
                                     W: tf.Variable) -> tf.Tensor:
        """TensorFlow automatic differentiation version."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        with tf.GradientTape() as tape:
            tape.watch(W)
            
            # Compute Z = XW
            Z = tf.matmul(X, W)
            
            # Compute distances
            diff = tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0)
            dist = tf.sqrt(tf.reduce_sum(diff**2, axis=2) + 1e-12)
            
            # Scaled distance
            theta_tf = tf.cast(self.theta, dtype=dist.dtype)
            r = dist / theta_tf
            sqrt3r = tf.sqrt(tf.cast(3.0, dtype=r.dtype)) * r
            
            # Compute covariance
            C = (1.0 + sqrt3r) * tf.exp(-sqrt3r)
            
            # Compute Σ
            n = tf.shape(C)[0]
            tau2_tf = tf.cast(self.tau2, dtype=C.dtype)
            nugget_tf = tf.cast(self.nugget, dtype=C.dtype)
            Sigma = tau2_tf * (C + nugget_tf * tf.eye(n, dtype=C.dtype))
            
            # Compute log-likelihood
            Y_reshaped = tf.reshape(Y, [-1, 1])
            L = tf.linalg.cholesky(Sigma)
            log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
            alpha = tf.linalg.cholesky_solve(L, Y_reshaped)
            quad_form = tf.matmul(Y_reshaped, alpha, transpose_a=True)
            n_float = tf.cast(n, dtype=Y.dtype)
            pi_val = tf.cast(np.pi, dtype=Y.dtype)
            log_lik = -0.5 * (log_det + quad_form[0, 0] + n_float * tf.math.log(2.0 * pi_val))
        
        grad_W = tape.gradient(log_lik, W)
        return grad_W


# =============================================================================
# Utility Functions
# =============================================================================

def compare_gradients(kernel: BaseKernel, Y: np.ndarray, X: np.ndarray, 
                     W: np.ndarray, epsilon: float = 1e-5) -> Dict[str, np.ndarray]:
    """
    Compare analytical gradient with numerical gradient (finite differences).
    
    Args:
        kernel: Kernel object
        Y: Response vector (n,)
        X: Input matrix (n, p)
        W: Projection matrix (p, D)
        epsilon: Finite difference step size
        
    Returns:
        Dictionary with analytical, numerical, and difference
    """
    # Analytical gradient
    grad_analytical = kernel.gradient_log_likelihood_W(Y, X, W)
    
    # Numerical gradient using finite differences
    p, D = W.shape
    grad_numerical = np.zeros((p, D))
    
    for i in range(p):
        for j in range(D):
            W_plus = W.copy()
            W_minus = W.copy()
            
            W_plus[i, j] += epsilon
            W_minus[i, j] -= epsilon
            
            Z_plus = X @ W_plus
            Z_minus = X @ W_minus
            
            lik_plus = kernel.log_likelihood(Y, Z_plus)
            lik_minus = kernel.log_likelihood(Y, Z_minus)
            
            grad_numerical[i, j] = (lik_plus - lik_minus) / (2.0 * epsilon)
    
    # Compute difference
    diff = np.abs(grad_analytical - grad_numerical)
    relative_diff = diff / (np.abs(grad_numerical) + 1e-10)
    
    return {
        'analytical': grad_analytical,
        'numerical': grad_numerical,
        'absolute_diff': diff,
        'relative_diff': relative_diff,
        'max_abs_diff': np.max(diff),
        'max_rel_diff': np.max(relative_diff)
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Covariance Kernel Functions and Gradients - Examples")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    n, p, D = 50, 10, 2
    
    X = np.random.randn(n, p)
    W_true = np.random.randn(p, D)
    W_true = W_true / np.linalg.norm(W_true, axis=0)
    
    Z = X @ W_true
    
    # Generate Y from GP
    theta = 1.0
    C = np.exp(-0.5 * np.sum((Z[:, np.newaxis, :] - Z[np.newaxis, :, :])**2, axis=2) / theta)
    Y = np.random.multivariate_normal(np.zeros(n), C + 0.01 * np.eye(n))
    
    print(f"\nData: n={n}, p={p}, D={D}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, W shape: {W_true.shape}")
    
    # Test each kernel
    kernels = [
        ("Isotropic SE", IsotropicSquaredExponentialKernel(lengthscale=1.0)),
        ("Separable SE", SeparableSquaredExponentialKernel(lengthscales=np.ones(D))),
        ("Separable Matérn-3/2", SeparableMatern32Kernel(lengthscales=np.ones(D))),
        ("Isotropic Matérn-3/2", IsotropicMatern32Kernel(lengthscale=1.0))
    ]
    
    for name, kernel in kernels:
        print(f"\n{'-'*70}")
        print(f"{name} Kernel")
        print(f"{'-'*70}")
        
        # Compute covariance
        Z_test = X @ W_true
        C = kernel.compute_covariance(Z_test)
        print(f"Covariance matrix shape: {C.shape}")
        print(f"Covariance range: [{C.min():.4f}, {C.max():.4f}]")
        
        # Compute log-likelihood
        log_lik = kernel.log_likelihood(Y, Z_test)
        print(f"Log-likelihood: {log_lik:.4f}")
        
        # Compute gradient
        grad = kernel.gradient_log_likelihood_W(Y, X, W_true)
        print(f"Gradient shape: {grad.shape}")
        print(f"Gradient norm: {np.linalg.norm(grad):.4f}")
        print(f"Gradient range: [{grad.min():.6f}, {grad.max():.6f}]")
        
        # Compare with numerical gradient (small subset for speed)
        if n <= 20:  # Only for small problems
            print("\nGradient verification (finite differences):")
            comparison = compare_gradients(kernel, Y, X, W_true, epsilon=1e-5)
            print(f"  Max absolute difference: {comparison['max_abs_diff']:.2e}")
            print(f"  Max relative difference: {comparison['max_rel_diff']:.2e}")
            
            if comparison['max_abs_diff'] < 1e-3:
                print("  ✓ Gradient verification PASSED")
            else:
                print("  ✗ Gradient verification FAILED (check implementation)")
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
