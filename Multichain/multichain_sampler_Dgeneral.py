"""
Multi-Chain Gibbs Sampler with Predictions and Diagnostics (D>1)

This module implements multi-chain MCMC sampling for D>1 (e.g., D=2, 3, 5) with:
    - Multiple chains for convergence diagnostics
    - Posterior predictions with uncertainty quantification
    - Performance metrics (RMSPE, NSME, CRPS, Score, BIC, MLPPD, CP, ALCI)
    - Comprehensive diagnostics (Gelman-Rubin, Heidelberg-Welch)
    - Visualization (trace plots, density plots, autocorrelation, etc.)
    
Key differences from D=1:
    - θ is a VECTOR (D,) with one lengthscale per dimension
    - W is a (p × D) matrix on Stiefel manifold
    - Plots for each dimension of θ
"""

import numpy as np
import pandas as pd
import time
import warnings
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import norm

# Add parent directories to path for imports
base_dir = Path(__file__).parent.parent
for folder in ["Gibbs Sampling", "Parameter Sampler", "BDR Metrics and Plot"]:
    folder_path = str(base_dir / folder)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)

from gibbs_sampler_layers_Dgeneral import GibbsSampler1Layer, GibbsSampler2Layer, GibbsSampler3Layer  # type: ignore[import]
from parameter_sampler_Dgeneral import covar_sep  # type: ignore[import]
from BDR_metrics import (  # type: ignore[import]
    compute_RMSPE, compute_NSME, compute_CRPS, compute_score, 
    compute_BIC, compute_MLPPD, compute_CP, compute_ALCI,
    compute_all_metrics_summary, compute_iteration_metrics,
    compute_multichain_parameter_diagnostics
)
from BDR_plot import (  # type: ignore[import]
    plot_trace, plot_density, plot_histogram, plot_autocorrelation,
    plot_W_trace_multichain, plot_actual_vs_predicted, 
    plot_convergence_diagnostics, plot_metrics_boxplot, plot_metrics_comparison_table
)


# =============================================================================
# Performance Metrics (same as D=1 but adapted for D>1)
# =============================================================================

def compute_RMSPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Predictive Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def compute_NSME(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Model Efficiency."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot


def compute_CRPS(y_true: np.ndarray, y_pred_mean: np.ndarray, 
                y_pred_std: np.ndarray) -> float:
    """Continuous Ranked Probability Score."""
    z = (y_true - y_pred_mean) / y_pred_std
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    crps_values = y_pred_std * (1/np.sqrt(np.pi) - 2*phi_z - z*(2*Phi_z - 1))
    return np.mean(crps_values)


def compute_score(y_true: np.ndarray, y_pred_mean: np.ndarray, 
                 Sigma_pred: np.ndarray) -> float:
    """Predictive Log-Likelihood Score."""
    residual = y_true - y_pred_mean
    
    sign, logdet = np.linalg.slogdet(Sigma_pred)
    if sign <= 0:
        return -np.inf
    
    try:
        Sigma_inv = np.linalg.inv(Sigma_pred)
        quad_form = residual.T @ Sigma_inv @ residual
    except:
        return -np.inf
    
    score = -logdet - quad_form
    return score


def compute_BIC(log_likelihood: float, n_params: int, n_train: int) -> float:
    """Bayesian Information Criterion."""
    return log_likelihood - 0.5 * n_params * np.log(n_train)


def compute_MLPPD(y_true: np.ndarray, y_pred_mean: np.ndarray, 
                 y_pred_var: np.ndarray) -> float:
    """Mean Log Pointwise Predictive Density."""
    log_lik_points = -0.5 * np.log(2 * np.pi * y_pred_var) - \
                     (y_true - y_pred_mean)**2 / (2 * y_pred_var)
    return np.mean(log_lik_points)


# =============================================================================
# Prediction Functions for D>1
# =============================================================================

def _as_2d(x: np.ndarray) -> np.ndarray:
    """Ensure input is 2D with shape (n, d)."""
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _as_scalar(x, default: float = 1.0) -> float:
    """Convert scalar/array-like to float safely."""
    if x is None:
        return float(default)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _as_vector(x, D: int, default: float = 1.0) -> np.ndarray:
    """Convert parameter to vector of length D."""
    if x is None:
        return np.full(D, float(default))
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(D, float(default))
    if arr.size == 1:
        return np.full(D, float(arr[0]))
    if arr.size < D:
        return np.concatenate([arr, np.full(D - arr.size, float(arr[0]))])
    return arr[:D]


def _stabilize_covariance(Sigma: np.ndarray, min_jitter: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    """Symmetrize and add adaptive jitter until covariance is numerically positive-definite."""
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)
    I = np.eye(Sigma.shape[0], dtype=float)

    jitter = min_jitter
    for _ in range(max_tries):
        Sigma_try = Sigma + jitter * I
        sign, _ = np.linalg.slogdet(Sigma_try)
        if sign > 0:
            return Sigma_try
        jitter *= 10.0

    return Sigma + jitter * I


def _gp_conditional(
    X_star: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    theta,
    g: float,
    tau2: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    GP conditioning step:
      y_star | y ~ N(K_*n K_nn^{-1} y, tau2 * (K_** - K_*n K_nn^{-1} K_n*)).
    """
    X_train = _as_2d(X_train)
    X_star = _as_2d(X_star)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    K_nn = covar_sep(X_train, theta, g)
    K_star_n = covar_sep_cross(X_star, X_train, theta)
    K_star_star = covar_sep(X_star, theta, g)

    jitter = 1e-8
    K_nn_j = K_nn + jitter * np.eye(K_nn.shape[0])

    try:
        alpha = np.linalg.solve(K_nn_j, y_train)
        pred_mean = K_star_n @ alpha
        v = np.linalg.solve(K_nn_j, K_star_n.T)
        pred_cov = tau2 * (K_star_star - K_star_n @ v)
        pred_cov = _stabilize_covariance(pred_cov)
    except np.linalg.LinAlgError:
        pred_mean = np.zeros(X_star.shape[0], dtype=float)
        pred_cov = _stabilize_covariance(tau2 * np.eye(X_star.shape[0], dtype=float))

    return pred_mean, pred_cov


def predict_gp_single_sample(
    X_new: np.ndarray,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    W: np.ndarray,
    theta_D_y: np.ndarray,
    g_y: float,
    tau2_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """1-layer posterior prediction: Z -> Y."""
    Z_train = X_train @ W
    Z_new = X_new @ W
    D = Z_train.shape[1]
    theta_y_vec = _as_vector(theta_D_y, D, default=1.0)
    return _gp_conditional(
        Z_new, Z_train, Y_train,
        theta=theta_y_vec,
        g=_as_scalar(g_y, default=0.0),
        tau2=_as_scalar(tau2_y)
    )


def predict_dgp_2layer_single_sample(
    X_new: np.ndarray,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    W: np.ndarray,
    Q_train: np.ndarray,
    theta_y: np.ndarray,
    g_y: float,
    tau2_y: float,
    theta_q: np.ndarray,
    g_q: np.ndarray,
    tau2_q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """2-layer posterior prediction: Z -> Q -> Y (stacking latent means)."""
    Z_train = _as_2d(X_train @ W)
    Z_new = _as_2d(X_new @ W)
    Q_train = _as_2d(Q_train)
    D = Q_train.shape[1]

    theta_q_vec = _as_vector(theta_q, D, default=1.0)
    g_q_vec = _as_vector(g_q, D, default=0.0)
    tau2_q_vec = _as_vector(tau2_q, D, default=1.0)

    Q_new = np.zeros((Z_new.shape[0], D), dtype=float)
    for d in range(D):
        mu_q_d, _ = _gp_conditional(
            Z_new[:, [d]], Z_train[:, [d]], Q_train[:, d],
            theta=theta_q_vec[d],
            g=g_q_vec[d],
            tau2=tau2_q_vec[d]
        )
        Q_new[:, d] = mu_q_d

    theta_y_vec = _as_vector(theta_y, D, default=1.0)
    mu_y, cov_y = _gp_conditional(
        Q_new, Q_train, Y_train,
        theta=theta_y_vec,
        g=_as_scalar(g_y, default=0.0),
        tau2=_as_scalar(tau2_y)
    )
    return mu_y, cov_y


def predict_dgp_3layer_single_sample(
    X_new: np.ndarray,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    W: np.ndarray,
    R_train: np.ndarray,
    Q_train: np.ndarray,
    theta_y: np.ndarray,
    g_y: float,
    tau2_y: float,
    theta_q: np.ndarray,
    g_q: np.ndarray,
    tau2_q: np.ndarray,
    theta_r: np.ndarray,
    g_r: np.ndarray,
    tau2_r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """3-layer posterior prediction: Z -> R -> Q -> Y (stacking latent means)."""
    Z_train = _as_2d(X_train @ W)
    Z_new = _as_2d(X_new @ W)
    R_train = _as_2d(R_train)
    Q_train = _as_2d(Q_train)
    D = Q_train.shape[1]

    theta_r_vec = _as_vector(theta_r, D, default=1.0)
    g_r_vec = _as_vector(g_r, D, default=0.0)
    tau2_r_vec = _as_vector(tau2_r, D, default=1.0)
    theta_q_vec = _as_vector(theta_q, D, default=1.0)
    g_q_vec = _as_vector(g_q, D, default=0.0)
    tau2_q_vec = _as_vector(tau2_q, D, default=1.0)

    R_new = np.zeros((Z_new.shape[0], D), dtype=float)
    for d in range(D):
        mu_r_d, _ = _gp_conditional(
            Z_new[:, [d]], Z_train[:, [d]], R_train[:, d],
            theta=theta_r_vec[d],
            g=g_r_vec[d],
            tau2=tau2_r_vec[d]
        )
        R_new[:, d] = mu_r_d

    Q_new = np.zeros((Z_new.shape[0], D), dtype=float)
    for d in range(D):
        mu_q_d, _ = _gp_conditional(
            R_new[:, [d]], R_train[:, [d]], Q_train[:, d],
            theta=theta_q_vec[d],
            g=g_q_vec[d],
            tau2=tau2_q_vec[d]
        )
        Q_new[:, d] = mu_q_d

    theta_y_vec = _as_vector(theta_y, D, default=1.0)
    mu_y, cov_y = _gp_conditional(
        Q_new, Q_train, Y_train,
        theta=theta_y_vec,
        g=_as_scalar(g_y, default=0.0),
        tau2=_as_scalar(tau2_y)
    )
    return mu_y, cov_y


def covar_sep_cross(Z1: np.ndarray, Z2: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute cross-covariance between two sets of points for D>1."""
    n1 = Z1.shape[0]
    n2 = Z2.shape[0]
    D = Z1.shape[1]
    
    theta = np.atleast_1d(theta)
    
    C_matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            sum_term = sum((Z1[i, k] - Z2[j, k])**2 / (2 * theta[k]**2) for k in range(D))
            C_matrix[i, j] = np.exp(-sum_term)
    
    return C_matrix


def _sample_array(samples: Dict, *keys: str) -> np.ndarray:
    """Return the first available sample array among candidate keys."""
    for key in keys:
        if key in samples:
            return samples[key]
    raise KeyError(f"None of the keys {keys} found in samples.")


def _diagnostic_chains_for_keys(
    chains_samples: List[Dict],
    keys: Tuple[str, ...],
    max_components: int = 12,
) -> List[np.ndarray]:
    """Flatten parameter chains for plotting, limiting high-dimensional arrays."""
    chains = []
    for samples in chains_samples:
        arr = np.asarray(_sample_array(samples, *keys), dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)
            arr = arr[:, :max_components]
        chains.append(arr)
    return chains


def _projection_chains_for_W(W_chains: List[np.ndarray], max_components: int = 12) -> List[np.ndarray]:
    """Flatten W W^T projection chains for plotting."""
    chains = []
    for chain_W in W_chains:
        chain_WWT = np.einsum('nkd,nld->nkl', np.asarray(chain_W, dtype=float), np.asarray(chain_W, dtype=float))
        chains.append(chain_WWT.reshape(chain_WWT.shape[0], -1)[:, :max_components])
    return chains


# =============================================================================
# Bayesian Metrics with Quantiles (D>1)
# =============================================================================

def compute_loglikelihood_layer_Dgeneral(Z: np.ndarray, Y_or_Q: np.ndarray, theta: np.ndarray,
                                         g: np.ndarray, tau2, D: int) -> float:
    """
    Compute log-likelihood for a single layer (D>1).
    
    CRITICAL: For D>1, compute per dimension and SUM!
    
    Args:
        Z: Inputs (n, D)
        Y_or_Q: Outputs (n, D) or (n,) if Y layer
        theta: Lengthscale vector (D,)
        g: Nugget vector (D,) or scalar
        tau2: Noise variance (scalar or vector)
        D: Dimension
        
    Returns:
        Log-likelihood value (sum over dimensions)
    """
    n = Z.shape[0]
    
    # If Y (response), just one dimension
    if Y_or_Q.ndim == 1 or Y_or_Q.shape[1] == 1:
        # Single output - use first dimension's parameters
        Y_or_Q = Y_or_Q.reshape(-1, 1)
        Z_use = Z if Z.shape[1] == Y_or_Q.shape[1] else Z
        # Handle theta: for separable kernels with D>1, need full vector
        # For isotropic kernels, use scalar
        if np.isscalar(theta):
            theta_use = theta
        elif hasattr(theta, '__len__'):
            # For separable kernels, need full theta vector of length D
            if len(theta) == D:
                theta_use = theta
            else:
                # If theta is shorter, pad or use first element
                theta_use = theta[0] if len(theta) > 0 else 1.0
        else:
            theta_use = 1.0  # Default fallback
        # For separable kernels, covar_sep expects theta to be vector of length D
        # Ensure theta_use is a vector of length D
        if np.isscalar(theta_use) or (hasattr(theta_use, '__len__') and len(theta_use) == 1):
            # If scalar, convert to vector of length D
            theta_use = np.full(D, theta_use if np.isscalar(theta_use) else theta_use[0])
        elif hasattr(theta_use, '__len__') and len(theta_use) != D:
            # If length doesn't match D, pad or truncate
            if len(theta_use) < D:
                theta_use = np.concatenate([theta_use, np.full(D - len(theta_use), theta_use[0])])
            else:
                theta_use = theta_use[:D]
        K = covar_sep(Z_use, theta_use, 
                     g if np.isscalar(g) else g[0])
        
        try:
            Ki = np.linalg.pinv(K)
            sign, logdet_K = np.linalg.slogdet(K)
            if sign <= 0:
                return -np.inf
            
            tau2_use = _as_scalar(tau2, default=1.0)
            loglik = -0.5 * (
                logdet_K +
                (Y_or_Q.T @ Ki @ Y_or_Q).item() / tau2_use +
                n * np.log(2 * np.pi * tau2_use)
            )
            return loglik
        except:
            return -np.inf
    
    # For Q or R with D>1: LOOP over dimensions and SUM
    loglikelihood_total = 0.0
    
    for k in range(D):
        # Handle theta: ensure it's a vector of length D, use first element if scalar
        if np.isscalar(theta) or (hasattr(theta, '__len__') and len(theta) == 1):
            theta_k = theta if np.isscalar(theta) else theta[0]
        elif hasattr(theta, '__len__') and len(theta) > k:
            theta_k = theta[k]
        else:
            theta_k = theta[0] if hasattr(theta, '__len__') and len(theta) > 0 else 1.0
        
        # Handle g: ensure it's a scalar or vector
        if np.isscalar(g) or (hasattr(g, '__len__') and len(g) == 1):
            g_k = g if np.isscalar(g) else g[0]
        elif hasattr(g, '__len__') and len(g) > k:
            g_k = g[k]
        else:
            g_k = g[0] if hasattr(g, '__len__') and len(g) > 0 else 0.01
        
        # Handle tau2: scalar or per-dimension vector
        if np.isscalar(tau2) or (hasattr(tau2, '__len__') and len(np.atleast_1d(tau2)) == 1):
            tau2_k = _as_scalar(tau2, default=1.0)
        else:
            tau2_arr = np.asarray(tau2, dtype=float).reshape(-1)
            tau2_k = float(tau2_arr[k]) if k < tau2_arr.size else float(tau2_arr[0])

        # Use scalar covariance for this dimension
        Z_k = Z[:, k:k+1]
        Q_k = Y_or_Q[:, k:k+1]
        
        K = covar_sep(Z_k, theta_k, g_k)
        
        try:
            Ki = np.linalg.pinv(K)
            sign, logdet_K = np.linalg.slogdet(K)
            
            if sign > 0:
                loglik_k = -0.5 * (
                    logdet_K +
                    (Q_k.T @ Ki @ Q_k).item() / tau2_k +
                    n * np.log(2 * np.pi * tau2_k)
                )
                loglikelihood_total += loglik_k
        except:
            pass
    
    return loglikelihood_total


def Bayesian_Metrics_with_quantiles(samples: Dict, X_train: np.ndarray, Y_train: np.ndarray,
                                   X_test: np.ndarray, Y_test: np.ndarray,
                                   layer: int = 1,
                                   quantiles: List[float] = [0.025, 0.975]) -> Dict:
    """
    Compute Bayesian metrics with CORRECT BIC for multi-layer models (D>1).
    
    CRITICAL: For D>1, log-likelihoods computed PER DIMENSION and SUMMED!
    
    Args:
        samples: Dictionary of MCMC samples
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        X_test: Test inputs (n_test, p)
        Y_test: Test responses (n_test,)
        layer: Number of layers (1, 2, or 3)
        quantiles: Quantiles for credible intervals
        
    Returns:
        Dictionary with metrics and predictions
    """
    tau2_samples = _sample_array(samples, 'tau2_y', 'tau2')
    n_samples = len(tau2_samples)
    n_test = len(Y_test)
    n_train = len(Y_train)
    p, D = samples['W'][0].shape
    
    # Storage
    y_pred_samples = np.zeros((n_test, n_samples))
    y_pred_var_samples = np.zeros((n_test, n_samples))
    y_pred_cov_samples = np.zeros((n_samples, n_test, n_test))
    BIC_samples = np.zeros(n_samples)
    
    # Compute for each MCMC sample
    for i in range(n_samples):
        W_i = samples['W'][i]
        tau2_y_i = _as_scalar(tau2_samples[i], default=1.0)
        
        if layer == 1:
            # 1-Layer
            g_y_i = _as_scalar(_sample_array(samples, 'g_y', 'g')[i], default=0.0)
            theta_y_i = _sample_array(samples, 'theta_D_y', 'theta_y', 'theta_D')[i]
            theta_y_vec = _as_vector(theta_y_i, D, default=1.0)

            pred_mean, pred_cov = predict_gp_single_sample(
                X_test, X_train, Y_train, W_i, theta_y_vec, g_y_i, tau2_y_i
            )
            
            # BIC: Y layer only
            Z_train = _as_2d(X_train @ W_i)
            loglik_y = compute_loglikelihood_layer_Dgeneral(
                Z_train, Y_train, theta_y_vec, g_y_i, tau2_y_i, D
            )
            
            # Parameter count: p*D + tau2 + theta(D) + g + extra
            k = p * D + 3  # Simplified for D>1
            BIC_samples[i] = loglik_y - 0.5 * k * np.log(n_train)
            
        elif layer == 2:
            # 2-Layer: SUM loglik_y + loglik_q
            Q_i = _as_2d(samples['Q'][i])  # (n, D)
            theta_y_i = _sample_array(samples, 'theta_y', 'theta_D_y')[i]
            g_y_i = _as_scalar(_sample_array(samples, 'g_y', 'g')[i], default=0.0)
            theta_q_i = _sample_array(samples, 'theta_q', 'theta_D_q')[i]
            g_q_i = _sample_array(samples, 'g_q')[i] if 'g_q' in samples else np.zeros(D)
            tau2_q_i = _sample_array(samples, 'tau2_q')[i] if 'tau2_q' in samples else np.ones(D)

            theta_y_vec = _as_vector(theta_y_i, D, default=1.0)
            theta_q_vec = _as_vector(theta_q_i, D, default=1.0)
            g_q_vec = _as_vector(g_q_i, D, default=0.0)
            tau2_q_vec = _as_vector(tau2_q_i, D, default=1.0)

            pred_mean, pred_cov = predict_dgp_2layer_single_sample(
                X_test, X_train, Y_train,
                W=W_i, Q_train=Q_i,
                theta_y=theta_y_vec, g_y=g_y_i, tau2_y=tau2_y_i,
                theta_q=theta_q_vec, g_q=g_q_vec, tau2_q=tau2_q_vec
            )
            
            Z_train = _as_2d(X_train @ W_i)
            
            # Q layer: Q | Z (D dimensions, loop and sum)
            loglik_q = compute_loglikelihood_layer_Dgeneral(
                Z_train, Q_i, theta_q_vec, g_q_vec, tau2_q_vec, D
            )
            
            # Y layer: Y | Q
            loglik_y = compute_loglikelihood_layer_Dgeneral(
                Q_i, Y_train, theta_y_vec, g_y_i, tau2_y_i, D
            )
            
            # Parameter count: varies with D
            k = 42 if D > 1 else (p * D + 1 + 3)  # From notebooks
            BIC_samples[i] = loglik_y + loglik_q - 0.5 * k * np.log(n_train)
            
        else:  # layer == 3
            # 3-Layer: SUM loglik_y + loglik_q + loglik_r
            R_i = _as_2d(samples['R'][i] if 'R' in samples else samples['Q'][i])
            Q_i = _as_2d(samples['Q'][i])
            theta_y_i = _sample_array(samples, 'theta_y', 'theta_D_y')[i]
            g_y_i = _as_scalar(_sample_array(samples, 'g_y', 'g')[i], default=0.0)
            theta_q_i = _sample_array(samples, 'theta_q', 'theta_D_q')[i]
            g_q_i = _sample_array(samples, 'g_q')[i] if 'g_q' in samples else np.zeros(D)
            theta_r_i = _sample_array(samples, 'theta_r', 'theta_D_r')[i] if ('theta_r' in samples or 'theta_D_r' in samples) else theta_q_i
            g_r_i = _sample_array(samples, 'g_r')[i] if 'g_r' in samples else g_q_i
            tau2_q_i = _sample_array(samples, 'tau2_q')[i] if 'tau2_q' in samples else np.ones(D)
            tau2_r_i = _sample_array(samples, 'tau2_r')[i] if 'tau2_r' in samples else np.ones(D)

            theta_y_vec = _as_vector(theta_y_i, D, default=1.0)
            theta_q_vec = _as_vector(theta_q_i, D, default=1.0)
            g_q_vec = _as_vector(g_q_i, D, default=0.0)
            theta_r_vec = _as_vector(theta_r_i, D, default=1.0)
            g_r_vec = _as_vector(g_r_i, D, default=0.0)
            tau2_q_vec = _as_vector(tau2_q_i, D, default=1.0)
            tau2_r_vec = _as_vector(tau2_r_i, D, default=1.0)

            pred_mean, pred_cov = predict_dgp_3layer_single_sample(
                X_test, X_train, Y_train,
                W=W_i, R_train=R_i, Q_train=Q_i,
                theta_y=theta_y_vec, g_y=g_y_i, tau2_y=tau2_y_i,
                theta_q=theta_q_vec, g_q=g_q_vec, tau2_q=tau2_q_vec,
                theta_r=theta_r_vec, g_r=g_r_vec, tau2_r=tau2_r_vec
            )
            
            Z_train = _as_2d(X_train @ W_i)
            
            # R layer: R | Z
            loglik_r = compute_loglikelihood_layer_Dgeneral(
                Z_train, R_i, theta_r_vec, g_r_vec, tau2_r_vec, D
            )
            
            # Q layer: Q | R
            loglik_q = compute_loglikelihood_layer_Dgeneral(
                R_i, Q_i, theta_q_vec, g_q_vec, tau2_q_vec, D
            )
            
            # Y layer: Y | Q
            loglik_y = compute_loglikelihood_layer_Dgeneral(
                Q_i, Y_train, theta_y_vec, g_y_i, tau2_y_i, D
            )
            
            # Parameter count
            k = 82 if D > 2 else 42 if D > 1 else 6  # From notebooks
            BIC_samples[i] = loglik_y + loglik_q + loglik_r - 0.5 * k * np.log(n_train)
        
        pred_cov = _stabilize_covariance(pred_cov)
        pred_var = np.clip(np.diag(pred_cov), 1e-12, None)
        y_pred_samples[:, i] = pred_mean
        y_pred_var_samples[:, i] = pred_var
        y_pred_cov_samples[i] = pred_cov

    # Posterior predictive moments:
    #   mu_bar = mean_c(mu_c)
    #   Sigma_bar = mean_c(Sigma_c) + sample_cov_c(mu_c)
    y_pred_mean = np.mean(y_pred_samples, axis=1)
    mean_cov = np.mean(y_pred_cov_samples, axis=0)
    if n_samples > 1:
        centered = y_pred_samples - y_pred_mean[:, None]
        between_cov = (centered @ centered.T) / (n_samples - 1)
    else:
        between_cov = np.zeros((n_test, n_test))
    Sigma_pred = _stabilize_covariance(mean_cov + between_cov)
    y_pred_var = np.clip(np.diag(Sigma_pred), 1e-12, None)
    y_pred_std = np.sqrt(y_pred_var)

    # Posterior predictive quantiles from Gaussian approximation N(mu_bar, Sigma_bar)
    z_vals = norm.ppf(np.asarray(quantiles, dtype=float))
    y_pred_quantiles = y_pred_mean[:, None] + y_pred_std[:, None] * z_vals[None, :]
    lower_bound = y_pred_quantiles[:, 0]
    upper_bound = y_pred_quantiles[:, -1]
    
    # Compute metrics
    rmspe = compute_RMSPE(Y_test, y_pred_mean)
    nsme = compute_NSME(Y_test, y_pred_mean)
    crps = compute_CRPS(Y_test, y_pred_mean, y_pred_std)
    
    score = compute_score(Y_test, y_pred_mean, Sigma_pred)
    
    # BIC: Use mean of BIC_samples
    bic = np.mean(BIC_samples)
    
    # MLPPD
    mlppd = compute_MLPPD(Y_test, y_pred_mean, y_pred_var)

    # Predictive interval metrics
    cp = compute_CP(Y_test, lower_bound, upper_bound)
    alci = compute_ALCI(lower_bound, upper_bound)
    
    # Per-iteration metrics (all metrics for every posterior sample)
    iteration_metrics = compute_iteration_metrics(
        y_true=Y_test,
        y_pred_samples=y_pred_samples,
        y_pred_var_samples=y_pred_var_samples,
        sample_axis=1,
        bic_samples=BIC_samples
    )
    
    return {
        'y_pred_samples': y_pred_samples,
        'y_pred_mean': y_pred_mean,
        'y_pred_std': y_pred_std,
        'y_pred_var': y_pred_var,
        'Sigma_pred': Sigma_pred,
        'y_pred_quantiles': y_pred_quantiles,
        'rmspe': rmspe,
        'nsme': nsme,
        'crps': crps,
        'score': score,
        'bic': bic,
        'BIC_samples': BIC_samples,  # Include per-sample BIC
        'mlppd': mlppd,
        'cp': cp,
        'alci': alci,
        'rmspe_samples': iteration_metrics['rmspe_samples'],
        'nsme_samples': iteration_metrics['nsme_samples'],
        'crps_samples': iteration_metrics['crps_samples'],
        'score_samples': iteration_metrics['score_samples'],
        'mlppd_samples': iteration_metrics['mlppd_samples'],
        'cp_samples': iteration_metrics['cp_samples'],
        'alci_samples': iteration_metrics['alci_samples'],
        'bic_samples': iteration_metrics['bic_samples']
    }


# =============================================================================
# Convergence Diagnostics (D>1)
# =============================================================================

def gelman_rubin_statistic(chains: List[np.ndarray]) -> float:
    """
    Compute Gelman-Rubin statistic (R-hat) for D>1.
    
    For vector parameters (like θ), returns max R-hat across dimensions.
    """
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    
    # Handle multi-dimensional parameters
    if chains[0].ndim == 1:
        chains = [c.reshape(-1, 1) for c in chains]
    
    n_dim = chains[0].shape[1]
    r_hats = np.zeros(n_dim)
    
    for d in range(n_dim):
        chain_means = np.array([np.mean(chains[i][:, d]) for i in range(n_chains)])
        chain_vars = np.array([np.var(chains[i][:, d], ddof=1) for i in range(n_chains)])
        
        B = n_samples * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        var_plus = ((n_samples - 1) / n_samples) * W + B / n_samples
        
        r_hats[d] = np.sqrt(var_plus / W) if W > 0 else 1.0
    
    return np.max(r_hats)


def heidelberg_welch_test(chain: np.ndarray, alpha: float = 0.05) -> Dict:
    """Heidelberg-Welch diagnostic test (works for vectors too)."""
    if chain.ndim > 1:
        # For vectors, test each dimension
        results = []
        for d in range(chain.shape[1]):
            results.append(heidelberg_welch_test(chain[:, d], alpha))
        # Return worst case
        stationary = all(r['stationary'] for r in results)
        halfwidth_test = all(r['halfwidth_test'] for r in results)
        return {
            'stationary': stationary,
            'halfwidth_test': halfwidth_test,
            'passed': stationary and halfwidth_test,
            'per_dimension': results
        }
    
    n = len(chain)
    
    # Need minimum samples for test
    if n < 10:
        return {
            'stationary': True,
            'halfwidth_test': True,
            'passed': True,
            'ks_statistic': np.nan,
            'ks_pvalue': np.nan,
            'relative_halfwidth': np.nan,
            'note': 'Insufficient samples for HW test (n<10)'
        }
    
    n_start = max(1, int(0.1 * n))
    n_end = max(2, int(0.5 * n))
    
    first_part = chain[:n_start]
    last_part = chain[-n_end:]
    
    if len(first_part) < 2 or len(last_part) < 2:
        return {'stationary': True, 'halfwidth_test': True, 'passed': True, 
                'note': 'Insufficient samples'}
    
    ks_stat, ks_pval = stats.ks_2samp(first_part, last_part)
    stationary = ks_pval > alpha
    
    chain_mean = np.mean(chain)
    chain_std = np.std(chain, ddof=1)
    halfwidth = 1.96 * chain_std / np.sqrt(n)
    relative_halfwidth = halfwidth / abs(chain_mean) if abs(chain_mean) > 1e-10 else halfwidth
    
    halfwidth_test = relative_halfwidth < 0.1
    
    return {
        'stationary': stationary,
        'halfwidth_test': halfwidth_test,
        'passed': stationary and halfwidth_test,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'relative_halfwidth': relative_halfwidth
    }


# =============================================================================
# Visualization Functions for D>1
# =============================================================================

def plot_trace(chains: List[np.ndarray], param_name: str, save_path: Optional[str] = None):
    """Trace plot for multiple chains (handles vectors)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['black', 'blue', 'red', 'green', 'purple']
    
    # Check if parameter is vector
    if chains[0].ndim > 1:
        D = chains[0].shape[1]
        for d in range(D):
            for i, chain in enumerate(chains):
                color = colors[i % len(colors)]
                ax.plot(chain[:, d], color=color, alpha=0.5, linewidth=0.8, 
                       label=f'Chain {i+1}, dim {d+1}' if d == 0 or i == 0 else '')
    else:
        for i, chain in enumerate(chains):
            color = colors[i % len(colors)]
            ax.plot(chain, color=color, alpha=0.7, linewidth=0.8, label=f'Chain {i+1}')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel(f'{param_name}', fontsize=14)
    ax.set_title(f'Trace Plot: {param_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_density(chains: List[np.ndarray], param_name: str, save_path: Optional[str] = None):
    """Density plot with mean and median (handles vectors)."""
    # If vector, plot each dimension separately
    if chains[0].ndim > 1:
        D = chains[0].shape[1]
        fig, axes = plt.subplots(1, D, figsize=(8*D, 6))
        if D == 1:
            axes = [axes]
        
        for d in range(D):
            ax = axes[d]
            all_samples = np.concatenate([c[:, d] for c in chains])
            
            ax.hist(all_samples, bins=50, density=True, alpha=0.6, 
                   color='skyblue', edgecolor='black')
            
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(all_samples)
            x_range = np.linspace(all_samples.min(), all_samples.max(), 200)
            ax.plot(x_range, kde(x_range), color='blue', linewidth=2)
            
            mean_val = np.mean(all_samples)
            median_val = np.median(all_samples)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean = {mean_val:.4f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                      label=f'Median = {median_val:.4f}')
            
            ax.set_xlabel(f'{param_name}[{d+1}]', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.set_title(f'Posterior Density: {param_name}[{d+1}]', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Scalar case (same as D=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        all_samples = np.concatenate(chains)
        
        ax.hist(all_samples, bins=50, density=True, alpha=0.6, 
               color='skyblue', edgecolor='black')
        
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(all_samples)
        x_range = np.linspace(all_samples.min(), all_samples.max(), 200)
        ax.plot(x_range, kde(x_range), color='blue', linewidth=2)
        
        mean_val = np.mean(all_samples)
        median_val = np.median(all_samples)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean = {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                  label=f'Median = {median_val:.4f}')
        
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'Posterior Density: {param_name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_autocorrelation(chain: np.ndarray, param_name: str, max_lag: int = 50,
                        save_path: Optional[str] = None):
    """Autocorrelation plot (handles vectors)."""
    # For vectors, plot each dimension
    if chain.ndim > 1:
        D = chain.shape[1]
        fig, axes = plt.subplots(1, D, figsize=(8*D, 6))
        if D == 1:
            axes = [axes]
        
        for d in range(D):
            ax = axes[d]
            acf_values = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    acf_values.append(1.0)
                else:
                    if len(chain[:, d]) > lag:
                        acf = np.corrcoef(chain[:-lag, d], chain[lag:, d])[0, 1]
                        acf_values.append(acf)
                    else:
                        acf_values.append(0.0)
            
            ax.bar(range(len(acf_values)), acf_values, color='steelblue', alpha=0.7)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax.axhline(1.96/np.sqrt(len(chain)), color='red', linestyle='--')
            ax.axhline(-1.96/np.sqrt(len(chain)), color='red', linestyle='--')
            
            ax.set_xlabel('Lag', fontsize=14)
            ax.set_ylabel('Autocorrelation', fontsize=14)
            ax.set_title(f'ACF: {param_name}[{d+1}]', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Scalar (same as D=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        acf_values = []
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                if len(chain) > lag:
                    acf = np.corrcoef(chain[:-lag], chain[lag:])[0, 1]
                    acf_values.append(acf)
        
        ax.bar(range(len(acf_values)), acf_values, color='steelblue', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(1.96/np.sqrt(len(chain)), color='red', linestyle='--', label='95% CI')
        ax.axhline(-1.96/np.sqrt(len(chain)), color='red', linestyle='--')
        
        ax.set_xlabel('Lag', fontsize=14)
        ax.set_ylabel('Autocorrelation', fontsize=14)
        ax.set_title(f'Autocorrelation: {param_name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_W_trace_multichain(chains_W: List[np.ndarray], save_path: Optional[str] = None):
    """Trace plot for W matrix (all elements, multiple chains)."""
    p, D = chains_W[0].shape[1], chains_W[0].shape[2]
    n_plots = min(p * D, 12)  # Limit to 12 plots
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    colors = ['black', 'blue', 'red', 'green']
    
    idx = 0
    for i in range(p):
        for j in range(D):
            if idx >= n_plots:
                break
            
            ax = axes[idx]
            
            for chain_id, chain_W in enumerate(chains_W):
                W_ij = chain_W[:, i, j]
                ax.plot(W_ij, color=colors[chain_id % len(colors)], 
                       alpha=0.7, linewidth=0.8, label=f'Chain {chain_id+1}')
            
            ax.set_ylabel(f'W_{i+1}{j+1}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8, ncol=len(chains_W))
            
            idx += 1
        if idx >= n_plots:
            break
    
    axes[-1].set_xlabel('Iteration', fontsize=12)
    plt.suptitle(f'Trace Plots: W Matrix (p×D={p}×{D}, showing first {n_plots})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                             ci_bounds: np.ndarray, save_path: Optional[str] = None):
    """Actual vs predicted plot with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    errors = np.array([
        y_pred - ci_bounds[:, 0],
        ci_bounds[:, 1] - y_pred
    ])
    
    ax.errorbar(y_true, y_pred, yerr=errors, fmt='o', alpha=0.5, capsize=3)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('Actual', fontsize=14)
    ax.set_ylabel('Predicted', fontsize=14)
    ax.set_title('Predicted vs Actual', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Multi-Chain Sampler for D>1
# =============================================================================

class MultiChainSampler:
    """
    Run multiple chains for D>1 and perform comprehensive diagnostics.
    """
    
    def __init__(self, D: int = 2, n_chains: int = 3, layer: int = 1,
                 n_iterations: int = 2000, burn_in: int = 500, thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g: bool = False,
                 use_mle_theta: bool = False,
                 use_mle_g_y: bool = False,      # For Layer 2: MLE for g_y (Y layer)
                 use_mle_theta_y: bool = False,   # For Layer 2: MLE for theta_y (Y layer)
                 use_mle_all: bool = False,
                 use_tf_gradients: bool = False,
                 kernel_type: str = 'separable_squared_exponential',
                 prior_M: Optional[np.ndarray] = None,
                 prior_V: Optional[np.ndarray] = None,
                 W_init: Optional[np.ndarray] = None,
                 M_init: Optional[np.ndarray] = None,
                 V_init: Optional[np.ndarray] = None,
                 Lambda_init: Optional[np.ndarray] = None,
                 tau2_y_init: float = 0.005,
                 tau2_q_init: Optional[Union[float, np.ndarray]] = 0.005,
                 tau2_r_init: Optional[Union[float, np.ndarray]] = 0.005,
                 g_y_init: float = 0.00009,
                 g_q_init: Optional[Union[float, np.ndarray]] = 0.00009,
                 g_r_init: Optional[Union[float, np.ndarray]] = 0.00009,
                 theta_y_init: Optional[Union[float, np.ndarray]] = 1.0,
                 theta_q_init: Optional[Union[float, np.ndarray]] = 1.0,
                 theta_r_init: Optional[Union[float, np.ndarray]] = 1.0,
                 Q_init: Optional[np.ndarray] = None,
                 R_init: Optional[np.ndarray] = None,
                 tau2_init: Optional[float] = None,
                 g_init: Optional[float] = None,
                 theta_init: Optional[Union[float, np.ndarray]] = None,
                 mv_sampler: str = "python",
                 rstiefel_rscol: Optional[int] = None):
        """
        Initialize multi-chain sampler for D>1.
        
        Args:
            D: Reduced dimension (must be ≥2)
            n_chains: Number of chains
            layer: Layer architecture
            n_iterations: Total iterations per chain
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for tau2 (default: False, for Layer 1)
            use_mle_g: Use MLE for g (default: False, for Layer 1)
            use_mle_theta: Use MLE for theta_D (default: False, for Layer 1)
            use_mle_all: Use MLE for all hyperparameters (overrides individual flags)
            use_tf_gradients: Use TensorFlow
            kernel_type: Kernel type for covariance functions
            prior_M: Prior for M (p, D), default: zeros
            prior_V: Prior for V (D, D), default: zeros
            mv_sampler: Backend for M/V Gibbs updates: 'python' or 'rstiefel'
            rstiefel_rscol: Optional number of columns for rstiefel simultaneous updates
        """
        if D < 2:
            raise ValueError("This module is for D>1. For D=1, use multichain_sampler_D1.py")
        
        self.D = D
        self.n_chains = n_chains
        self.layer = layer
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g = use_mle_g
        self.use_mle_theta = use_mle_theta
        self.use_mle_g_y = use_mle_g_y
        self.use_mle_theta_y = use_mle_theta_y
        self.use_mle_all = use_mle_all
        self.use_tf = use_tf_gradients
        self.kernel_type = kernel_type
        self.mv_sampler = mv_sampler
        self.rstiefel_rscol = rstiefel_rscol
        self.prior_M = prior_M
        self.prior_V = prior_V

        # Initial values (backward-compatible aliases map to layer-y values)
        self.W_init = W_init
        self.M_init = M_init
        self.V_init = V_init
        self.Lambda_init = Lambda_init
        self.tau2_y_init = tau2_y_init if tau2_init is None else float(tau2_init)
        self.tau2_q_init = tau2_q_init if tau2_q_init is not None else self.tau2_y_init
        self.tau2_r_init = tau2_r_init if tau2_r_init is not None else self.tau2_q_init
        self.g_y_init = g_y_init if g_init is None else float(g_init)
        self.g_q_init = g_q_init
        self.g_r_init = g_r_init
        self.theta_y_init = theta_y_init if theta_init is None else theta_init
        self.theta_q_init = theta_q_init
        self.theta_r_init = theta_r_init
        self.Q_init = Q_init
        self.R_init = R_init
        
        self.chains_samples = []
        self.chains_metrics = []
        self.computation_times = []
    
    def run_chains(self, Y_train: np.ndarray, X_train: np.ndarray,
                  Y_test: np.ndarray, X_test: np.ndarray,
                  verbose: bool = True) -> Dict:
        """Run multiple chains and compute predictions."""
        self.Y_train = Y_train
        self.X_train = X_train
        self.Y_test = Y_test
        self.X_test = X_test

        if verbose:
            print("="*70)
            print(f"Running {self.n_chains} Chains for {self.layer}-Layer Model (D={self.D})")
            print("="*70)
        
        for chain_id in range(self.n_chains):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Chain {chain_id + 1}/{self.n_chains}")
                print(f"{'='*70}")
            
            start_time = time.time()
            
            # Select sampler
            if self.layer == 1:
                sampler = GibbsSampler1Layer(
                    Y=Y_train, X=X_train, D=self.D,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g=self.use_mle_g,
                    use_mle_theta=self.use_mle_theta,
                    use_mle_all=self.use_mle_all,
                    use_tf_gradients=self.use_tf,
                    kernel_type=self.kernel_type,
                    prior_M=self.prior_M,
                    prior_V=self.prior_V,
                    W_init=self.W_init,
                    M_init=self.M_init,
                    V_init=self.V_init,
                    Lambda_init=self.Lambda_init,
                    tau2_y_init=self.tau2_y_init,
                    g_y_init=self.g_y_init,
                    theta_y_init=self.theta_y_init,
                    mv_sampler=self.mv_sampler,
                    rstiefel_rscol=self.rstiefel_rscol
                )
            elif self.layer == 2:
                sampler = GibbsSampler2Layer(
                    Y=Y_train, X=X_train, D=self.D,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g_y=self.use_mle_g_y,
                    use_mle_theta_y=self.use_mle_theta_y,
                    use_mle_all=self.use_mle_all,
                    use_tf_gradients=self.use_tf,
                    kernel_type=self.kernel_type,
                    prior_M=self.prior_M,
                    prior_V=self.prior_V,
                    W_init=self.W_init,
                    M_init=self.M_init,
                    V_init=self.V_init,
                    Lambda_init=self.Lambda_init,
                    tau2_y_init=self.tau2_y_init,
                    tau2_q_init=self.tau2_q_init,
                    g_y_init=self.g_y_init,
                    g_q_init=self.g_q_init,
                    theta_y_init=self.theta_y_init,
                    theta_q_init=self.theta_q_init,
                    Q_init=self.Q_init,
                    mv_sampler=self.mv_sampler,
                    rstiefel_rscol=self.rstiefel_rscol
                )
            else:
                sampler = GibbsSampler3Layer(
                    Y=Y_train, X=X_train, D=self.D,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g_y=self.use_mle_g_y,
                    use_mle_theta_y=self.use_mle_theta_y,
                    use_mle_all=self.use_mle_all,
                    use_tf_gradients=self.use_tf,
                    kernel_type=self.kernel_type,
                    prior_M=self.prior_M,
                    prior_V=self.prior_V,
                    W_init=self.W_init,
                    M_init=self.M_init,
                    V_init=self.V_init,
                    Lambda_init=self.Lambda_init,
                    tau2_y_init=self.tau2_y_init,
                    tau2_q_init=self.tau2_q_init,
                    tau2_r_init=self.tau2_r_init,
                    g_y_init=self.g_y_init,
                    g_q_init=self.g_q_init,
                    g_r_init=self.g_r_init,
                    theta_y_init=self.theta_y_init,
                    theta_q_init=self.theta_q_init,
                    theta_r_init=self.theta_r_init,
                    Q_init=self.Q_init,
                    R_init=self.R_init,
                    mv_sampler=self.mv_sampler,
                    rstiefel_rscol=self.rstiefel_rscol
                )
            
            samples = sampler.run(verbose=verbose)
            
            # Compute predictions and metrics
            metrics = Bayesian_Metrics_with_quantiles(
                samples, X_train, Y_train, X_test, Y_test, layer=self.layer
            )
            
            elapsed_time = time.time() - start_time
            
            self.chains_samples.append(samples)
            self.chains_metrics.append(metrics)
            self.computation_times.append(elapsed_time)
            
            if verbose:
                print(f"\nChain {chain_id + 1} complete in {elapsed_time:.1f}s")
                print(
                    f"RMSPE: {metrics['rmspe']:.4f}, NSME: {metrics['nsme']:.4f}, "
                    f"CP: {metrics['cp']:.4f}, ALCI: {metrics['alci']:.4f}"
                )
        
        if verbose:
            print("\n" + "="*70)
            print("All chains complete!")
            print("="*70)
        
        return self.compute_summary()
    
    def compute_summary(self) -> Dict:
        """Compute summary statistics across chains (adapts to layer and D>1)."""
        # Extract parameter chains based on layer
        tau2_chains = [_sample_array(s, 'tau2_y', 'tau2') for s in self.chains_samples]
        
        convergence_dict = {}
        
        # Layer-dependent parameters
        if self.layer == 1:
            # 1-Layer: tau2_y, g_y, theta_D_y (vector for D>1), W
            g_chains = [_sample_array(s, 'g_y', 'g') for s in self.chains_samples]
            theta_chains = [_sample_array(s, 'theta_D_y', 'theta_D') for s in self.chains_samples]  # (n_samples, D)
            
            convergence_dict['r_hat_tau2'] = gelman_rubin_statistic(tau2_chains)
            convergence_dict['r_hat_g'] = gelman_rubin_statistic(g_chains)
            convergence_dict['r_hat_theta'] = gelman_rubin_statistic(theta_chains)  # Max across D
            convergence_dict['hw_tau2'] = heidelberg_welch_test(tau2_chains[0])
            convergence_dict['hw_g'] = heidelberg_welch_test(g_chains[0])
            convergence_dict['hw_theta'] = heidelberg_welch_test(theta_chains[0])  # Per dimension
            
        elif self.layer == 2:
            # 2-Layer: tau2, g_y, g_q, theta_y (vector), theta_q (vector), Q, W
            g_y_chains = [s['g_y'] for s in self.chains_samples]
            g_q_chains = [s['g_q'] for s in self.chains_samples]
            theta_y_chains = [s['theta_y'] for s in self.chains_samples]  # (n_samples, D)
            theta_q_chains = [s['theta_q'] for s in self.chains_samples]  # (n_samples, D)
            
            convergence_dict['r_hat_tau2'] = gelman_rubin_statistic(tau2_chains)
            convergence_dict['r_hat_g_y'] = gelman_rubin_statistic(g_y_chains)
            convergence_dict['r_hat_g_q'] = gelman_rubin_statistic(g_q_chains)
            convergence_dict['r_hat_theta_y'] = gelman_rubin_statistic(theta_y_chains)
            convergence_dict['r_hat_theta_q'] = gelman_rubin_statistic(theta_q_chains)
            convergence_dict['hw_tau2'] = heidelberg_welch_test(tau2_chains[0])
            convergence_dict['hw_g_y'] = heidelberg_welch_test(g_y_chains[0])
            convergence_dict['hw_g_q'] = heidelberg_welch_test(g_q_chains[0])
            convergence_dict['hw_theta_y'] = heidelberg_welch_test(theta_y_chains[0])
            convergence_dict['hw_theta_q'] = heidelberg_welch_test(theta_q_chains[0])
            
        else:  # layer == 3
            # 3-Layer: tau2, g_y, g_q, g_r, theta_y, theta_q, theta_r (all vectors for D>1), R, Q, W
            g_y_chains = [s['g_y'] for s in self.chains_samples]
            g_q_chains = [s['g_q'] for s in self.chains_samples]
            g_r_chains = [s['g_r'] for s in self.chains_samples]
            theta_y_chains = [s['theta_y'] for s in self.chains_samples]
            theta_q_chains = [s['theta_q'] for s in self.chains_samples]
            theta_r_chains = [s['theta_r'] for s in self.chains_samples]
            
            convergence_dict['r_hat_tau2'] = gelman_rubin_statistic(tau2_chains)
            convergence_dict['r_hat_g_y'] = gelman_rubin_statistic(g_y_chains)
            convergence_dict['r_hat_g_q'] = gelman_rubin_statistic(g_q_chains)
            convergence_dict['r_hat_g_r'] = gelman_rubin_statistic(g_r_chains)
            convergence_dict['r_hat_theta_y'] = gelman_rubin_statistic(theta_y_chains)
            convergence_dict['r_hat_theta_q'] = gelman_rubin_statistic(theta_q_chains)
            convergence_dict['r_hat_theta_r'] = gelman_rubin_statistic(theta_r_chains)
            convergence_dict['hw_tau2'] = heidelberg_welch_test(tau2_chains[0])
            convergence_dict['hw_g_y'] = heidelberg_welch_test(g_y_chains[0])
            convergence_dict['hw_g_q'] = heidelberg_welch_test(g_q_chains[0])
            convergence_dict['hw_g_r'] = heidelberg_welch_test(g_r_chains[0])
            convergence_dict['hw_theta_y'] = heidelberg_welch_test(theta_y_chains[0])
            convergence_dict['hw_theta_q'] = heidelberg_welch_test(theta_q_chains[0])
            convergence_dict['hw_theta_r'] = heidelberg_welch_test(theta_r_chains[0])
        
        # Metrics
        all_rmspe = [m['rmspe'] for m in self.chains_metrics]
        all_nsme = [m['nsme'] for m in self.chains_metrics]
        all_crps = [m['crps'] for m in self.chains_metrics]
        all_score = [m['score'] for m in self.chains_metrics]
        all_mlppd = [m['mlppd'] for m in self.chains_metrics]
        all_bic = [m['bic'] for m in self.chains_metrics]
        all_cp = [m['cp'] for m in self.chains_metrics]
        all_alci = [m['alci'] for m in self.chains_metrics]
        
        # Helper function for summary with median and CI
        def get_summary(values):
            return {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
        
        parameter_diagnostics = None
        parameter_diagnostics_error = None
        try:
            parameter_diagnostics = compute_multichain_parameter_diagnostics(
                chains=self.chains_samples,
                burn=self.burn_in,
                ci=0.95,
                use_projection_for_W=False
            )
        except Exception as exc:
            parameter_diagnostics_error = str(exc)
        
        results = {
            'chains_samples': self.chains_samples,
            'chains_metrics': self.chains_metrics,
            'computation_times': self.computation_times,
            'convergence': convergence_dict,
            'metrics_summary': {
                'rmspe': get_summary(all_rmspe),
                'nsme': get_summary(all_nsme),
                'crps': get_summary(all_crps),
                'score': get_summary(all_score),
                'bic': get_summary(all_bic),
                'mlppd': get_summary(all_mlppd),
                'cp': get_summary(all_cp),
                'alci': get_summary(all_alci)
            }
        }
        
        if parameter_diagnostics is not None:
            results['parameter_diagnostics'] = parameter_diagnostics
        if parameter_diagnostics_error is not None:
            results['parameter_diagnostics_error'] = parameter_diagnostics_error
        
        return results
    
    def create_all_diagnostics(self, output_dir: str = './plots'):
        """Create all diagnostic plots."""
        import os
        from BDR_plot import (  # type: ignore[import]
            plot_trace as bdr_plot_trace,
            plot_density as bdr_plot_density,
            plot_autocorrelation as bdr_plot_autocorrelation,
            plot_W_trace_multichain as bdr_plot_W_trace_multichain,
            plot_W_projection_trace_multichain as bdr_plot_W_projection_trace_multichain,
            plot_actual_vs_predicted as bdr_plot_actual_vs_predicted,
        )

        os.makedirs(output_dir, exist_ok=True)

        parameter_specs = [
            ('tau2_y', ('tau2_y', 'tau2')),
            ('g_y', ('g_y', 'g')),
            ('theta_y', ('theta_D_y', 'theta_D', 'theta_y')),
        ]
        if self.layer >= 2:
            parameter_specs.extend([
                ('g_q', ('g_q',)),
                ('theta_q', ('theta_q',)),
            ])
        if self.layer >= 3:
            parameter_specs.extend([
                ('g_r', ('g_r',)),
                ('theta_r', ('theta_r',)),
            ])
        component_parameter_specs = [
            ('Lambda', ('Lambda',)),
            ('M', ('M',)),
            ('V', ('V',)),
        ]
        if self.layer >= 2:
            component_parameter_specs.append(('Q', ('Q',)))
        if self.layer >= 3:
            component_parameter_specs.append(('R', ('R',)))

        for param_name, keys in parameter_specs:
            chains = [_sample_array(s, *keys) for s in self.chains_samples]
            bdr_plot_trace(chains, param_name, f'{output_dir}/trace_{param_name}_D{self.D}.png')
            bdr_plot_density(chains, param_name, f'{output_dir}/density_{param_name}_D{self.D}.png')
            bdr_plot_autocorrelation(
                chains[0],
                param_name,
                save_path=f'{output_dir}/acf_{param_name}_D{self.D}.png'
            )

        for param_name, keys in component_parameter_specs:
            chains = _diagnostic_chains_for_keys(self.chains_samples, keys)
            bdr_plot_trace(chains, param_name, f'{output_dir}/trace_{param_name}_D{self.D}.png')
            bdr_plot_density(chains, param_name, f'{output_dir}/density_{param_name}_D{self.D}.png')
            bdr_plot_autocorrelation(
                chains[0],
                param_name,
                save_path=f'{output_dir}/acf_{param_name}_D{self.D}.png'
            )
        
        # W trace
        W_chains = [s['W'] for s in self.chains_samples]
        W_component_chains = _diagnostic_chains_for_keys(self.chains_samples, ('W',))
        WWT_component_chains = _projection_chains_for_W(W_chains)
        bdr_plot_W_trace_multichain(W_chains, save_path=f'{output_dir}/trace_W_D{self.D}.png')
        bdr_plot_W_projection_trace_multichain(W_chains, save_path=f'{output_dir}/trace_WWT_D{self.D}.png')
        bdr_plot_density(W_component_chains, 'W', save_path=f'{output_dir}/density_W_D{self.D}.png')
        bdr_plot_autocorrelation(W_component_chains[0], 'W', save_path=f'{output_dir}/acf_W_D{self.D}.png')
        bdr_plot_density(WWT_component_chains, 'WWT', save_path=f'{output_dir}/density_WWT_D{self.D}.png')
        bdr_plot_autocorrelation(WWT_component_chains[0], 'WWT', save_path=f'{output_dir}/acf_WWT_D{self.D}.png')
        
        # Predictions
        metrics = self.chains_metrics[0]
        bdr_plot_actual_vs_predicted(
            self.Y_test, metrics['y_pred_mean'], metrics['y_pred_quantiles'],
            save_path=f'{output_dir}/pred_vs_actual_D{self.D}.png'
        )
        
        print(f"\nAll diagnostic plots saved to: {output_dir}/")


if __name__ == "__main__":
    print("="*70)
    print("Multi-Chain Sampler for D>1 (Test with 2 samples)")
    print("="*70)
    
    # Generate test data
    np.random.seed(42)
    n_train, n_test, p, D = 25, 10, 5, 2
    
    X_train = np.random.randn(n_train, p)
    X_test = np.random.randn(n_test, p)
    
    W_true = np.random.randn(p, D)
    W_true, _ = np.linalg.qr(W_true)
    
    Z_train = X_train @ W_true
    theta_true = np.array([1.0, 1.5])
    C = covar_sep(Z_train, theta_true, g=0.01)
    Y_train = np.random.multivariate_normal(np.zeros(n_train), C)
    Y_test = np.sin(X_test @ W_true).sum(axis=1) + 0.1 * np.random.randn(n_test)
    
    print(f"\nData: n_train={n_train}, n_test={n_test}, p={p}, D={D}")
    
    # Run multi-chain (2 samples per chain for testing)
    multichain = MultiChainSampler(
        D=D,
        n_chains=2,
        layer=1,
        n_iterations=2,
        burn_in=0,
        thin=1,
        use_mle_all=True
    )
    
    results = multichain.run_chains(Y_train, X_train, Y_test, X_test, verbose=True)
    
    print("\n" + "="*70)
    print("Multi-Chain Results Summary")
    print("="*70)
    
    print(f"\nConvergence Diagnostics:")
    print(f"  R-hat tau2: {results['convergence']['r_hat_tau2']:.4f}")
    print(f"  R-hat g: {results['convergence']['r_hat_g']:.4f}")
    print(f"  R-hat theta (max across D={D}): {results['convergence']['r_hat_theta']:.4f}")
    
    print(f"\nMetrics (mean ± std across chains):")
    print(f"  RMSPE: {results['metrics_summary']['rmspe']['mean']:.4f} ± {results['metrics_summary']['rmspe']['std']:.4f}")
    print(f"  NSME: {results['metrics_summary']['nsme']['mean']:.4f} ± {results['metrics_summary']['nsme']['std']:.4f}")
    print(f"  MLPPD: {results['metrics_summary']['mlppd']['mean']:.4f} ± {results['metrics_summary']['mlppd']['std']:.4f}")
    print(f"  CP: {results['metrics_summary']['cp']['mean']:.4f} ± {results['metrics_summary']['cp']['std']:.4f}")
    print(f"  ALCI: {results['metrics_summary']['alci']['mean']:.4f} ± {results['metrics_summary']['alci']['std']:.4f}")
    
    print("\n" + "="*70)
    print(f"✓✓✓ Multi-chain test complete for D={D} with 2 samples! ✓✓✓")
    print("="*70)
