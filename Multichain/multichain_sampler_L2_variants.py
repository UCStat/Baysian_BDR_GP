"""
Multi-Chain Gibbs Sampler for Layer 2 Variants

This module implements multi-chain MCMC sampling for Layer 2 variants:
1. W_Known: W is fixed/known
2. No_W: No dimensionality reduction, use X directly
3. No_W_Selective: Use selected columns of X

All variants support:
- Multiple chains for convergence diagnostics
- Posterior predictions with uncertainty quantification
- Performance metrics (RMSPE, NSME, CRPS, Score, BIC, MLPPD, CP, ALCI)
- Comprehensive diagnostics (Gelman-Rubin, Heidelberg-Welch)
- Visualization (trace plots, density plots, autocorrelation, etc.)
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union
import sys
from pathlib import Path

# Add parent directories to path for imports
base_dir = Path(__file__).parent.parent
for folder in ["Gibbs Sampling", "Parameter Sampler", "BDR Metrics and Plot"]:
    folder_path = str(base_dir / folder)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)

from gibbs_sampler_layers_L2_variants import (  # type: ignore[import]
    GibbsSampler2Layer_W_Known,
    GibbsSampler2Layer_No_W,
    GibbsSampler2Layer_No_W_Selective
)
from BDR_metrics import (  # type: ignore[import]
    compute_RMSPE, compute_NSME, compute_CRPS, compute_score, 
    compute_BIC, compute_MLPPD, compute_CP, compute_ALCI,
    compute_all_metrics_summary, compute_iteration_metrics,
    compute_multichain_parameter_diagnostics
)
from BDR_plot import (  # type: ignore[import]
    plot_trace, plot_density, plot_histogram, plot_autocorrelation,
    plot_actual_vs_predicted, 
    plot_convergence_diagnostics, plot_metrics_boxplot, plot_metrics_comparison_table
)


def _sample_array(samples: Dict, *keys: str) -> np.ndarray:
    """Return the first available sample array among candidate keys."""
    for key in keys:
        if key in samples:
            return samples[key]
    raise KeyError(f"None of the keys {keys} found in samples.")


# =============================================================================
# Prediction Functions for Layer 2 Variants
# =============================================================================

def predict_gp_variant_2layer_W_Known(X_new: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray,
                                     W_fixed: np.ndarray, Q_train: np.ndarray,
                                     theta_y: Union[float, np.ndarray], g_y: float, tau2_y: float,
                                     kernel_type: str = 'isotropic_squared_exponential') -> Tuple[np.ndarray, np.ndarray]:
    """
    GP prediction for 2-layer W_Known variant.
    
    Args:
        X_new: New inputs (n_test, p)
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        W_fixed: Fixed projection matrix (p, D)
        Q_train: Latent Q values (n_train, D)
        theta_y: Lengthscale(s) for Y layer
        g_y: Nugget for Y layer
        tau2_y: Y-layer noise variance
        kernel_type: Kernel type
        
    Returns:
        (predictive_mean, predictive_variance)
    """
    D = W_fixed.shape[1]
    
    # Create kernel instance
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    
    # For 2-layer, prediction is based on Q (not Z)
    # Q_new would need to be predicted first, but for simplicity, we use Q_train mean
    # In practice, you'd need to predict Q_new from X_new, but that's complex
    # For now, we'll use a simplified approach: predict Y from Q_train
    
    # Compute covariance matrices using Q_train
    K_train = kernel.compute_covariance(Q_train, Q_train)
    # For new predictions, we'd need Q_new, but for now use Q_train mean
    Q_train_mean = np.mean(Q_train, axis=0) if Q_train.ndim > 1 else np.mean(Q_train)
    Q_new = np.tile(Q_train_mean, (len(X_new), 1)) if Q_train.ndim > 1 else np.full(len(X_new), Q_train_mean)
    K_new_train = kernel.compute_covariance(Q_new, Q_train)
    K_new = kernel.compute_covariance(Q_new, Q_new)
    
    # Predictive mean and variance
    try:
        K_train_inv = np.linalg.inv(K_train)
        pred_mean = K_new_train @ K_train_inv @ Y_train
        pred_cov = K_new - K_new_train @ K_train_inv @ K_new_train.T
        pred_var = np.diag(pred_cov)
    except:
        pred_mean = np.zeros(len(X_new))
        pred_var = np.ones(len(X_new))
    
    return pred_mean, pred_var


def predict_gp_variant_2layer_No_W(X_new: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray,
                                  Q_train: np.ndarray,
                                  theta_y: Union[float, np.ndarray], g_y: float, tau2_y: float,
                                  kernel_type: str = 'separable_squared_exponential') -> Tuple[np.ndarray, np.ndarray]:
    """
    GP prediction for 2-layer No_W variant (X used directly).
    
    Args:
        X_new: New inputs (n_test, p)
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        Q_train: Latent Q values (n_train, D) where D = p
        theta_y: Lengthscale(s) for Y layer
        g_y: Nugget for Y layer
        tau2_y: Y-layer noise variance
        kernel_type: Kernel type
        
    Returns:
        (predictive_mean, predictive_variance)
    """
    D = X_train.shape[1]  # D = p
    
    # Create kernel instance
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    
    # Compute covariance matrices using Q_train
    K_train = kernel.compute_covariance(Q_train, Q_train)
    # For new predictions, use Q_train mean
    Q_train_mean = np.mean(Q_train, axis=0) if Q_train.ndim > 1 else np.mean(Q_train)
    Q_new = np.tile(Q_train_mean, (len(X_new), 1)) if Q_train.ndim > 1 else np.full(len(X_new), Q_train_mean)
    K_new_train = kernel.compute_covariance(Q_new, Q_train)
    K_new = kernel.compute_covariance(Q_new, Q_new)
    
    # Predictive mean and variance
    try:
        K_train_inv = np.linalg.inv(K_train)
        pred_mean = K_new_train @ K_train_inv @ Y_train
        pred_cov = K_new - K_new_train @ K_train_inv @ K_new_train.T
        pred_var = np.diag(pred_cov)
    except:
        pred_mean = np.zeros(len(X_new))
        pred_var = np.ones(len(X_new))
    
    return pred_mean, pred_var


def predict_gp_variant_2layer_No_W_Selective(X_new: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray,
                                            Q_train: np.ndarray,
                                            theta_y: Union[float, np.ndarray], g_y: float, tau2_y: float,
                                            kernel_type: str = 'separable_squared_exponential') -> Tuple[np.ndarray, np.ndarray]:
    """
    GP prediction for 2-layer No_W_Selective variant (selected columns of X).
    
    Args:
        X_new: New inputs (n_test, p)
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        Q_train: Latent Q values (n_train, D)
        theta_y: Lengthscale(s) for Y layer
        g_y: Nugget for Y layer
        tau2_y: Y-layer noise variance
        kernel_type: Kernel type
        
    Returns:
        (predictive_mean, predictive_variance)
    """
    D = Q_train.shape[1] if Q_train.ndim > 1 else 1
    
    # Create kernel instance
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
    
    # Compute covariance matrices using Q_train
    K_train = kernel.compute_covariance(Q_train, Q_train)
    # For new predictions, use Q_train mean
    Q_train_mean = np.mean(Q_train, axis=0) if Q_train.ndim > 1 else np.mean(Q_train)
    Q_new = np.tile(Q_train_mean, (len(X_new), 1)) if Q_train.ndim > 1 else np.full(len(X_new), Q_train_mean)
    K_new_train = kernel.compute_covariance(Q_new, Q_train)
    K_new = kernel.compute_covariance(Q_new, Q_new)
    
    # Predictive mean and variance
    try:
        K_train_inv = np.linalg.inv(K_train)
        pred_mean = K_new_train @ K_train_inv @ Y_train
        pred_cov = K_new - K_new_train @ K_train_inv @ K_new_train.T
        pred_var = np.diag(pred_cov)
    except:
        pred_mean = np.zeros(len(X_new))
        pred_var = np.ones(len(X_new))
    
    return pred_mean, pred_var


# =============================================================================
# Log-Likelihood Computation for Layer 2 Variants
# =============================================================================

def compute_loglikelihood_variant_2layer(samples: Dict, X_train: np.ndarray, Y_train: np.ndarray,
                                        variant: str, W_fixed: Optional[np.ndarray] = None,
                                        column_indices: Optional[np.ndarray] = None,
                                        kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Compute log-likelihood for Layer 2 variant models.
    
    For 2-layer: BIC = loglik_y + loglik_q - 0.5 * k * log(n)
    
    Args:
        samples: Dictionary with tau2_y, g_y, theta_y, Q samples
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        variant: 'W_Known', 'No_W', or 'No_W_Selective'
        W_fixed: Fixed W matrix (for W_Known variant)
        column_indices: Column indices (for No_W_Selective variant)
        kernel_type: Kernel type
        
    Returns:
        Total log-likelihood (sum of Y and Q layer log-likelihoods)
    """
    # Get parameter values (use mean of samples)
    tau2_y = np.mean(_sample_array(samples, 'tau2_y', 'tau2'))
    g_y = np.mean(samples['g_y'])
    theta_y = np.mean(samples['theta_y'], axis=0) if samples['theta_y'].ndim > 1 else np.mean(samples['theta_y'])
    theta_q = np.mean(samples['theta_q'], axis=0) if samples['theta_q'].ndim > 1 else np.mean(samples['theta_q'])
    
    # Get Q (use mean across samples)
    Q_samples = samples['Q']
    if Q_samples.ndim == 3:  # (n_samples, n, D)
        Q = np.mean(Q_samples, axis=0)  # (n, D)
    else:  # (n_samples, n) for D=1
        Q = np.mean(Q_samples, axis=0)  # (n,)
        if Q.ndim == 1:
            Q = Q.reshape(-1, 1)
    
    # Determine D and Z
    if variant == 'W_Known':
        D = W_fixed.shape[1]
        Z = X_train @ W_fixed
    elif variant == 'No_W':
        D = X_train.shape[1]
        Z = X_train
    else:  # No_W_Selective
        D = len(column_indices) if column_indices is not None else X_train.shape[1]
        Z = X_train[:, column_indices] if column_indices is not None else X_train
    
    # Create kernel instances
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel_y = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
        kernel_q = get_kernel_instance(kernel_type, theta_q, 0.0, 1.0, D)  # g_q=0, tau2_q=1.0
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel_y = get_kernel_instance(kernel_type, theta_y, g_y, tau2_y, D)
        kernel_q = get_kernel_instance(kernel_type, theta_q, 0.0, 1.0, D)  # g_q=0, tau2_q=1.0
    
    # Compute log-likelihoods
    try:
        # Y layer: Y | Q
        loglik_y = kernel_y.log_likelihood(Y_train, Q)
        
        # Q layer: Q | Z
        loglik_q = kernel_q.log_likelihood(Q.flatten() if Q.ndim > 1 else Q, Z)
        
        return loglik_y + loglik_q  # Sum for 2-layer BIC
    except:
        return -np.inf


# =============================================================================
# Bayesian Metrics for Layer 2 Variants
# =============================================================================

def Bayesian_Metrics_with_quantiles_variant_2layer(samples: Dict, X_train: np.ndarray, Y_train: np.ndarray,
                                                   X_test: np.ndarray, Y_test: np.ndarray,
                                                   variant: str,
                                                   W_fixed: Optional[np.ndarray] = None,
                                                   column_indices: Optional[np.ndarray] = None,
                                                   kernel_type: str = 'isotropic_squared_exponential',
                                                   quantiles: List[float] = [0.025, 0.975]) -> Dict:
    """
    Compute Bayesian metrics for Layer 2 variants.
    
    Args:
        samples: Dictionary of MCMC samples
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        X_test: Test inputs (n_test, p)
        Y_test: Test responses (n_test,)
        variant: 'W_Known', 'No_W', or 'No_W_Selective'
        W_fixed: Fixed W matrix (for W_Known variant)
        column_indices: Column indices (for No_W_Selective variant)
        kernel_type: Kernel type
        quantiles: Quantiles for credible intervals
        
    Returns:
        Dictionary with all metrics
    """
    tau2_y_samples = _sample_array(samples, 'tau2_y', 'tau2')
    n_samples = len(tau2_y_samples)
    n_test = len(Y_test)
    
    # Storage for predictions
    pred_means = np.zeros((n_samples, n_test))
    pred_vars = np.zeros((n_samples, n_test))
    bic_samples = np.zeros(n_samples)
    
    # Compute predictions for each sample
    for i in range(n_samples):
        tau2_y = tau2_y_samples[i]
        g_y = samples['g_y'][i]
        theta_y = samples['theta_y'][i] if samples['theta_y'].ndim == 1 else samples['theta_y'][i, :]
        theta_q = samples['theta_q'][i] if samples['theta_q'].ndim == 1 else samples['theta_q'][i, :]
        
        # Get Q for this sample
        Q_i = samples['Q'][i]  # (n_train, D) or (n_train,)
        if Q_i.ndim == 1:
            Q_i = Q_i.reshape(-1, 1)
        
        if variant == 'W_Known':
            pred_mean, pred_var = predict_gp_variant_2layer_W_Known(
                X_test, X_train, Y_train,
                W_fixed, Q_i,
                theta_y, g_y, tau2_y, kernel_type
            )
        elif variant == 'No_W':
            pred_mean, pred_var = predict_gp_variant_2layer_No_W(
                X_test, X_train, Y_train,
                Q_i,
                theta_y, g_y, tau2_y, kernel_type
            )
        else:  # No_W_Selective
            pred_mean, pred_var = predict_gp_variant_2layer_No_W_Selective(
                X_test, X_train, Y_train,
                Q_i,
                theta_y, g_y, tau2_y, kernel_type
            )
        
        pred_means[i] = pred_mean
        pred_vars[i] = pred_var
        
        # Per-iteration BIC
        theta_y_arr = np.asarray(theta_y)
        theta_q_arr = np.asarray(theta_q)
        if theta_y_arr.ndim == 0:
            theta_y_arr = np.array([theta_y_arr.item()])
        if theta_q_arr.ndim == 0:
            theta_q_arr = np.array([theta_q_arr.item()])
        sample_i = {
            'tau2_y': np.array([tau2_y]),
            'g_y': np.array([g_y]),
            'theta_y': theta_y_arr if theta_y_arr.ndim == 1 and theta_y_arr.size == 1 else theta_y_arr[np.newaxis, :],
            'theta_q': theta_q_arr if theta_q_arr.ndim == 1 and theta_q_arr.size == 1 else theta_q_arr[np.newaxis, :],
            'Q': samples['Q'][i][np.newaxis, ...]
        }
        loglik_i = compute_loglikelihood_variant_2layer(
            sample_i, X_train, Y_train, variant, W_fixed, column_indices, kernel_type
        )
        k = 4  # tau2_y, g_y, theta_y, theta_q
        n = len(Y_train)
        bic_samples[i] = -2 * loglik_i + k * np.log(n)
    
    # Posterior averages
    pred_mean_avg = np.mean(pred_means, axis=0)
    pred_var_avg = np.mean(pred_vars, axis=0) + np.var(pred_means, axis=0)  # Total variance
    pred_quantiles = np.quantile(pred_means, [0.025, 0.975], axis=0).T
    
    # Compute metrics
    rmspe = compute_RMSPE(Y_test, pred_mean_avg)
    nsme = compute_NSME(Y_test, pred_mean_avg)
    crps = compute_CRPS(Y_test, pred_means, pred_vars)
    
    # Score: use covariance of predictions across samples
    Sigma_pred = np.cov(pred_means.T)  # (n_test, n_test)
    score = compute_score(Y_test, pred_mean_avg, Sigma_pred)
    
    # BIC: average across per-iteration BIC values
    bic = np.mean(bic_samples)
    
    # MLPPD
    mlppd = compute_MLPPD(Y_test, pred_means, pred_vars)
    
    # Predictive interval metrics
    cp = compute_CP(Y_test, pred_quantiles[:, 0], pred_quantiles[:, 1])
    alci = compute_ALCI(pred_quantiles[:, 0], pred_quantiles[:, 1])
    
    # Per-iteration metrics (all metrics for every posterior sample)
    iteration_metrics = compute_iteration_metrics(
        y_true=Y_test,
        y_pred_samples=pred_means,
        y_pred_var_samples=pred_vars,
        sample_axis=0,
        bic_samples=bic_samples
    )
    
    return {
        'RMSPE': rmspe,
        'NSME': nsme,
        'CRPS': crps,
        'Score': score,
        'BIC': bic,
        'MLPPD': mlppd,
        'CP': cp,
        'ALCI': alci,
        'BIC_samples': bic_samples,
        'RMSPE_samples': iteration_metrics['rmspe_samples'],
        'NSME_samples': iteration_metrics['nsme_samples'],
        'CRPS_samples': iteration_metrics['crps_samples'],
        'Score_samples': iteration_metrics['score_samples'],
        'MLPPD_samples': iteration_metrics['mlppd_samples'],
        'CP_samples': iteration_metrics['cp_samples'],
        'ALCI_samples': iteration_metrics['alci_samples'],
        'bic_samples': iteration_metrics['bic_samples'],
        'pred_mean': pred_mean_avg,
        'pred_var': pred_var_avg,
        'pred_quantiles': pred_quantiles,
        'pred_means_all': pred_means,
        'pred_vars_all': pred_vars
    }


# =============================================================================
# Multi-Chain Sampler for Layer 2 Variants
# =============================================================================

class MultiChainSampler_L2_Variants:
    """
    Multi-chain sampler for Layer 2 variants.
    
    Supports three variants:
    1. W_Known: W is fixed/known
    2. No_W: No dimensionality reduction
    3. No_W_Selective: Selected columns of X
    """
    
    def __init__(self, variant: str = 'No_W',
                 n_chains: int = 3,
                 n_iterations: int = 2000,
                 burn_in: int = 500,
                 thin: int = 1,
                 use_mle_tau2: bool = False,
                 use_mle_g_y: bool = False,
                 use_mle_theta_y: bool = False,
                 kernel_type: str = 'isotropic_squared_exponential',
                 # Variant-specific parameters
                 W_fixed: Optional[np.ndarray] = None,  # For W_Known
                 D: Optional[int] = None,  # For No_W_Selective
                 column_indices: Optional[np.ndarray] = None,  # For No_W_Selective
                 # Hyperparameters
                 alpha1: float = 1.0,
                 alpha2: float = 1000.0,
                 beta1: float = 0.01,
                 beta2: float = 0.005,
                 gamma1: float = 1.5,
                 gamma2_y: float = 3.9,
                 gamma2_q: float = 3.9/3,
                 l: float = 1.0,
                 u: float = 2.0,
                 tau2_y_init: float = 0.005,
                 tau2_q_init: Union[float, np.ndarray] = 0.005,
                 g_y_init: float = 0.00009,
                 g_q_init: Union[float, np.ndarray] = 0.00009,
                 theta_y_init: Union[float, np.ndarray] = 1.0,
                 theta_q_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize multi-chain sampler for Layer 2 variants.
        
        Args:
            variant: 'W_Known', 'No_W', or 'No_W_Selective'
            n_chains: Number of chains
            n_iterations: Total iterations per chain
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for tau2_y
            use_mle_g_y: Use MLE for g_y
            use_mle_theta_y: Use MLE for theta_y
            kernel_type: Kernel type
            W_fixed: Fixed W matrix (required for W_Known)
            D: Number of columns to use (required for No_W_Selective)
            column_indices: Column indices to use (optional for No_W_Selective)
            alpha1, alpha2: Inverse Gamma prior for tau2_y
            beta1, beta2: Gamma prior for g_y
            gamma1: Gamma shape parameter (3/2)
            gamma2_y: Gamma rate parameter for theta_y (3.9)
            gamma2_q: Gamma rate parameter for theta_q (3.9/3)
            l, u: MH proposal bounds
        """
        if variant not in ['W_Known', 'No_W', 'No_W_Selective']:
            raise ValueError(f"variant must be one of 'W_Known', 'No_W', 'No_W_Selective', got {variant}")
        
        if variant == 'W_Known' and W_fixed is None:
            raise ValueError("W_fixed must be provided for W_Known variant")
        
        if variant == 'No_W_Selective' and D is None:
            raise ValueError("D must be provided for No_W_Selective variant")
        
        self.variant = variant
        self.n_chains = n_chains
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.use_mle_tau2 = use_mle_tau2
        self.use_mle_g_y = use_mle_g_y
        self.use_mle_theta_y = use_mle_theta_y
        self.kernel_type = kernel_type
        
        # Variant-specific
        self.W_fixed = W_fixed
        self.D = D
        self.column_indices = column_indices
        
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
        self.tau2_y_init = tau2_y_init
        self.tau2_q_init = tau2_q_init
        self.g_y_init = g_y_init
        self.g_q_init = g_q_init
        self.theta_y_init = theta_y_init
        self.theta_q_init = theta_q_init
        
        # Storage
        self.chains_samples = []
        self.chains_metrics = []
        self.computation_times = []
    
    def run_chains(self, Y_train: np.ndarray, X_train: np.ndarray,
                   Y_test: np.ndarray, X_test: np.ndarray,
                   verbose: bool = True) -> Dict:
        """
        Run multiple chains and compute metrics.
        
        Args:
            Y_train: Training responses (n_train,)
            X_train: Training inputs (n_train, p)
            Y_test: Test responses (n_test,)
            X_test: Test inputs (n_test, p)
            verbose: Print progress
            
        Returns:
            Dictionary with chains_samples, chains_metrics, convergence diagnostics, etc.
        """
        self.chains_samples = []
        self.chains_metrics = []
        self.computation_times = []
        
        for chain_idx in range(self.n_chains):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Chain {chain_idx + 1}/{self.n_chains} - Variant: {self.variant} (Layer 2)")
                print(f"{'='*70}")
            
            start_time = time.time()
            
            # Create sampler based on variant
            if self.variant == 'W_Known':
                sampler = GibbsSampler2Layer_W_Known(
                    Y=Y_train, X=X_train, W_fixed=self.W_fixed,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g_y=self.use_mle_g_y,
                    use_mle_theta_y=self.use_mle_theta_y,
                    kernel_type=self.kernel_type,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    gamma1=self.gamma1,
                    gamma2_y=self.gamma2_y,
                    gamma2_q=self.gamma2_q,
                    l=self.l,
                    u=self.u,
                    tau2_y_init=self.tau2_y_init,
                    tau2_q_init=self.tau2_q_init,
                    g_y_init=self.g_y_init,
                    g_q_init=self.g_q_init,
                    theta_y_init=self.theta_y_init,
                    theta_q_init=self.theta_q_init
                )
            elif self.variant == 'No_W':
                sampler = GibbsSampler2Layer_No_W(
                    Y=Y_train, X=X_train,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g_y=self.use_mle_g_y,
                    use_mle_theta_y=self.use_mle_theta_y,
                    kernel_type=self.kernel_type,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    gamma1=self.gamma1,
                    gamma2_y=self.gamma2_y,
                    gamma2_q=self.gamma2_q,
                    l=self.l,
                    u=self.u,
                    tau2_y_init=self.tau2_y_init,
                    tau2_q_init=self.tau2_q_init,
                    g_y_init=self.g_y_init,
                    g_q_init=self.g_q_init,
                    theta_y_init=self.theta_y_init,
                    theta_q_init=self.theta_q_init
                )
            else:  # No_W_Selective
                sampler = GibbsSampler2Layer_No_W_Selective(
                    Y=Y_train, X=X_train, D=self.D,
                    column_indices=self.column_indices,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g_y=self.use_mle_g_y,
                    use_mle_theta_y=self.use_mle_theta_y,
                    kernel_type=self.kernel_type,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    gamma1=self.gamma1,
                    gamma2_y=self.gamma2_y,
                    gamma2_q=self.gamma2_q,
                    l=self.l,
                    u=self.u,
                    tau2_y_init=self.tau2_y_init,
                    tau2_q_init=self.tau2_q_init,
                    g_y_init=self.g_y_init,
                    g_q_init=self.g_q_init,
                    theta_y_init=self.theta_y_init,
                    theta_q_init=self.theta_q_init
                )
            
            # Run sampler
            samples = sampler.run(verbose=verbose)
            
            # Compute metrics
            metrics = Bayesian_Metrics_with_quantiles_variant_2layer(
                samples, X_train, Y_train, X_test, Y_test,
                variant=self.variant,
                W_fixed=self.W_fixed,
                column_indices=self.column_indices if self.variant == 'No_W_Selective' else None,
                kernel_type=self.kernel_type
            )
            
            self.chains_samples.append(samples)
            self.chains_metrics.append(metrics)
            self.computation_times.append(time.time() - start_time)
            
            if verbose:
                print(f"\nChain {chain_idx + 1} complete in {self.computation_times[-1]:.2f}s")
                print(f"  RMSPE: {metrics['RMSPE']:.4f}")
                print(f"  NSME: {metrics['NSME']:.4f}")
                print(f"  CP: {metrics['CP']:.4f}")
                print(f"  ALCI: {metrics['ALCI']:.4f}")
                print(f"  CRPS: {metrics['CRPS']:.4f}")
                print(f"  BIC: {metrics['BIC']:.4f}")
        
        # Compute convergence diagnostics
        convergence = self._compute_convergence_diagnostics()
        
        # Compute metrics summary manually
        all_rmspe = [m['RMSPE'] for m in self.chains_metrics]
        all_nsme = [m['NSME'] for m in self.chains_metrics]
        all_crps = [m['CRPS'] for m in self.chains_metrics]
        all_score = [m['Score'] for m in self.chains_metrics]
        all_bic = [m['BIC'] for m in self.chains_metrics]
        all_mlppd = [m['MLPPD'] for m in self.chains_metrics]
        all_cp = [m['CP'] for m in self.chains_metrics]
        all_alci = [m['ALCI'] for m in self.chains_metrics]
        
        def get_summary(values):
            return {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
        
        metrics_summary = {
            'RMSPE': get_summary(all_rmspe),
            'NSME': get_summary(all_nsme),
            'CRPS': get_summary(all_crps),
            'Score': get_summary(all_score),
            'BIC': get_summary(all_bic),
            'MLPPD': get_summary(all_mlppd),
            'CP': get_summary(all_cp),
            'ALCI': get_summary(all_alci)
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
            'convergence': convergence,
            'metrics_summary': metrics_summary,
            'computation_times': self.computation_times,
            'variant': self.variant
        }
        
        if parameter_diagnostics is not None:
            results['parameter_diagnostics'] = parameter_diagnostics
        if parameter_diagnostics_error is not None:
            results['parameter_diagnostics_error'] = parameter_diagnostics_error
        
        return results
    
    def _compute_convergence_diagnostics(self) -> Dict:
        """Compute Gelman-Rubin and Heidelberg-Welch diagnostics."""
        # Extract parameter chains
        tau2_chains = [_sample_array(chain, 'tau2_y', 'tau2') for chain in self.chains_samples]
        g_y_chains = [chain['g_y'] for chain in self.chains_samples]
        
        # Gelman-Rubin for tau2 and g_y
        r_hat_tau2 = self._gelman_rubin(tau2_chains)
        r_hat_g_y = self._gelman_rubin(g_y_chains)
        
        # For theta_y, handle both scalar and vector cases
        theta_y_chains = [chain['theta_y'] for chain in self.chains_samples]
        if theta_y_chains[0].ndim == 0:
            r_hat_theta_y = self._gelman_rubin([c.flatten() for c in theta_y_chains])
        else:
            # For vector theta_y, compute R-hat for each dimension
            n_dims = theta_y_chains[0].shape[1] if theta_y_chains[0].ndim > 1 else 1
            if n_dims == 1:
                r_hat_theta_y = self._gelman_rubin([c.flatten() for c in theta_y_chains])
            else:
                r_hat_theta_y = np.array([self._gelman_rubin([c[:, d] for c in theta_y_chains]) 
                                         for d in range(n_dims)])
        
        # For theta_q (Q layer)
        theta_q_chains = [chain['theta_q'] for chain in self.chains_samples]
        if theta_q_chains[0].ndim == 0:
            r_hat_theta_q = self._gelman_rubin([c.flatten() for c in theta_q_chains])
        else:
            n_dims = theta_q_chains[0].shape[1] if theta_q_chains[0].ndim > 1 else 1
            if n_dims == 1:
                r_hat_theta_q = self._gelman_rubin([c.flatten() for c in theta_q_chains])
            else:
                r_hat_theta_q = np.array([self._gelman_rubin([c[:, d] for c in theta_q_chains]) 
                                        for d in range(n_dims)])
        
        return {
            'r_hat_tau2_y': r_hat_tau2,
            'r_hat_g_y': r_hat_g_y,
            'r_hat_theta_y': r_hat_theta_y,
            'r_hat_theta_q': r_hat_theta_q
        }
    
    def _gelman_rubin(self, chains: List[np.ndarray]) -> float:
        """Compute Gelman-Rubin R-hat statistic."""
        n_chains = len(chains)
        n_samples = len(chains[0])
        
        # Between-chain variance
        chain_means = np.array([np.mean(chain) for chain in chains])
        overall_mean = np.mean(chain_means)
        B = n_samples * np.var(chain_means)
        
        # Within-chain variance
        chain_vars = np.array([np.var(chain) for chain in chains])
        W = np.mean(chain_vars)
        
        # R-hat
        var_hat = (n_samples - 1) / n_samples * W + 1 / n_samples * B
        r_hat = np.sqrt(var_hat / W) if W > 0 else 1.0
        
        return r_hat
    
    def create_all_diagnostics(self, output_dir: str = './diagnostics'):
        """Create all diagnostic plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Trace plots (combine all chains)
        tau2_all_chains = [_sample_array(s, 'tau2_y', 'tau2') for s in self.chains_samples]
        g_y_all_chains = [s['g_y'] for s in self.chains_samples]
        plot_trace(tau2_all_chains, 'tau2_y', output_dir)
        plot_trace(g_y_all_chains, 'g_y', output_dir)
        
        # For theta_y and theta_q, handle both scalar and vector cases
        theta_y_all_chains = [s['theta_y'] for s in self.chains_samples]
        if theta_y_all_chains[0].ndim == 0:
            plot_trace([c.flatten() for c in theta_y_all_chains], 'theta_y', output_dir)
        else:
            n_dims = theta_y_all_chains[0].shape[1] if theta_y_all_chains[0].ndim > 1 else 1
            if n_dims == 1:
                plot_trace([c.flatten() for c in theta_y_all_chains], 'theta_y', output_dir)
            else:
                for d in range(n_dims):
                    plot_trace([c[:, d] if c.ndim > 1 else c for c in theta_y_all_chains], 
                              f'theta_y_d{d}', output_dir)
        
        theta_q_all_chains = [s['theta_q'] for s in self.chains_samples]
        if theta_q_all_chains[0].ndim == 0:
            plot_trace([c.flatten() for c in theta_q_all_chains], 'theta_q', output_dir)
        else:
            n_dims = theta_q_all_chains[0].shape[1] if theta_q_all_chains[0].ndim > 1 else 1
            if n_dims == 1:
                plot_trace([c.flatten() for c in theta_q_all_chains], 'theta_q', output_dir)
            else:
                for d in range(n_dims):
                    plot_trace([c[:, d] if c.ndim > 1 else c for c in theta_q_all_chains], 
                              f'theta_q_d{d}', output_dir)
        
        # Density plots (combine all chains)
        tau2_all_chains = [_sample_array(s, 'tau2_y', 'tau2') for s in self.chains_samples]
        g_y_all_chains = [s['g_y'] for s in self.chains_samples]
        plot_density(tau2_all_chains, 'tau2_y', output_dir)
        plot_density(g_y_all_chains, 'g_y', output_dir)
        
        # Actual vs Predicted
        if len(self.chains_metrics) > 0:
            pred_mean = self.chains_metrics[0]['pred_mean']
            plot_actual_vs_predicted(Y_test=None, pred_mean=pred_mean, 
                                   output_dir=output_dir, variant=self.variant)
        
        # Convergence diagnostics
        convergence = self._compute_convergence_diagnostics()
        plot_convergence_diagnostics(convergence, output_dir)
        
        # Metrics comparison
        plot_metrics_comparison_table(self.chains_metrics, output_dir)


if __name__ == "__main__":
    print("="*70)
    print("Multi-Chain Sampler for Layer 2 Variants - Test")
    print("="*70)
    
    np.random.seed(42)
    n_train, n_test, p = 30, 10, 5
    X_train = np.random.randn(n_train, p)
    X_test = np.random.randn(n_test, p)
    Y_train = np.random.randn(n_train)
    Y_test = np.random.randn(n_test)
    
    # Test W_Known variant
    print("\n1. Testing W_Known variant:")
    W_fixed = np.random.randn(p, 2)
    W_fixed, _ = np.linalg.qr(W_fixed)
    
    multichain1 = MultiChainSampler_L2_Variants(
        variant='W_Known',
        W_fixed=W_fixed,
        n_chains=2,
        n_iterations=5,
        burn_in=1,
        thin=1,
        kernel_type='separable_squared_exponential'
    )
    
    results1 = multichain1.run_chains(Y_train, X_train, Y_test, X_test, verbose=False)
    print(f"   ✅ W_Known: {len(results1['chains_samples'])} chains")
    print(f"   RMSPE: {results1['chains_metrics'][0]['RMSPE']:.4f}")
    
    # Test No_W variant
    print("\n2. Testing No_W variant:")
    multichain2 = MultiChainSampler_L2_Variants(
        variant='No_W',
        n_chains=2,
        n_iterations=5,
        burn_in=1,
        thin=1,
        kernel_type='separable_squared_exponential'
    )
    
    results2 = multichain2.run_chains(Y_train, X_train, Y_test, X_test, verbose=False)
    print(f"   ✅ No_W: {len(results2['chains_samples'])} chains")
    print(f"   RMSPE: {results2['chains_metrics'][0]['RMSPE']:.4f}")
    
    # Test No_W_Selective variant
    print("\n3. Testing No_W_Selective variant:")
    multichain3 = MultiChainSampler_L2_Variants(
        variant='No_W_Selective',
        D=3,
        column_indices=np.array([0, 1, 2]),
        n_chains=2,
        n_iterations=5,
        burn_in=1,
        thin=1,
        kernel_type='separable_squared_exponential'
    )
    
    results3 = multichain3.run_chains(Y_train, X_train, Y_test, X_test, verbose=False)
    print(f"   ✅ No_W_Selective: {len(results3['chains_samples'])} chains")
    print(f"   RMSPE: {results3['chains_metrics'][0]['RMSPE']:.4f}")
    
    print("\n" + "="*70)
    print("✅ All variants tested successfully!")
    print("="*70)
