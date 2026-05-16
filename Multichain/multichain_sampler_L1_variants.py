"""
Multi-Chain Gibbs Sampler for Layer 1 Variants

This module implements multi-chain MCMC sampling for Layer 1 variants:
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

from gibbs_sampler_layers_L1_variants import (  # type: ignore[import]
    GibbsSampler1Layer_W_Known,
    GibbsSampler1Layer_No_W,
    GibbsSampler1Layer_No_W_Selective
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
# Prediction Functions for Variants
# =============================================================================

def predict_gp_variant_W_Known(X_new: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray,
                               W_fixed: np.ndarray, theta_D_y: Union[float, np.ndarray], 
                               g_y: float, tau2_y: float, kernel_type: str = 'isotropic_squared_exponential') -> Tuple[np.ndarray, np.ndarray]:
    """
    GP prediction for W_Known variant.
    
    Args:
        X_new: New inputs (n_test, p)
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        W_fixed: Fixed projection matrix (p, D)
        theta_D_y: Y-layer lengthscale(s)
        g_y: Y-layer nugget
        tau2_y: Y-layer noise variance
        kernel_type: Kernel type
        
    Returns:
        (predictive_mean, predictive_variance)
    """
    from parameter_sampler_D1 import get_kernel_instance
    from parameter_sampler_Dgeneral import get_kernel_instance as get_kernel_instance_Dgen
    
    Z_train = X_train @ W_fixed
    Z_new = X_new @ W_fixed
    
    D = W_fixed.shape[1]
    
    # Create kernel instance
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    
    # Compute covariance matrices
    K_train = kernel.compute_covariance(Z_train, Z_train)
    K_new_train = kernel.compute_covariance(Z_new, Z_train)
    K_new = kernel.compute_covariance(Z_new, Z_new)
    
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


def predict_gp_variant_No_W(X_new: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray,
                            theta_D_y: Union[float, np.ndarray], g_y: float, tau2_y: float,
                            kernel_type: str = 'separable_squared_exponential') -> Tuple[np.ndarray, np.ndarray]:
    """
    GP prediction for No_W variant (X used directly).
    
    Args:
        X_new: New inputs (n_test, p)
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        theta_D_y: Y-layer lengthscale(s)
        g_y: Y-layer nugget
        tau2_y: Y-layer noise variance
        kernel_type: Kernel type
        
    Returns:
        (predictive_mean, predictive_variance)
    """
    D = X_train.shape[1]  # D = p
    
    # Create kernel instance
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    
    # Compute covariance matrices
    K_train = kernel.compute_covariance(X_train, X_train)
    K_new_train = kernel.compute_covariance(X_new, X_train)
    K_new = kernel.compute_covariance(X_new, X_new)
    
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


def predict_gp_variant_No_W_Selective(X_new: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray,
                                     X_selected: np.ndarray, X_new_selected: np.ndarray,
                                     theta_D_y: Union[float, np.ndarray], g_y: float, tau2_y: float,
                                     kernel_type: str = 'separable_squared_exponential') -> Tuple[np.ndarray, np.ndarray]:
    """
    GP prediction for No_W_Selective variant (selected columns of X).
    
    Args:
        X_new: New inputs (n_test, p)
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        X_selected: Selected columns of X_train (n_train, D)
        X_new_selected: Selected columns of X_new (n_test, D)
        theta_D_y: Y-layer lengthscale(s)
        g_y: Y-layer nugget
        tau2_y: Y-layer noise variance
        kernel_type: Kernel type
        
    Returns:
        (predictive_mean, predictive_variance)
    """
    D = X_selected.shape[1]
    
    # Create kernel instance
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    
    # Compute covariance matrices using selected columns
    K_train = kernel.compute_covariance(X_selected, X_selected)
    K_new_train = kernel.compute_covariance(X_new_selected, X_selected)
    K_new = kernel.compute_covariance(X_new_selected, X_new_selected)
    
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
# Log-Likelihood Computation for Variants
# =============================================================================

def compute_loglikelihood_variant(samples: Dict, X_train: np.ndarray, Y_train: np.ndarray,
                                  variant: str, W_fixed: Optional[np.ndarray] = None,
                                  column_indices: Optional[np.ndarray] = None,
                                  kernel_type: str = 'isotropic_squared_exponential') -> float:
    """
    Compute log-likelihood for Layer 1 variant models.
    
    Args:
        samples: Dictionary with tau2_y, g_y, theta_D_y samples
        X_train: Training inputs (n_train, p)
        Y_train: Training responses (n_train,)
        variant: 'W_Known', 'No_W', or 'No_W_Selective'
        W_fixed: Fixed W matrix (for W_Known variant)
        column_indices: Column indices (for No_W_Selective variant)
        kernel_type: Kernel type
        
    Returns:
        Log-likelihood value
    """
    
    # Get parameter values (use mean of samples)
    tau2_y_samples = _sample_array(samples, 'tau2_y', 'tau2')
    g_y_samples = _sample_array(samples, 'g_y', 'g')
    theta_D_y_samples = _sample_array(samples, 'theta_D_y', 'theta_D')
    tau2_y = np.mean(tau2_y_samples)
    g_y = np.mean(g_y_samples)
    theta_D_y = np.mean(theta_D_y_samples, axis=0) if theta_D_y_samples.ndim > 1 else np.mean(theta_D_y_samples)
    
    # Determine D
    if variant == 'W_Known':
        D = W_fixed.shape[1]
        Z = X_train @ W_fixed
    elif variant == 'No_W':
        D = X_train.shape[1]
        Z = X_train
    else:  # No_W_Selective
        D = len(column_indices) if column_indices is not None else X_train.shape[1]
        Z = X_train[:, column_indices] if column_indices is not None else X_train
    
    # Create kernel instance
    if D == 1:
        from parameter_sampler_D1 import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    else:
        from parameter_sampler_Dgeneral import get_kernel_instance
        kernel = get_kernel_instance(kernel_type, theta_D_y, g_y, tau2_y, D)
    
    # Compute log-likelihood
    try:
        loglik = kernel.log_likelihood(Y_train, Z)
        return loglik
    except:
        return -np.inf


# =============================================================================
# Bayesian Metrics for Variants
# =============================================================================

def Bayesian_Metrics_with_quantiles_variant(samples: Dict, X_train: np.ndarray, Y_train: np.ndarray,
                                           X_test: np.ndarray, Y_test: np.ndarray,
                                           variant: str,
                                           W_fixed: Optional[np.ndarray] = None,
                                           column_indices: Optional[np.ndarray] = None,
                                           kernel_type: str = 'isotropic_squared_exponential',
                                           quantiles: List[float] = [0.025, 0.975]) -> Dict:
    """
    Compute Bayesian metrics for Layer 1 variants.
    
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
    g_y_samples = _sample_array(samples, 'g_y', 'g')
    theta_D_y_samples = _sample_array(samples, 'theta_D_y', 'theta_D')
    n_samples = len(tau2_y_samples)
    n_test = len(Y_test)
    
    # Storage for predictions
    pred_means = np.zeros((n_samples, n_test))
    pred_vars = np.zeros((n_samples, n_test))
    bic_samples = np.zeros(n_samples)
    
    # Prepare inputs based on variant
    if variant == 'W_Known':
        X_train_input = X_train
        X_test_input = X_test
    elif variant == 'No_W':
        X_train_input = X_train
        X_test_input = X_test
    else:  # No_W_Selective
        X_train_input = X_train[:, column_indices] if column_indices is not None else X_train
        X_test_input = X_test[:, column_indices] if column_indices is not None else X_test
    
    # Compute predictions for each sample
    for i in range(n_samples):
        tau2_y = tau2_y_samples[i]
        g_y = g_y_samples[i]
        theta_D_y = theta_D_y_samples[i] if theta_D_y_samples.ndim == 1 else theta_D_y_samples[i, :]
        
        if variant == 'W_Known':
            pred_mean, pred_var = predict_gp_variant_W_Known(
                X_test_input, X_train_input, Y_train,
                W_fixed, theta_D_y, g_y, tau2_y, kernel_type
            )
        elif variant == 'No_W':
            pred_mean, pred_var = predict_gp_variant_No_W(
                X_test_input, X_train_input, Y_train,
                theta_D_y, g_y, tau2_y, kernel_type
            )
        else:  # No_W_Selective
            pred_mean, pred_var = predict_gp_variant_No_W_Selective(
                X_test, X_train, Y_train,
                X_train_input, X_test_input,
                theta_D_y, g_y, tau2_y, kernel_type
            )
        
        pred_means[i] = pred_mean
        pred_vars[i] = pred_var
        
        # Per-iteration BIC
        theta_sample = np.asarray(theta_D_y)
        theta_input = theta_sample if theta_sample.ndim > 0 else np.array([theta_sample])
        if theta_input.ndim > 1:
            theta_input = theta_input.flatten()
        
        sample_i = {
            'tau2_y': np.array([tau2_y]),
            'g_y': np.array([g_y]),
            'theta_D_y': theta_input if theta_input.ndim == 1 and theta_input.size == 1 else theta_input[np.newaxis, :]
        }
        loglik_i = compute_loglikelihood_variant(
            sample_i, X_train, Y_train, variant, W_fixed, column_indices, kernel_type
        )
        k = 3  # tau2_y, g_y, theta_D_y
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
    
    # For score, use covariance of predictions across samples (like original multichain)
    # Score expects a single covariance matrix (n_test, n_test)
    Sigma_pred = np.cov(pred_means.T)  # (n_test, n_test) - covariance across samples
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
# Multi-Chain Sampler for Layer 1 Variants
# =============================================================================

class MultiChainSampler_L1_Variants:
    """
    Multi-chain sampler for Layer 1 variants.
    
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
                 use_mle_g: bool = False,
                 use_mle_theta: bool = False,
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
                 gamma2: float = 3.9,
                 l: float = 1.0,
                 u: float = 2.0,
                 tau2_init: float = 0.005,
                 g_init: float = 0.00009,
                 theta_init: Union[float, np.ndarray] = 1.0):
        """
        Initialize multi-chain sampler for Layer 1 variants.
        
        Args:
            variant: 'W_Known', 'No_W', or 'No_W_Selective'
            n_chains: Number of chains
            n_iterations: Total iterations per chain
            burn_in: Burn-in period
            thin: Thinning interval
            use_mle_tau2: Use MLE for tau2
            use_mle_g: Use MLE for g
            use_mle_theta: Use MLE for theta
            kernel_type: Kernel type
            W_fixed: Fixed W matrix (required for W_Known)
            D: Number of columns to use (required for No_W_Selective)
            column_indices: Column indices to use (optional for No_W_Selective)
            alpha1, alpha2: Inverse Gamma prior for tau2
            beta1, beta2: Gamma prior for g
            gamma1, gamma2: Gamma prior for theta
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
        self.use_mle_g = use_mle_g
        self.use_mle_theta = use_mle_theta
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
        self.gamma2 = gamma2
        self.l = l
        self.u = u
        self.tau2_init = tau2_init
        self.g_init = g_init
        self.theta_init = theta_init
        
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
                print(f"Chain {chain_idx + 1}/{self.n_chains} - Variant: {self.variant}")
                print(f"{'='*70}")
            
            start_time = time.time()
            
            # Create sampler based on variant
            if self.variant == 'W_Known':
                sampler = GibbsSampler1Layer_W_Known(
                    Y=Y_train, X=X_train, W_fixed=self.W_fixed,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g=self.use_mle_g,
                    use_mle_theta=self.use_mle_theta,
                    kernel_type=self.kernel_type,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    gamma1=self.gamma1,
                    gamma2=self.gamma2,
                    l=self.l,
                    u=self.u,
                    tau2_init=self.tau2_init,
                    g_init=self.g_init,
                    theta_init=self.theta_init
                )
            elif self.variant == 'No_W':
                sampler = GibbsSampler1Layer_No_W(
                    Y=Y_train, X=X_train,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g=self.use_mle_g,
                    use_mle_theta=self.use_mle_theta,
                    kernel_type=self.kernel_type,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    gamma1=self.gamma1,
                    gamma2=self.gamma2,
                    l=self.l,
                    u=self.u,
                    tau2_init=self.tau2_init,
                    g_init=self.g_init,
                    theta_init=self.theta_init
                )
            else:  # No_W_Selective
                sampler = GibbsSampler1Layer_No_W_Selective(
                    Y=Y_train, X=X_train, D=self.D,
                    column_indices=self.column_indices,
                    n_iterations=self.n_iterations,
                    burn_in=self.burn_in,
                    thin=self.thin,
                    use_mle_tau2=self.use_mle_tau2,
                    use_mle_g=self.use_mle_g,
                    use_mle_theta=self.use_mle_theta,
                    kernel_type=self.kernel_type,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    gamma1=self.gamma1,
                    gamma2=self.gamma2,
                    l=self.l,
                    u=self.u,
                    tau2_init=self.tau2_init,
                    g_init=self.g_init,
                    theta_init=self.theta_init
                )
            
            # Run sampler
            samples = sampler.run(verbose=verbose)
            
            # Compute metrics
            metrics = Bayesian_Metrics_with_quantiles_variant(
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
        
        # Compute metrics summary manually (since we have different format)
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
        g_chains = [_sample_array(chain, 'g_y', 'g') for chain in self.chains_samples]
        
        # Gelman-Rubin for tau2 and g
        r_hat_tau2 = self._gelman_rubin(tau2_chains)
        r_hat_g = self._gelman_rubin(g_chains)
        
        # For theta_D, handle both scalar and vector cases
        theta_chains = [_sample_array(chain, 'theta_D_y', 'theta_D') for chain in self.chains_samples]
        if theta_chains[0].ndim == 0:
            r_hat_theta = self._gelman_rubin([c.flatten() for c in theta_chains])
        else:
            # For vector theta, compute R-hat for each dimension
            n_dims = theta_chains[0].shape[1] if theta_chains[0].ndim > 1 else 1
            if n_dims == 1:
                r_hat_theta = self._gelman_rubin([c.flatten() for c in theta_chains])
            else:
                r_hat_theta = np.array([self._gelman_rubin([c[:, d] for c in theta_chains]) 
                                       for d in range(n_dims)])
        
        return {
            'r_hat_tau2_y': r_hat_tau2,
            'r_hat_g': r_hat_g,
            'r_hat_theta': r_hat_theta
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
        
        # Trace plots
        for chain_idx, samples in enumerate(self.chains_samples):
            plot_trace(_sample_array(samples, 'tau2_y', 'tau2'), f'tau2_y_chain{chain_idx+1}', output_dir)
            plot_trace(_sample_array(samples, 'g_y', 'g'), f'g_y_chain{chain_idx+1}', output_dir)
            
            theta = _sample_array(samples, 'theta_D_y', 'theta_D')
            if theta.ndim == 0:
                plot_trace(theta.flatten(), f'theta_D_y_chain{chain_idx+1}', output_dir)
            else:
                for d in range(theta.shape[1] if theta.ndim > 1 else 1):
                    plot_trace(theta[:, d] if theta.ndim > 1 else theta, 
                              f'theta_D_y_d{d}_chain{chain_idx+1}', output_dir)
        
        # Density plots
        for chain_idx, samples in enumerate(self.chains_samples):
            plot_density(_sample_array(samples, 'tau2_y', 'tau2'), f'tau2_y_chain{chain_idx+1}', output_dir)
            plot_density(_sample_array(samples, 'g_y', 'g'), f'g_y_chain{chain_idx+1}', output_dir)
        
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
    print("Multi-Chain Sampler for Layer 1 Variants - Test")
    print("="*70)
    
    np.random.seed(42)
    n_train, n_test, p = 50, 20, 5
    X_train = np.random.randn(n_train, p)
    X_test = np.random.randn(n_test, p)
    Y_train = np.random.randn(n_train)
    Y_test = np.random.randn(n_test)
    
    # Test W_Known variant
    print("\n1. Testing W_Known variant:")
    W_fixed = np.random.randn(p, 2)
    W_fixed, _ = np.linalg.qr(W_fixed)
    
    multichain = MultiChainSampler_L1_Variants(
        variant='W_Known',
        W_fixed=W_fixed,
        n_chains=2,
        n_iterations=10,
        burn_in=2,
        thin=1,
        kernel_type='separable_squared_exponential'
    )
    
    results = multichain.run_chains(Y_train, X_train, Y_test, X_test, verbose=False)
    print(f"   ✅ W_Known: {len(results['chains_samples'])} chains")
    print(f"   RMSPE: {results['chains_metrics'][0]['RMSPE']:.4f}")
    
    # Test No_W variant
    print("\n2. Testing No_W variant:")
    multichain2 = MultiChainSampler_L1_Variants(
        variant='No_W',
        n_chains=2,
        n_iterations=10,
        burn_in=2,
        thin=1,
        kernel_type='separable_squared_exponential'
    )
    
    results2 = multichain2.run_chains(Y_train, X_train, Y_test, X_test, verbose=False)
    print(f"   ✅ No_W: {len(results2['chains_samples'])} chains")
    print(f"   RMSPE: {results2['chains_metrics'][0]['RMSPE']:.4f}")
    
    # Test No_W_Selective variant
    print("\n3. Testing No_W_Selective variant:")
    multichain3 = MultiChainSampler_L1_Variants(
        variant='No_W_Selective',
        D=3,
        column_indices=np.array([0, 1, 2]),
        n_chains=2,
        n_iterations=10,
        burn_in=2,
        thin=1,
        kernel_type='separable_squared_exponential'
    )
    
    results3 = multichain3.run_chains(Y_train, X_train, Y_test, X_test, verbose=False)
    print(f"   ✅ No_W_Selective: {len(results3['chains_samples'])} chains")
    print(f"   RMSPE: {results3['chains_metrics'][0]['RMSPE']:.4f}")
    
    print("\n" + "="*70)
    print("✅ All variants tested successfully!")
    print("="*70)
