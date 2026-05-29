"""
BDR Metrics - Performance Metrics for Bayesian Dimensionality Reduction

This module contains all performance metrics used to evaluate GP models:
    - RMSPE: Root Mean Square Predictive Error
    - NSME: Nash-Sutcliffe Model Efficiency
    - CRPS: Continuous Ranked Probability Score
    - Score: Predictive Log-Likelihood Score
    - BIC: Bayesian Information Criterion
    - MLPPD: Mean Log Pointwise Predictive Density
    - CP: Empirical coverage probability of predictive 95% intervals
    - ALCI: Average length of predictive 95% credible intervals
"""

import numpy as np
from scipy.stats import norm
from typing import Optional
import os


def compute_RMSPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Predictive Error.
    
    Formula: RMSPE = sqrt(mean((y_true - y_pred)²))
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSPE value
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))


def compute_NSME(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Model Efficiency.
    
    Formula: NSME = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        NSME value (1 = perfect, 0 = as good as mean, <0 = worse than mean)
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot


def compute_CRPS(y_true: np.ndarray, y_pred_mean: np.ndarray, 
                y_pred_std: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (for Gaussian predictive distributions).
    
    Formula: CRPS = (1/n) Σ σ_i [1/√π - 2φ(z_i) - z_i(2Φ(z_i) - 1)]
    where z_i = (y_i - ŷ_i) / σ_i
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        y_pred_std: Predicted standard deviations
        
    Returns:
        CRPS value (lower is better)
    """
    z = (y_true - y_pred_mean) / (y_pred_std + 1e-10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    crps_values = y_pred_std * (1/np.sqrt(np.pi) - 2*phi_z - z*(2*Phi_z - 1))
    return np.mean(crps_values)


def compute_score(y_true: np.ndarray, y_pred_mean: np.ndarray, 
                 Sigma_pred: np.ndarray) -> float:
    """
    Predictive Log-Likelihood Score.
    
    Formula: Score = -log|Σ| - (y - ŷ)^T Σ^{-1} (y - ŷ)
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        Sigma_pred: Predictive covariance matrix
        
    Returns:
        Score value (higher is better)
    """
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
    """
    Bayesian Information Criterion.
    
    Formula: BIC = log L(θ*; Z, Y) - 0.5 * #params * log(n_train)
    
    For multi-layer models, log_likelihood should be the SUM across all layers.
    
    Args:
        log_likelihood: Sum of log-likelihoods across all layers
        n_params: Total number of parameters
        n_train: Number of training observations
        
    Returns:
        BIC value (higher is better)
    """
    return log_likelihood - 0.5 * n_params * np.log(n_train)


def compute_MLPPD(y_true: np.ndarray, y_pred_mean: np.ndarray, 
                 y_pred_var: np.ndarray) -> float:
    """
    Mean Log Pointwise Predictive Density.
    
    Formula: MLPPD = (1/n) Σ [-0.5*log(2πσ²_i) - (y_i - ŷ_i)² / (2σ²_i)]
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        y_pred_var: Predicted variances
        
    Returns:
        MLPPD value (higher is better)
    """
    log_lik_points = -0.5 * np.log(2 * np.pi * y_pred_var) - \
                     (y_true - y_pred_mean)**2 / (2 * y_pred_var)
    return np.mean(log_lik_points)


def compute_CP(y_true: np.ndarray, lower_bound: np.ndarray,
               upper_bound: np.ndarray) -> float:
    """
    Empirical coverage probability (CP) for predictive credible intervals.

    Formula:
        CP = (1 / n_test) * sum( I(y_i in [l_i, u_i]) )
    
    Args:
        y_true: True values
        lower_bound: Lower predictive bounds (l_i)
        upper_bound: Upper predictive bounds (u_i)
        
    Returns:
        Coverage probability (proportion of points within intervals)
    """
    within = (lower_bound <= y_true) & (y_true <= upper_bound)
    return float(np.mean(within))


def compute_ALCI(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """
    Average length of credible intervals (ALCI).

    Formula:
        ALCI = (1 / n_test) * sum( u_i - l_i )
    
    Args:
        lower_bound: Lower bounds
        upper_bound: Upper bounds
        
    Returns:
        Average interval length
    """
    return float(np.mean(upper_bound - lower_bound))


def compute_coverage_probability(y_true: np.ndarray, lower_bound: np.ndarray,
                                 upper_bound: np.ndarray) -> float:
    """
    Backward-compatible alias for empirical coverage probability (CP).
    """
    return compute_CP(y_true, lower_bound, upper_bound)


def compute_interval_length(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """
    Backward-compatible alias for average length of credible intervals (ALCI).
    """
    return compute_ALCI(lower_bound, upper_bound)


def compute_all_metrics_summary(y_true: np.ndarray, y_pred_samples: np.ndarray,
                                BIC_samples: np.ndarray) -> dict:
    """
    Compute summary statistics (mean, median, std, credible intervals) for all metrics.
    
    Args:
        y_true: True test values
        y_pred_samples: Posterior predictive samples (n_test, n_samples)
        BIC_samples: BIC values per MCMC sample (n_samples,)
        
    Returns:
        Dictionary with summary statistics for all metrics
    """
    # Predictions
    y_pred_mean = np.mean(y_pred_samples, axis=1)
    y_pred_std = np.std(y_pred_samples, axis=1)
    
    # Compute metrics per sample
    n_samples = y_pred_samples.shape[1]
    rmspe_samples = np.array([compute_RMSPE(y_true, y_pred_samples[:, i]) for i in range(n_samples)])
    nsme_samples = np.array([compute_NSME(y_true, y_pred_samples[:, i]) for i in range(n_samples)])
    crps_samples = np.array([compute_CRPS(y_true, y_pred_samples[:, i], y_pred_std) for i in range(n_samples)])
    score_samples = np.array([
        compute_score(y_true, y_pred_samples[:, i], np.diag(np.maximum(y_pred_std**2, 1e-12)))
        for i in range(n_samples)
    ])
    
    # Compute MLPPD per sample
    mlppd_samples = np.array([compute_MLPPD(y_true, y_pred_samples[:, i], y_pred_std**2) for i in range(n_samples)])
    
    # Predictive 95% credible interval bounds for CP/ALCI
    lower_bound = np.percentile(y_pred_samples, 2.5, axis=1)
    upper_bound = np.percentile(y_pred_samples, 97.5, axis=1)
    cp = compute_CP(y_true, lower_bound, upper_bound)
    alci = compute_ALCI(lower_bound, upper_bound)

    # Summary statistics
    def get_summary(samples):
        return {
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'ci_lower': np.percentile(samples, 2.5),
            'ci_upper': np.percentile(samples, 97.5)
        }

    def get_fixed_summary(value: float):
        return {
            'mean': value,
            'median': value,
            'std': 0.0,
            'ci_lower': value,
            'ci_upper': value
        }
    
    return {
        'rmspe': get_summary(rmspe_samples),
        'nsme': get_summary(nsme_samples),
        'crps': get_summary(crps_samples),
        'score': get_summary(score_samples),
        'bic': get_summary(BIC_samples),
        'mlppd': get_summary(mlppd_samples),
        'cp': get_fixed_summary(cp),
        'alci': get_fixed_summary(alci)
    }


def compute_iteration_metrics(
    y_true: np.ndarray,
    y_pred_samples: np.ndarray,
    y_pred_var_samples: np.ndarray,
    sample_axis: int = 1,
    bic_samples: Optional[np.ndarray] = None
) -> dict:
    """
    Compute ALL metrics for every MCMC iteration/sample.

    Args:
        y_true: True test values (n_test,)
        y_pred_samples: Predictive means from each iteration
            - shape (n_test, n_samples) if sample_axis=1
            - shape (n_samples, n_test) if sample_axis=0
        y_pred_var_samples: Predictive variances from each iteration (same shape as y_pred_samples)
        sample_axis: Axis indexing posterior samples (0 or 1)
        bic_samples: Optional BIC values per iteration (n_samples,)

    Returns:
        Dictionary with per-iteration arrays for all metrics
    """
    if sample_axis not in (0, 1):
        raise ValueError("sample_axis must be 0 or 1")
    
    if y_pred_samples.shape != y_pred_var_samples.shape:
        raise ValueError("y_pred_samples and y_pred_var_samples must have the same shape")
    
    # Standardize to (n_samples, n_test)
    if sample_axis == 1:
        pred_means = y_pred_samples.T
        pred_vars = y_pred_var_samples.T
    else:
        pred_means = y_pred_samples
        pred_vars = y_pred_var_samples
    
    n_samples = pred_means.shape[0]
    z_975 = norm.ppf(0.975)  # ~1.96
    
    rmspe_samples = np.zeros(n_samples)
    nsme_samples = np.zeros(n_samples)
    crps_samples = np.zeros(n_samples)
    score_samples = np.zeros(n_samples)
    mlppd_samples = np.zeros(n_samples)
    cp_samples = np.zeros(n_samples)
    alci_samples = np.zeros(n_samples)
    
    for i in range(n_samples):
        mean_i = pred_means[i]
        var_i = np.maximum(pred_vars[i], 1e-12)
        std_i = np.sqrt(var_i)
        
        lower_i = mean_i - z_975 * std_i
        upper_i = mean_i + z_975 * std_i
        
        rmspe_samples[i] = compute_RMSPE(y_true, mean_i)
        nsme_samples[i] = compute_NSME(y_true, mean_i)
        crps_samples[i] = compute_CRPS(y_true, mean_i, std_i)
        mlppd_samples[i] = compute_MLPPD(y_true, mean_i, var_i)
        cp_samples[i] = compute_CP(y_true, lower_i, upper_i)
        alci_samples[i] = compute_ALCI(lower_i, upper_i)
        score_samples[i] = compute_score(y_true, mean_i, np.diag(var_i))
    
    metrics = {
        'rmspe_samples': rmspe_samples,
        'nsme_samples': nsme_samples,
        'crps_samples': crps_samples,
        'score_samples': score_samples,
        'mlppd_samples': mlppd_samples,
        'cp_samples': cp_samples,
        'alci_samples': alci_samples
    }
    
    if bic_samples is not None:
        metrics['bic_samples'] = np.asarray(bic_samples)
    
    return metrics


def _get_r_coda_handles(r_home: Optional[str] = None):
    """Load R/coda via rpy2."""
    if r_home is not None:
        os.environ["R_HOME"] = r_home
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr, isinstalled
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:
        raise RuntimeError(
            "rpy2 could not load R. Make sure R is installed and R_HOME is correct."
        ) from exc
    
    if not isinstalled("coda"):
        utils = importr("utils")
        utils.install_packages("coda", repos="https://cloud.r-project.org")
    
    coda = importr("coda")
    return ro, coda, pandas2ri, localconverter


def _component_names(prefix: str, shape: tuple) -> list[str]:
    """Generate parameter component names for vector/tensor parameters."""
    if len(shape) == 0:
        return [prefix]
    
    names: list[str] = []
    for idx in np.ndindex(shape):
        idx1 = ",".join(str(i + 1) for i in idx)
        names.append(f"{prefix}[{idx1}]")
    return names


def _extract_parameter_array(
    chain: dict,
    parameter: str,
    burn: int = 500,
    use_projection_for_W: bool = False
) -> tuple[np.ndarray, list[str]]:
    """
    Extract parameter draws as a 2D array (n_draws, n_components).
    Special handling for W matrix chains with optional W W^T projection.
    """
    if parameter not in chain:
        raise KeyError(f"Parameter '{parameter}' not found in chain keys: {list(chain.keys())}")
    
    arr = np.asarray(chain[parameter], dtype=float)
    if arr.ndim < 1:
        raise ValueError(f"Expected at least 1D array for '{parameter}', got shape {arr.shape}")
    
    arr = arr[burn:]
    if arr.shape[0] == 0:
        raise ValueError(f"No draws left for '{parameter}' after burn={burn}")
    
    if parameter == "W" and arr.ndim == 3:
        n, p, d = arr.shape
        rows = []
        
        if use_projection_for_W:
            for W in arr:
                rows.append((W @ W.T).ravel())
            X = np.vstack(rows)
            names = [f"WWT[{i+1},{j+1}]" for i in range(p) for j in range(p)]
        else:
            for W in arr:
                rows.append(W.ravel())
            X = np.vstack(rows)
            names = [f"W[{i+1},{j+1}]" for i in range(p) for j in range(d)]
        
        return X, names
    
    n = arr.shape[0]
    tail_shape = tuple(arr.shape[1:])
    X = arr.reshape(n, -1)
    names = _component_names(parameter, tail_shape)
    return X, names


def _parameter_diagnostics_from_arrays(
    arrays: list[np.ndarray],
    param_names: list[str],
    ci: float,
    ro,
    coda,
    pandas2ri,
    localconverter
):
    """Compute summary, Heidelberger-Welch/ESS, and R-hat from extracted arrays."""
    import pandas as pd
    
    alpha = 1.0 - ci
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    
    summary_rows = []
    for chain_id, X in enumerate(arrays, start=1):
        for j, nm in enumerate(param_names):
            x = X[:, j]
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            
            summary_rows.append({
                "chain": chain_id,
                "parameter": nm,
                "median": float(np.median(x)),
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                f"ci_{100*q_low:.1f}%": float(np.quantile(x, q_low)),
                f"ci_{100*q_high:.1f}%": float(np.quantile(x, q_high)),
                "n": int(x.size)
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    heidel_rows = []
    for chain_id, X in enumerate(arrays, start=1):
        for j, nm in enumerate(param_names):
            x = X[:, j]
            x = x[np.isfinite(x)]
            if x.size < 5:
                continue
            
            r_vec = ro.FloatVector(x.tolist())
            mcmc_obj = coda.mcmc(r_vec, start=1)
            
            out = coda.heidel_diag(mcmc_obj)
            ess = coda.effectiveSize(mcmc_obj)
            
            with localconverter(ro.default_converter + pandas2ri.converter):
                df = ro.conversion.rpy2py(out)
                ess_py = ro.conversion.rpy2py(ess)
            
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            
            if df.shape[1] >= 6:
                df = df.iloc[:, :6].copy()
            else:
                vals = df.to_numpy().ravel()
                vals = np.pad(vals, (0, max(0, 6 - vals.size)), constant_values=np.nan)[:6]
                df = pd.DataFrame([vals])
            
            df.columns = ["stest", "start", "pvalue", "htest", "mean", "halfwidth"]
            df["stest"] = df["stest"].map({1.0: "passed", 0.0: "failed"}).fillna(df["stest"])
            df["htest"] = df["htest"].map({1.0: "passed", 0.0: "failed"}).fillna(df["htest"])
            df["ess"] = float(np.asarray(ess_py).squeeze())
            
            df.insert(0, "parameter", nm)
            df.insert(0, "chain", chain_id)
            heidel_rows.append(df)
    
    heidel_df = pd.concat(heidel_rows, ignore_index=True) if heidel_rows else pd.DataFrame()
    
    n = min(a.shape[0] for a in arrays)
    arrays_trunc = [a[:n, :] for a in arrays]
    m = len(arrays_trunc)
    
    if m < 2:
        raise ValueError("Need at least 2 chains for R-hat.")
    
    chain_means = np.vstack([a.mean(axis=0) for a in arrays_trunc])
    chain_vars = np.vstack([a.var(axis=0, ddof=1) for a in arrays_trunc])
    
    W_within = chain_vars.mean(axis=0)
    mean_of_means = chain_means.mean(axis=0)
    B = (n / (m - 1)) * ((chain_means - mean_of_means) ** 2).sum(axis=0)
    var_hat = ((n - 1) / n) * W_within + (1 / n) * B
    
    rhat = np.full_like(W_within, np.nan, dtype=float)
    ok = W_within > 0
    rhat[ok] = np.sqrt(var_hat[ok] / W_within[ok])
    
    rhat_df = pd.DataFrame({"parameter": param_names, "rhat": rhat})
    
    return {"summary": summary_df, "heidel": heidel_df, "rhat": rhat_df}


def parameter_diagnostics(
    chains: list[dict],
    parameter: str,
    burn: int = 500,
    ci: float = 0.95,
    r_home: Optional[str] = None,
    use_projection_for_W: bool = False
):
    """
    Full diagnostics for one parameter across chains (summary, Heidel/ESS, R-hat).
    """
    ro, coda, pandas2ri, localconverter = _get_r_coda_handles(r_home=r_home)
    extracted = [
        _extract_parameter_array(ch, parameter, burn=burn, use_projection_for_W=use_projection_for_W)
        for ch in chains
    ]
    
    arrays = [item[0] for item in extracted]
    param_names = extracted[0][1]
    return _parameter_diagnostics_from_arrays(arrays, param_names, ci, ro, coda, pandas2ri, localconverter)


def W_diagnostics(
    chains: list[dict],
    burn: int = 500,
    ci: float = 0.95,
    use_projection: bool = False,
    r_home: Optional[str] = None
):
    """
    Full diagnostics for W across chains.

    Returns a dictionary with:
      - 'summary': median, mean, sd, CI for each chain and W component
      - 'heidel': Heidelberger-Welch + ESS for each chain and component
      - 'rhat': Gelman-Rubin R-hat for each component
    """
    return parameter_diagnostics(
        chains=chains,
        parameter="W",
        burn=burn,
        ci=ci,
        r_home=r_home,
        use_projection_for_W=use_projection
    )


def compute_multichain_parameter_diagnostics(
    chains: list[dict],
    burn: int = 500,
    ci: float = 0.95,
    use_projection_for_W: bool = False,
    r_home: Optional[str] = None,
    parameters: Optional[list[str]] = None
):
    """
    Compute coda diagnostics for multiple parameters; W uses matrix-specific handling.
    """
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for multi-chain diagnostics.")
    
    if parameters is None:
        default_order = [
            "tau2_y", "g_y", "theta_D_y",
            "tau2", "g", "theta_D",
            "g_q", "g_r",
            "theta_y", "theta_q", "theta_r",
            "W"
        ]
        chain_keys = set(chains[0].keys())
        parameters = [p for p in default_order if p in chain_keys]
    
    if len(parameters) == 0:
        raise ValueError("No matching parameters found for diagnostics.")
    
    ro, coda, pandas2ri, localconverter = _get_r_coda_handles(r_home=r_home)
    diagnostics = {}
    
    for param in parameters:
        extracted = [
            _extract_parameter_array(
                ch,
                parameter=param,
                burn=burn,
                use_projection_for_W=(use_projection_for_W if param == "W" else False)
            )
            for ch in chains
        ]
        
        arrays = [item[0] for item in extracted]
        param_names = extracted[0][1]
        diagnostics[param] = _parameter_diagnostics_from_arrays(
            arrays, param_names, ci, ro, coda, pandas2ri, localconverter
        )
    
    return diagnostics
