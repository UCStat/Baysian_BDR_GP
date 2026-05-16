"""
Run Multi-Chain Gibbs Sampler - Complete Interface with All Parameters

This is the main entry point for running multi-chain MCMC sampling.
ALL hyperparameters are exposed for user configuration.
"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import Optional, Union, List, Dict
from scipy.linalg import svd

# Lambda prior hyperparameters (Gamma shape-rate)
LAMBDA_GAMMA_SHAPE = 5.0 / 2.0
LAMBDA_GAMMA_RATE = 10.0 / 3.0


def _draw_prior_lambda_diag(D: int) -> np.ndarray:
    """
    Draw diagonal Lambda prior values from Gamma(5/2, 10/3) and return diag matrix.
    Uses scipy/numpy shape-scale parameterization with scale = 1/rate.
    """
    draws = np.random.gamma(
        shape=LAMBDA_GAMMA_SHAPE,
        scale=1.0 / LAMBDA_GAMMA_RATE,
        size=D
    ).astype(float)
    return np.diag(draws)

# Add module directories to path
base_dir = Path(__file__).parent
for folder in ["Multichain", "Gibbs Sampling", "Parameter Sampler", "BDR Metrics and Plot", "Data Generation"]:
    folder_path = str(base_dir / folder)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)

from multichain_sampler_D1 import MultiChainSampler as MultiChainSampler_D1  # type: ignore[import]
from multichain_sampler_Dgeneral import MultiChainSampler as MultiChainSampler_Dgeneral  # type: ignore[import]
from multichain_sampler_L1_variants import MultiChainSampler_L1_Variants  # type: ignore[import]
from multichain_sampler_L2_variants import MultiChainSampler_L2_Variants  # type: ignore[import]
from multichain_sampler_L3_variants import MultiChainSampler_L3_Variants  # type: ignore[import]
from Data_generation import generate_case1_1d, generate_case1_2d  # type: ignore[import]
from parameter_sampler_D1 import rmf_matrixN, rmf_matrix  # type: ignore[import]
from parameter_sampler_Dgeneral import rmf_matrixN as rmf_matrixN_Dgeneral, rmf_matrix as rmf_matrix_Dgeneral  # type: ignore[import]

# Import Layer 1 variants
try:
    from gibbs_sampler_layers_L1_variants import (  # type: ignore[import]
        GibbsSampler1Layer_W_Known,
        GibbsSampler1Layer_No_W,
        GibbsSampler1Layer_No_W_Selective
    )
except ImportError:
    print("Warning: Could not import Layer 1 variant samplers")

# Import Layer 2 variants
try:
    from gibbs_sampler_layers_L2_variants import (  # type: ignore[import]
        GibbsSampler2Layer_W_Known,
        GibbsSampler2Layer_No_W,
        GibbsSampler2Layer_No_W_Selective
    )
except ImportError:
    print("Warning: Could not import Layer 2 variant samplers")

# Import Layer 3 variants
try:
    from gibbs_sampler_layers_L3_variants import (  # type: ignore[import]
        GibbsSampler3Layer_W_Known,
        GibbsSampler3Layer_No_W,
        GibbsSampler3Layer_No_W_Selective
    )
except ImportError:
    print("Warning: Could not import Layer 3 variant samplers")


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


def _component_names(prefix: str, shape: tuple) -> List[str]:
    """Generate parameter component names for vector/tensor parameters."""
    if len(shape) == 0:
        return [prefix]
    
    names: List[str] = []
    for idx in np.ndindex(shape):
        idx1 = ",".join(str(i + 1) for i in idx)
        names.append(f"{prefix}[{idx1}]")
    return names


def _extract_parameter_array(
    chain: dict,
    parameter: str,
    burn: int = 500,
    use_projection_for_W: bool = False
) -> tuple[np.ndarray, List[str]]:
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
    arrays: List[np.ndarray],
    param_names: List[str],
    ci: float,
    ro,
    coda,
    pandas2ri,
    localconverter
) -> Dict[str, "pd.DataFrame"]:
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
                # Fallback if conversion shape differs
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
    chains: List[dict],
    parameter: str,
    burn: int = 500,
    ci: float = 0.95,
    r_home: Optional[str] = None,
    use_projection_for_W: bool = False
) -> Dict[str, "pd.DataFrame"]:
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
    chains: List[dict],
    burn: int = 500,
    ci: float = 0.95,
    use_projection: bool = False,
    r_home: Optional[str] = None
) -> Dict[str, "pd.DataFrame"]:
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
    chains: List[dict],
    burn: int = 500,
    ci: float = 0.95,
    use_projection_for_W: bool = False,
    r_home: Optional[str] = None,
    parameters: Optional[List[str]] = None
) -> Dict[str, Dict[str, "pd.DataFrame"]]:
    """
    Compute coda diagnostics for multiple parameters; W uses matrix-specific handling.
    """
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for multi-chain diagnostics.")
    
    if parameters is None:
        # Core model parameters across all sampler variants/layers
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
    diagnostics: Dict[str, Dict[str, "pd.DataFrame"]] = {}
    
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


# =============================================================================
# HELPER FUNCTION FOR INITIALIZATION (D=1)
# =============================================================================

def initialize_M_Lambda_V_W_D1(p: int, D: int = 1, seed: Optional[int] = None) -> dict:
    """
    Initialize M, Lambda, V, W, and their priors for D=1 using SVD-based procedure.
    
    Args:
        p: Input dimension
        D: Reduced dimension (must be 1)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - M_init: Initial M matrix (p, D)
            - Lambda_init: Initial Lambda matrix (D, D)
            - V_init: Initial V matrix (D, D)
            - W_init: Initial W matrix (p, D)
            - prior_M: Prior for M (p, D)
            - prior_V: Prior for V (D, D)
            - prior_Lambda: Prior for Lambda (D, D)
    """
    if D != 1:
        raise ValueError("This function only supports D=1")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Generate random matrix
    F = np.random.randn(p, D)
    
    # Step 2: SVD
    Mm_init, Ll, V_init = svd(F, full_matrices=False)
    
    # Step 3: Extract M_init (for D=1, take first column)
    M_init = Mm_init[:, :D]  # Shape: (p, 1)
    
    # Step 4: Lambda_init from SVD singular values
    Lambda_init = np.diag(Ll)  # Shape: (1, 1)
    # Step 5: Compute priors
    prior_M = rmf_matrixN(M=M_init.reshape(-1, 1))  # Shape: (p, 1)
    prior_V = rmf_matrix(M=V_init)  # Shape: (1, 1)
    prior_Lambda = _draw_prior_lambda_diag(D)  # Shape: (1, 1), Gamma(5/2, 10/3)
    
    # Step 6: Compute W_init
    # W_init = rmf_matrixN(M=(M_init.reshape(-1, 1) @ Lambda_init) @ V_init)
    # For D=1: M_init is (p, 1), Lambda_init is (1, 1), V_init is (1, 1)
    W_init = rmf_matrixN(M=(M_init @ Lambda_init) @ V_init)  # Shape: (p, 1)
    
    return {
        'M_init': M_init,
        'Lambda_init': Lambda_init,
        'V_init': V_init,
        'W_init': W_init,
        'prior_M': prior_M,
        'prior_V': prior_V,
        'prior_Lambda': prior_Lambda
    }


# =============================================================================
# HELPER FUNCTION FOR INITIALIZATION (D>1)
# =============================================================================

def initialize_M_Lambda_V_W_Dgeneral(p: int, D: int, seed: Optional[int] = None) -> dict:
    """
    Initialize M, Lambda, V, W, and their priors for D>1 using SVD-based procedure.
    
    Args:
        p: Input dimension
        D: Reduced dimension (must be >1, e.g., 2, 3, 5)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - M_init: Initial M matrix (p, D)
            - Lambda_init: Initial Lambda matrix (D, D)
            - V_init: Initial V matrix (D, D)
            - W_init: Initial W matrix (p, D)
            - prior_M: Prior for M (p, D)
            - prior_V: Prior for V (D, D)
            - prior_Lambda: Prior for Lambda (D, D)
    """
    if D < 2:
        raise ValueError("This function only supports D>1. For D=1, use initialize_M_Lambda_V_W_D1")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: F = np.random.randn(p, D)
    F = np.random.randn(p, D)
    
    # Step 2: Mm_init, Ll, V_init = svd(F)
    Mm_init, Ll, V_init = svd(F, full_matrices=False)
    
    # Step 3: M_init = Mm_init[:, :D]
    M_init = Mm_init[:, :D]
    
    # Step 4: Lambda_init = np.diag(Ll)
    Lambda_init = np.diag(Ll)
    
    # Step 5: prior_M = rmf_matrixN(M=M_init)
    prior_M = rmf_matrixN_Dgeneral(M=M_init)
    
    # Step 6: prior_V = rmf_matrix(M=V_init)
    prior_V = rmf_matrix_Dgeneral(M=V_init)
    
    # Step 7: prior_Lambda ~ Gamma(5/2, 10/3) componentwise on the diagonal
    prior_Lambda = _draw_prior_lambda_diag(D)
    
    # Step 8: W_init = rmf_matrixN(M=(M_init @ Lambda_init) @ V_init.T)
    W_init = rmf_matrixN_Dgeneral(M=(M_init @ Lambda_init) @ V_init.T)
    
    return {
        'M_init': M_init,
        'Lambda_init': Lambda_init,
        'V_init': V_init,
        'W_init': W_init,
        'prior_M': prior_M,
        'prior_V': prior_V,
        'prior_Lambda': prior_Lambda
    }


# =============================================================================
# CONFIGURATION FUNCTIONS FOR DIFFERENT D AND LAYER COMBINATIONS
# =============================================================================

def get_default_config(D: int, layer: int, n_train: int, p: int) -> dict:
    """
    Get default configuration for specific D and layer combination.
    
    Args:
        D: Reduced dimension (1, 2, 3, 4, 5, ...)
        layer: Layer architecture (1, 2, or 3)
        n_train: Number of training samples
        p: Input dimension
        
    Returns:
        Dictionary with all configuration parameters
    """
    config = {
        # Model
        'D': D,
        'layer': layer,
        
        # MCMC defaults
        'n_chains': 3,
        'n_iterations': 2000,
        'burn_in': 500,
        'thin': 2,
        
        # Estimation
        'use_mle_tau2': False,
        'use_mle_g': False,
        'use_mle_theta': False,
        'use_mle_all': False,
        'use_tf_gradients': True if D > 1 else False,
        
        # Kernel selection (for D=1, Layer=1)
        'kernel_type': 'isotropic_squared_exponential',
        
        # HMC parameters
        'eps_hmc': 0.001,
        'T_step_hmc': 17,
        'M_hmc': 1,
        
        # Priors: tau2
        'alpha1_tau2': 1.0,
        'alpha2_tau2': 1000.0,
        
        # Priors: g (nugget)
        'beta1_g': 0.01,
        'beta2_g': 0.005,
        'l_g': 1.0,
        'u_g': 2.0,
        
        # Priors: theta (lengthscale) - Gamma(3/2, b)
        'gamma1_theta': 1.5,
        'l_theta': 1.0,
        'u_theta': 2.0,
        
        # Priors: Lambda
        'nu_lambda': None,
        'epsilon_lambda': 2.0,
        'max_iter_lambda': 1000,
        
        # Priors: M and V
        'prior_M': None,  # Default: zeros(p, D)
        'prior_V': None,  # Default: zeros(D, D)
        'mv_sampler': 'python',
        'rstiefel_rscol': None,
        
        # Initialization: common
        'W_init': None,  # Will be randomly initialized
        'tau2_y_init': 0.005,
    }
    
    # Layer-specific initialization and priors
    if layer == 1:
        config.update({
            'gamma2_theta': 3.9,  # b_theta = 3.9 for 1-layer
            'g_y_init': 0.00009,
            'theta_y_init': np.ones(D) if D > 1 else 1.0,
        })
    elif layer == 2:
        config.update({
            'gamma2_theta_y': 3.9,    # Outer layer: b_y = 3.9
            'gamma2_theta_q': 3.9/3,  # Inner layer: b_q = 3.9/3 = 1.3
            'tau2_q_init': 0.005,
            'g_y_init': 0.00009,
            'g_q_init': 0.00009,
            'theta_y_init': np.ones(D) if D > 1 else 1.0,
            'theta_q_init': np.ones(D) if D > 1 else 1.0,
            'Q_init': None,  # Will be randomly initialized
        })
    else:  # layer == 3
        config.update({
            'gamma2_theta_y': 3.9,    # Outer layer: b_y = 3.9
            'gamma2_theta_q': 3.9/3,  # Middle layer: b_q = 3.9/3 = 1.3
            'gamma2_theta_r': 3.9/6,  # Inner layer: b_r = 3.9/6 = 0.65
            'tau2_q_init': 0.005,
            'tau2_r_init': 0.005,
            'g_y_init': 0.00009,
            'g_q_init': 0.00009,
            'g_r_init': 0.00009,
            'theta_y_init': np.ones(D) if D > 1 else 1.0,
            'theta_q_init': np.ones(D) if D > 1 else 1.0,
            'theta_r_init': np.ones(D) if D > 1 else 1.0,
            'Q_init': None,
            'R_init': None,
        })
    
    return config


def create_config_D1_L1(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=1, Layer=1 (simplest case).
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 1, 'layer': 1,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g': False, 'use_mle_theta': False,
        'use_mle_all': False, 'use_tf_gradients': False,
        'kernel_type': 'isotropic_squared_exponential',
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5, 'gamma2_theta': 3.9, 'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'g_y_init': 0.00009, 'theta_y_init': 1.0,
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_D1(p=p, D=1, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D1_L2(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=1, Layer=2 (2-layer Deep GP).
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 1, 'layer': 2,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False,      # Individual MLE flag for tau2 (Y layer)
        'use_mle_g_y': False,       # Individual MLE flag for g_y (Y layer)
        'use_mle_theta_y': False,   # Individual MLE flag for theta_y (Y layer)
        'use_mle_all': False,       # Joint MLE for all (overrides individual flags)
        'use_tf_gradients': False,
        'kernel_type': 'isotropic_squared_exponential',  # Kernel selection
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9,     # Outer layer (Y): b_y = 3.9
        'gamma2_theta_q': 3.9/3,   # Inner layer (Q): b_q = 3.9/3 = 1.3
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009,
        'theta_y_init': 1.0, 'theta_q_init': 1.0,
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_D1(p=p, D=1, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D1_L3(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=1, Layer=3 (3-layer Deep GP).
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 1, 'layer': 3,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False,      # Individual MLE flag for tau2 (Y layer)
        'use_mle_g_y': False,       # Individual MLE flag for g_y (Y layer)
        'use_mle_theta_y': False,   # Individual MLE flag for theta_y (Y layer)
        'use_mle_all': False,       # Joint MLE for all (overrides individual flags)
        'use_tf_gradients': False,
        'kernel_type': 'isotropic_squared_exponential',  # Kernel selection
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9,     # Outer layer (Y): b_y = 3.9
        'gamma2_theta_q': 3.9/3,   # Middle layer (Q): b_q = 3.9/3 = 1.3
        'gamma2_theta_r': 3.9/6,   # Inner layer (R): b_r = 3.9/6 = 0.65
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'tau2_r_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009, 'g_r_init': 0.00009,
        'theta_y_init': 1.0, 'theta_q_init': 1.0, 'theta_r_init': 1.0,
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_D1(p=p, D=1, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D2_L1(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=2, Layer=1.
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 2, 'layer': 1,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g': False, 'use_mle_theta': False,
        'use_mle_g_y': False, 'use_mle_theta_y': False,
        'use_mle_all': False, 'use_tf_gradients': True,
        'kernel_type': 'separable_squared_exponential',  # TF recommended for D>1
        'kernel_type': 'separable_squared_exponential',
        'eps_hmc': 0.09, 'T_step_hmc': 15,
        'alpha1_tau2': 0.001, 'alpha2_tau2': 0.001,
        'beta1_g': 3/2, 'beta2_g': 3.9, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5, 'gamma2_theta': 3.9, 'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'g_y_init': 0.00009, 'theta_y_init': np.ones(2),
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_Dgeneral(p=p, D=2, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D2_L2(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=2, Layer=2.
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 2, 'layer': 2,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_all': False, 'use_tf_gradients': True,
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9, 'gamma2_theta_q': 3.9/3,
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009,
        'theta_y_init': np.ones(2), 'theta_q_init': np.ones(2),
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_Dgeneral(p=p, D=2, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D2_L3(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=2, Layer=3.
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 2, 'layer': 3,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_all': False, 'use_tf_gradients': True,
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9, 'gamma2_theta_q': 3.9/3, 'gamma2_theta_r': 3.9/6,
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'tau2_r_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009, 'g_r_init': 0.00009,
        'theta_y_init': np.ones(2), 'theta_q_init': np.ones(2), 'theta_r_init': np.ones(2),
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_Dgeneral(p=p, D=2, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D3_L1(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=3, Layer=1.
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 3, 'layer': 1,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g': False, 'use_mle_theta': False,
        'use_mle_g_y': False, 'use_mle_theta_y': False,
        'use_mle_all': False, 'use_tf_gradients': True,
        'kernel_type': 'separable_squared_exponential',
        'kernel_type': 'separable_squared_exponential',
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5, 'gamma2_theta': 3.9, 'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'g_y_init': 0.00009, 'theta_y_init': np.ones(3),
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_Dgeneral(p=p, D=3, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D3_L2(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=3, Layer=2.
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 3, 'layer': 2,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_all': False, 'use_tf_gradients': True,
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9, 'gamma2_theta_q': 3.9/3,
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009,
        'theta_y_init': np.ones(3), 'theta_q_init': np.ones(3),
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_Dgeneral(p=p, D=3, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D3_L3(p: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> dict:
    """
    Configuration for D=3, Layer=3.
    
    Args:
        p: Input dimension. If provided, will initialize M, Lambda, V, W using SVD procedure.
        seed: Random seed for initialization (if p is provided).
        **kwargs: Additional configuration overrides.
    """
    base = {
        'D': 3, 'layer': 3,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_all': False, 'use_tf_gradients': True,
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9, 'gamma2_theta_q': 3.9/3, 'gamma2_theta_r': 3.9/6,
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'tau2_r_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009, 'g_r_init': 0.00009,
        'theta_y_init': np.ones(3), 'theta_q_init': np.ones(3), 'theta_r_init': np.ones(3),
        'W_init': None, 'M_init': None, 'V_init': None, 'Lambda_init': None,
    }
    
    # If p is provided, initialize M, Lambda, V, W using SVD procedure
    if p is not None:
        init_dict = initialize_M_Lambda_V_W_Dgeneral(p=p, D=3, seed=seed)
        base.update({
            'W_init': init_dict['W_init'],
            'M_init': init_dict['M_init'],
            'V_init': init_dict['V_init'],
            'Lambda_init': init_dict['Lambda_init'],
            'prior_M': init_dict['prior_M'],
            'prior_V': init_dict['prior_V'],
        })
    
    base.update(kwargs)
    return base


def create_config_D5_L1(**kwargs) -> dict:
    """Configuration for D=5, Layer=1."""
    base = {
        'D': 5, 'layer': 1,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_all': True,  # MLE recommended for higher D
        'use_tf_gradients': True,
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5, 'gamma2_theta': 3.9, 'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'g_y_init': 0.00009, 'theta_y_init': np.ones(5),
    }
    base.update(kwargs)
    return base


def create_config_D5_L2(**kwargs) -> dict:
    """Configuration for D=5, Layer=2."""
    base = {
        'D': 5, 'layer': 2,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_all': True,
        'use_tf_gradients': True,
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9, 'gamma2_theta_q': 3.9/3,
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009,
        'theta_y_init': np.ones(5), 'theta_q_init': np.ones(5),
    }
    base.update(kwargs)
    return base


def create_config_D5_L3(**kwargs) -> dict:
    """Configuration for D=5, Layer=3."""
    base = {
        'D': 5, 'layer': 3,
        'n_chains': 3, 'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_all': True,
        'use_tf_gradients': True,
        'eps_hmc': 0.001, 'T_step_hmc': 17,
        'alpha1_tau2': 1.0, 'alpha2_tau2': 1000.0,
        'beta1_g': 0.01, 'beta2_g': 0.005, 'l_g': 1.0, 'u_g': 2.0,
        'gamma1_theta': 1.5,
        'gamma2_theta_y': 3.9, 'gamma2_theta_q': 3.9/3, 'gamma2_theta_r': 3.9/6,
        'l_theta': 1.0, 'u_theta': 2.0,
        'nu_lambda': None, 'epsilon_lambda': 2.0,
        'prior_M': None, 'prior_V': None,
        'tau2_y_init': 0.005, 'tau2_q_init': 0.005, 'tau2_r_init': 0.005, 'g_y_init': 0.00009, 'g_q_init': 0.00009, 'g_r_init': 0.00009,
        'theta_y_init': np.ones(5), 'theta_q_init': np.ones(5), 'theta_r_init': np.ones(5),
    }
    base.update(kwargs)
    return base


# =============================================================================
# CONFIGURATION FUNCTIONS FOR LAYER 1 VARIANTS
# =============================================================================

def create_config_L1_W_Known(W_fixed: np.ndarray, **kwargs) -> dict:
    """
    Configuration for Layer 1 with known/fixed W.
    
    Args:
        W_fixed: Fixed projection matrix (p, D) - must be provided
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler1Layer_W_Known
    """
    p, D = W_fixed.shape
    
    base = {
        'variant': 'W_Known',
        'W_fixed': W_fixed,
        'D': D,
        'layer': 1,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g': False, 'use_mle_theta': False,
        'kernel_type': 'isotropic_squared_exponential' if D == 1 else 'separable_squared_exponential',
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2': 3.9,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


def create_config_L1_No_W(**kwargs) -> dict:
    """
    Configuration for Layer 1 without W (using X directly).
    
    Args:
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler1Layer_No_W
    """
    base = {
        'variant': 'No_W',
        'layer': 1,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g': False, 'use_mle_theta': False,
        'kernel_type': 'separable_squared_exponential',  # Default for multi-dimensional X
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2': 3.9,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


def create_config_L1_No_W_Selective(D: int, column_indices: Optional[np.ndarray] = None, **kwargs) -> dict:
    """
    Configuration for Layer 1 without W, using selected columns of X.
    
    Args:
        D: Number of columns to use from X (must be <= p)
        column_indices: Optional array of column indices to use. 
                       If None, uses first D columns. Shape: (D,)
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler1Layer_No_W_Selective
    """
    base = {
        'variant': 'No_W_Selective',
        'D': D,
        'column_indices': column_indices,  # None means use first D columns
        'layer': 1,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g': False, 'use_mle_theta': False,
        'kernel_type': 'separable_squared_exponential' if D > 1 else 'isotropic_squared_exponential',
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2': 3.9,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


# =============================================================================
# CONFIGURATION FUNCTIONS FOR LAYER 2 VARIANTS
# =============================================================================

def create_config_L2_W_Known(W_fixed: np.ndarray, **kwargs) -> dict:
    """
    Configuration for Layer 2 with known/fixed W.
    
    Args:
        W_fixed: Fixed projection matrix (p, D) - must be provided
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler2Layer_W_Known
    """
    p, D = W_fixed.shape
    
    base = {
        'variant': 'W_Known',
        'W_fixed': W_fixed,
        'D': D,
        'layer': 2,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g_y': False, 'use_mle_theta_y': False,
        'kernel_type': 'isotropic_squared_exponential' if D == 1 else 'separable_squared_exponential',
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2_y': 3.9, 'gamma2_q': 3.9/3,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


def create_config_L2_No_W(**kwargs) -> dict:
    """
    Configuration for Layer 2 without W (using X directly).
    
    Args:
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler2Layer_No_W
    """
    base = {
        'variant': 'No_W',
        'layer': 2,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g_y': False, 'use_mle_theta_y': False,
        'kernel_type': 'separable_squared_exponential',  # Default for multi-dimensional X
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2_y': 3.9, 'gamma2_q': 3.9/3,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


def create_config_L2_No_W_Selective(D: int, column_indices: Optional[np.ndarray] = None, **kwargs) -> dict:
    """
    Configuration for Layer 2 without W, using selected columns of X.
    
    Args:
        D: Number of columns to use from X (must be <= p)
        column_indices: Optional array of column indices to use. 
                       If None, uses first D columns. Shape: (D,)
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler2Layer_No_W_Selective
    """
    base = {
        'variant': 'No_W_Selective',
        'D': D,
        'column_indices': column_indices,  # None means use first D columns
        'layer': 2,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g_y': False, 'use_mle_theta_y': False,
        'kernel_type': 'separable_squared_exponential' if D > 1 else 'isotropic_squared_exponential',
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2_y': 3.9, 'gamma2_q': 3.9/3,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


# =============================================================================
# CONFIGURATION FUNCTIONS FOR LAYER 3 VARIANTS
# =============================================================================

def create_config_L3_W_Known(W_fixed: np.ndarray, **kwargs) -> dict:
    """
    Configuration for Layer 3 with known/fixed W.
    
    Args:
        W_fixed: Fixed projection matrix (p, D) - must be provided
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler3Layer_W_Known
    """
    p, D = W_fixed.shape
    
    base = {
        'variant': 'W_Known',
        'W_fixed': W_fixed,
        'D': D,
        'layer': 3,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g_y': False, 'use_mle_theta_y': False,
        'kernel_type': 'isotropic_squared_exponential' if D == 1 else 'separable_squared_exponential',
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2_y': 3.9, 'gamma2_q': 3.9/3, 'gamma2_r': 3.9/6,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


def create_config_L3_No_W(**kwargs) -> dict:
    """
    Configuration for Layer 3 without W (using X directly).
    
    Args:
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler3Layer_No_W
    """
    base = {
        'variant': 'No_W',
        'layer': 3,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g_y': False, 'use_mle_theta_y': False,
        'kernel_type': 'separable_squared_exponential',  # Default for multi-dimensional X
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2_y': 3.9, 'gamma2_q': 3.9/3, 'gamma2_r': 3.9/6,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


def create_config_L3_No_W_Selective(D: int, column_indices: Optional[np.ndarray] = None, **kwargs) -> dict:
    """
    Configuration for Layer 3 without W, using selected columns of X.
    
    Args:
        D: Number of columns to use from X (must be <= p)
        column_indices: Optional array of column indices to use. 
                       If None, uses first D columns. Shape: (D,)
        **kwargs: Additional configuration overrides.
        
    Returns:
        Configuration dictionary for GibbsSampler3Layer_No_W_Selective
    """
    base = {
        'variant': 'No_W_Selective',
        'D': D,
        'column_indices': column_indices,  # None means use first D columns
        'layer': 3,
        'n_iterations': 2000, 'burn_in': 500, 'thin': 2,
        'use_mle_tau2': False, 'use_mle_g_y': False, 'use_mle_theta_y': False,
        'kernel_type': 'separable_squared_exponential' if D > 1 else 'isotropic_squared_exponential',
        'alpha1': 1.0, 'alpha2': 1000.0,
        'beta1': 0.01, 'beta2': 0.005,
        'gamma1': 1.5, 'gamma2_y': 3.9, 'gamma2_q': 3.9/3, 'gamma2_r': 3.9/6,
        'l': 1.0, 'u': 2.0,
    }
    
    base.update(kwargs)
    return base


# Quick access dictionary
CONFIG_FUNCTIONS = {
    (1, 1): create_config_D1_L1,
    (1, 2): create_config_D1_L2,
    (1, 3): create_config_D1_L3,
    (2, 1): create_config_D2_L1,
    (2, 2): create_config_D2_L2,
    (2, 3): create_config_D2_L3,
    (3, 1): create_config_D3_L1,
    (3, 2): create_config_D3_L2,
    (3, 3): create_config_D3_L3,
    (5, 1): create_config_D5_L1,
    (5, 2): create_config_D5_L2,
    (5, 3): create_config_D5_L3,
}


def get_config_for(D: int, layer: int, **overrides) -> dict:
    """
    Get configuration for specific D and layer combination.
    
    Args:
        D: Reduced dimension (1, 2, 3, 5)
        layer: Layer architecture (1, 2, 3)
        **overrides: Any parameters to override
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = get_config_for(D=2, layer=1, use_mle_all=True, n_iterations=5000)
        >>> results = run_multichain_analysis(Y_train, X_train, Y_test, X_test, **config)
    """
    if (D, layer) in CONFIG_FUNCTIONS:
        return CONFIG_FUNCTIONS[(D, layer)](**overrides)
    else:
        # Fallback: use generic config
        print(f"Warning: No preset for D={D}, layer={layer}. Using generic config.")
        return get_default_config(D, layer, n_train=100, p=10)


def run_multichain_analysis(
    # ========================================================================
    # DATA
    # ========================================================================
    Y_train: np.ndarray,
    X_train: np.ndarray,
    Y_test: np.ndarray,
    X_test: np.ndarray,
    
    # ========================================================================
    # MODEL SPECIFICATION
    # ========================================================================
    D: int = 1,
    layer: int = 1,
    variant: Optional[str] = None,  # W variants for layers 1/2/3: 'W_Known', 'No_W', 'No_W_Selective'
    W_fixed: Optional[np.ndarray] = None,  # For W_Known variant
    column_indices: Optional[np.ndarray] = None,  # For No_W_Selective variant
    
    # ========================================================================
    # MCMC SETTINGS
    # ========================================================================
    n_chains: int = 3,
    n_iterations: int = 2000,
    burn_in: int = 500,
    thin: int = 1,
    
    # ========================================================================
    # ESTIMATION METHOD OPTIONS
    # ========================================================================
    use_mle_tau2: bool = False,
    use_mle_g: bool = False,
    use_mle_theta: bool = False,
    use_mle_g_y: bool = False,      # For Layer 2: MLE for g_y (Y layer)
    use_mle_theta_y: bool = False,   # For Layer 2: MLE for theta_y (Y layer)
    use_mle_all: bool = False,
    use_tf_gradients: bool = False,
    
    # ========================================================================
    # KERNEL SELECTION (for D=1, Layer 1 and 2)
    # ========================================================================
    kernel_type: str = 'isotropic_squared_exponential',  # Options: 'isotropic_squared_exponential', 'separable_squared_exponential', 'isotropic_matern32', 'separable_matern32'
    
    # ========================================================================
    # HMC PARAMETERS (for W sampling on Stiefel manifold)
    # ========================================================================
    eps_hmc: float = 0.09,
    T_step_hmc: int = 15,
    M_hmc: int = 1,
    
    # ========================================================================
    # PRIOR HYPERPARAMETERS - TAU2 (observation noise)
    # ========================================================================
    alpha1_tau2: float = 1.0,      # InvGamma shape
    alpha2_tau2: float = 1000.0,   # InvGamma scale
    
    # ========================================================================
    # PRIOR HYPERPARAMETERS - G (nugget)
    # ========================================================================
    beta1_g: float = 0.01,         # Gamma shape
    beta2_g: float = 0.005,        # Gamma rate
    l_g: float = 1.0,              # MH proposal lower multiplier
    u_g: float = 2.0,              # MH proposal upper multiplier
    
    # ========================================================================
    # PRIOR HYPERPARAMETERS - THETA (lengthscale)
    # For multi-layer: specify per-layer rates (gamma2_theta_y, gamma2_theta_q, gamma2_theta_r)
    # Standard hierarchical values: b_y=3.9, b_q=3.9/3, b_r=3.9/6
    # ========================================================================
    gamma1_theta: float = 1.5,     # Gamma shape (3/2)
    gamma2_theta: float = 3.9,     # Gamma rate (1-layer or theta_y for multi-layer)
    gamma2_theta_y: float = 3.9,   # Gamma rate for outer layer (Y)
    gamma2_theta_q: float = 3.9/3, # Gamma rate for middle layer (Q)
    gamma2_theta_r: float = 3.9/6, # Gamma rate for inner layer (R)
    l_theta: float = 1.0,          # MH proposal lower multiplier
    u_theta: float = 2.0,          # MH proposal upper multiplier
    
    # ========================================================================
    # PRIOR HYPERPARAMETERS - LAMBDA (Matrix Langevin concentration)
    # ========================================================================
    nu_lambda: Optional[np.ndarray] = None,  # Compatibility placeholder; current Gibbs samplers draw from Gamma(5/2, 10/3)
    epsilon_lambda: float = 2.0,             # Minimum threshold
    max_iter_lambda: int = 1000,             # Max iterations for elliptical slice
    
    # ========================================================================
    # PRIOR PARAMETERS - M and V (Matrix Langevin)
    # ========================================================================
    prior_M: Optional[np.ndarray] = None,  # Prior for M (default: zeros(p, D))
    prior_V: Optional[np.ndarray] = None,  # Prior for V (default: zeros(D, D))
    mv_sampler: str = 'python',            # M/V sampler backend: 'python' or 'rstiefel'
    rstiefel_rscol: Optional[int] = None,  # Optional rstiefel simultaneous column update count
    
    # ========================================================================
    # INITIALIZATION VALUES (optional - will use defaults if None)
    # ========================================================================
    W_init: Optional[np.ndarray] = None,
    M_init: Optional[np.ndarray] = None,
    V_init: Optional[np.ndarray] = None,
    Lambda_init: Optional[np.ndarray] = None,
    tau2_init: Optional[float] = None,  # Backward-compatible alias for tau2_y_init
    g_init: Optional[float] = None,     # Backward-compatible alias for g_y_init
    theta_init: Optional[Union[float, np.ndarray]] = None,  # Alias for theta_y_init
    tau2_y_init: float = 0.005,
    tau2_q_init: Optional[Union[float, np.ndarray]] = 0.005,
    tau2_r_init: Optional[Union[float, np.ndarray]] = 0.005,
    
    # For multi-layer models
    g_y_init: float = 0.00009,
    g_q_init: float = 0.00009,
    g_r_init: float = 0.00009,
    theta_y_init: Optional[Union[float, np.ndarray]] = 1.0,
    theta_q_init: Optional[Union[float, np.ndarray]] = 1.0,
    theta_r_init: Optional[Union[float, np.ndarray]] = 1.0,
    
    # ========================================================================
    # LATENT VARIABLE SAMPLING PARAMETERS
    # ========================================================================
    Q_init: Optional[np.ndarray] = None,  # Initial Q for 2/3-layer
    R_init: Optional[np.ndarray] = None,  # Initial R for 3-layer
    
    # ========================================================================
    # OUTPUT OPTIONS
    # ========================================================================
    output_dir: str = './diagnostics',
    save_samples: bool = True,
    save_plots: bool = True,
    verbose: bool = True,
    
    # ========================================================================
    # OPTIONAL CODA/R DIAGNOSTICS (W + OTHER PARAMETERS)
    # ========================================================================
    compute_parameter_diagnostics: bool = True,
    diagnostics_burn: Optional[int] = None,
    diagnostics_ci: float = 0.95,
    diagnostics_use_projection_for_W: bool = False,
    diagnostics_r_home: Optional[str] = None,
    diagnostics_parameters: Optional[List[str]] = None
):
    """
    Run complete multi-chain Gibbs sampling analysis with full diagnostic suite.
    
    This function exposes ALL hyperparameters for complete user control.
    
    Args:
        ====================================================================
        DATA:
        ====================================================================
        Y_train: Training responses (n_train,)
        X_train: Training inputs (n_train, p)
        Y_test: Test responses (n_test,)
        X_test: Test inputs (n_test, p)
        
        ====================================================================
        MODEL SPECIFICATION:
        ====================================================================
        D: Reduced dimension (1, 2, 3, 5, ...)
        layer: Number of layers (1, 2, or 3)
            1-Layer: X → Z → Y
            2-Layer: X → Z → Q → Y
            3-Layer: X → Z → R → Q → Y
        
        ====================================================================
        MCMC SETTINGS:
        ====================================================================
        n_chains: Number of independent MCMC chains
        n_iterations: Total iterations per chain
        burn_in: Burn-in period (discarded)
        thin: Thinning interval (keep every thin-th sample)
        
        ====================================================================
        ESTIMATION OPTIONS:
        ====================================================================
        use_mle_tau2: Use MLE for τ² (faster than MCMC)
        use_mle_g: Use MLE for g (faster)
        use_mle_theta: Use MLE for θ (faster)
        use_mle_all: Use MLE for all hyperparameters (overrides above)
        use_tf_gradients: Use TensorFlow for W gradients (faster)
        
        ====================================================================
        HMC PARAMETERS (for W sampling on Stiefel manifold):
        ====================================================================
        eps_hmc: Step size for leapfrog integration (default: 0.09)
        T_step_hmc: Number of leapfrog steps (default: 15)
        M_hmc: Number of HMC samples to generate (default: 1)
        
        ====================================================================
        PRIOR HYPERPARAMETERS:
        ====================================================================
        
        For τ² (observation noise):
            alpha1_tau2: Inverse-Gamma shape parameter (default: 1.0)
            alpha2_tau2: Inverse-Gamma scale parameter (default: 1000.0)
            Prior: τ² ~ InvGamma(α₁, α₂)
        
        For g (nugget):
            beta1_g: Gamma shape parameter (default: 0.01)
            beta2_g: Gamma rate parameter (default: 0.005)
            l_g, u_g: MH proposal bounds [l*g/u, u*g/l] (default: 1.0, 2.0)
            Prior: g ~ Gamma(β₁, β₂)
        
        For θ (lengthscale):
            gamma1_theta: Gamma shape parameter (default: 1.5, i.e., 3/2)
            gamma2_theta: Gamma rate for 1-layer (default: 3.9)
            gamma2_theta_y: Gamma rate for outer layer Y (default: 3.9)
            gamma2_theta_q: Gamma rate for middle layer Q (default: 3.9/3 = 1.3)
            gamma2_theta_r: Gamma rate for inner layer R (default: 3.9/6 = 0.65)
            l_theta, u_theta: MH proposal bounds (default: 1.0, 2.0)
            Prior: θ ~ Gamma(3/2, b) with layer-specific rates b
            Hierarchical structure: b_r < b_q < b_y
        
        For Λ (Matrix Langevin concentration):
            nu_lambda: Compatibility placeholder (currently not used by full-model wrappers)
                Active implementation in Gibbs samplers draws per-iteration prior values from Gamma(5/2, 10/3).
            epsilon_lambda: Minimum threshold (default: 2.0)
            max_iter_lambda: Max iterations for elliptical slice (default: 1000)
        
        ====================================================================
        INITIALIZATION VALUES:
        ====================================================================
        W_init: Initial projection matrix (p, D) - uses random if None
        M_init, V_init, Lambda_init: Optional initial matrix Langevin factors
        mv_sampler: Backend for posterior M/V Gibbs updates.
            - 'python': local Python implementation (default)
            - 'rstiefel': R rstiefel::rmf.matrix.gibbs via rpy2
        rstiefel_rscol: Optional number of columns to update simultaneously in rstiefel
        tau2_y_init: Initial τ² for Y layer (default: 0.005)
        g_y_init: Initial g for Y layer (default: 0.00009)
        theta_y_init: Initial θ for Y layer (default: 1.0 / ones(D))
        tau2_init, g_init, theta_init: Backward-compatible aliases for Y-layer initials
        
        For multi-layer models:
            tau2_q_init, tau2_r_init: Initial τ² for latent layers (default: 0.005)
            g_q_init, g_r_init: Initial nuggets for latent layers (default: 0.00009)
            theta_q_init, theta_r_init: Initial lengthscales for latent layers (default: 1.0)
            Q_init: Initial latent Q (2/3-layer)
            R_init: Initial latent R (3-layer)
        
        ====================================================================
        OUTPUT:
        ====================================================================
        output_dir: Directory for diagnostic plots
        save_samples: Save MCMC samples to file
        save_plots: Save diagnostic plots
        verbose: Print progress
        
        Optional coda diagnostics (via rpy2 + R package coda):
            compute_parameter_diagnostics: Compute post-sampling diagnostics for parameters
            diagnostics_burn: Burn-in for diagnostics (default: burn_in)
            diagnostics_ci: CI level for summary table (default: 0.95)
            diagnostics_use_projection_for_W: For W, analyze W W^T instead of W entries
            diagnostics_r_home: Explicit R home path (if needed by rpy2)
            diagnostics_parameters: Specific parameters to analyze
        
    Returns:
        Dictionary with:
            - chains_samples: All MCMC samples per chain
            - chains_metrics: Performance metrics per chain
              (includes scalar metrics and per-iteration metric arrays)
            - convergence: R-hat and Heidelberg-Welch diagnostics
            - metrics_summary: Mean, median, std, CI for all metrics
            - parameter_diagnostics: Optional coda diagnostics (summary, heidel/ESS, rhat)
            - computation_times: Time per chain
    
    Example:
        >>> from Data_generation import generate_case1_1d
        >>> data = generate_case1_1d(n=200, seed=42)
        >>> results = run_multichain_analysis(
        ...     Y_train=data['y_train'],
        ...     X_train=data['X_train'],
        ...     Y_test=data['y_test'],
        ...     X_test=data['X_test'],
        ...     D=1,
        ...     layer=2,
        ...     n_chains=3,
        ...     n_iterations=2000,
        ...     burn_in=500,
        ...     use_mle_all=True,
        ...     eps_hmc=0.001,
        ...     T_step_hmc=17
        ... )
    """
    
    if verbose:
        print("="*70)
        print("MULTI-CHAIN GIBBS SAMPLER FOR BAYESIAN DIMENSIONALITY REDUCTION")
        print("="*70)
        print(f"\n{'MODEL CONFIGURATION':-^70}")
        print(f"  Reduced dimension (D): {D}")
        print(f"  Layer architecture: {layer}-layer")
        if layer == 1:
            print(f"    Structure: X → Z → Y")
            print(f"    Parameters: tau2_y, g_y, theta_D_y, W, M, V, Lambda")
        elif layer == 2:
            print(f"    Structure: X → Z → Q → Y")
            print(f"    Parameters: tau2_y, g_y, g_q, theta_y, theta_q, Q, W, M, V, Lambda")
        else:
            print(f"    Structure: X → Z → R → Q → Y")
            print(f"    Parameters: tau2_y, g_y, g_q, g_r, theta_y, theta_q, theta_r, R, Q, W, M, V, Lambda")
        
        print(f"\n{'MCMC CONFIGURATION':-^70}")
        print(f"  Number of chains: {n_chains}")
        print(f"  Iterations per chain: {n_iterations}")
        print(f"  Burn-in: {burn_in}")
        print(f"  Thinning: {thin}")
        print(f"  Saved samples per chain: {(n_iterations - burn_in) // thin}")
        
        print(f"\n{'ESTIMATION METHOD':-^70}")
        if use_mle_all:
            print(f"  Hyperparameters: MLE (all) - FAST")
        else:
            print(f"  tau2: {'MLE' if use_mle_tau2 else 'MCMC'}")
            print(f"  g: {'MLE' if use_mle_g else 'MCMC'}")
            print(f"  theta: {'MLE' if use_mle_theta else 'MCMC'}")
        print(f"  W gradients: {'TensorFlow' if use_tf_gradients else 'NumPy analytical'}")
        print(f"  M/V sampler: {mv_sampler}" + (f" (rstiefel rscol={rstiefel_rscol})" if rstiefel_rscol else ""))
        
        print(f"\n{'HMC PARAMETERS':-^70}")
        print(f"  Step size (eps): {eps_hmc}")
        print(f"  Leapfrog steps (T_step): {T_step_hmc}")
        print(f"  Samples per iteration (M): {M_hmc}")
        
        print(f"\n{'PRIOR HYPERPARAMETERS':-^70}")
        print(f"  tau2 ~ InvGamma({alpha1_tau2}, {alpha2_tau2})")
        print(f"  g ~ Gamma({beta1_g}, {beta2_g}) with MH bounds [{l_g}, {u_g}]")
        print(f"  theta ~ Gamma({gamma1_theta}, {gamma2_theta:.4f}) with MH bounds [{l_theta}, {u_theta}]")
        print("  Lambda prior draws: Gamma(5/2, 10/3)")
        print(f"  Lambda slice threshold epsilon: {epsilon_lambda}")
        
        print("="*70)

    # Backward-compatible alias resolution for initialization parameters
    effective_tau2_y_init = tau2_y_init if tau2_init is None else float(tau2_init)
    effective_tau2_q_init = effective_tau2_y_init if tau2_q_init is None else tau2_q_init
    effective_tau2_r_init = effective_tau2_q_init if tau2_r_init is None else tau2_r_init
    effective_g_y_init = g_y_init if g_init is None else float(g_init)
    effective_theta_y_init = theta_y_init if theta_init is None else theta_init
    
    # Select and configure sampler
    # Check if using Layer 1 variants
    if variant is not None and layer == 1:
        # Use Layer 1 variants multi-chain sampler
        multichain = MultiChainSampler_L1_Variants(
            variant=variant,
            n_chains=n_chains,
            n_iterations=n_iterations,
            burn_in=burn_in,
            thin=thin,
            use_mle_tau2=use_mle_tau2,
            use_mle_g=use_mle_g,
            use_mle_theta=use_mle_theta,
            kernel_type=kernel_type,
            W_fixed=W_fixed,
            D=D if variant == 'No_W_Selective' else None,
            column_indices=column_indices,
            alpha1=alpha1_tau2,
            alpha2=alpha2_tau2,
            beta1=beta1_g,
            beta2=beta2_g,
            gamma1=gamma1_theta,
            gamma2=gamma2_theta,
            l=l_g,
            u=u_g,
            tau2_init=effective_tau2_y_init,
            g_init=effective_g_y_init,
            theta_init=effective_theta_y_init
        )
    # Check if using Layer 2 variants
    elif variant is not None and layer == 2:
        # Use Layer 2 variants multi-chain sampler
        multichain = MultiChainSampler_L2_Variants(
            variant=variant,
            n_chains=n_chains,
            n_iterations=n_iterations,
            burn_in=burn_in,
            thin=thin,
            use_mle_tau2=use_mle_tau2,
            use_mle_g_y=use_mle_g_y,
            use_mle_theta_y=use_mle_theta_y,
            kernel_type=kernel_type,
            W_fixed=W_fixed,
            D=D if variant == 'No_W_Selective' else None,
            column_indices=column_indices,
            alpha1=alpha1_tau2,
            alpha2=alpha2_tau2,
            beta1=beta1_g,
            beta2=beta2_g,
            gamma1=gamma1_theta,
            gamma2_y=gamma2_theta_y,
            gamma2_q=gamma2_theta_q,
            l=l_g,
            u=u_g,
            tau2_y_init=effective_tau2_y_init,
            tau2_q_init=effective_tau2_q_init,
            g_y_init=effective_g_y_init,
            g_q_init=g_q_init,
            theta_y_init=effective_theta_y_init,
            theta_q_init=theta_q_init
        )
    # Check if using Layer 3 variants
    elif variant is not None and layer == 3:
        # Use Layer 3 variants multi-chain sampler
        multichain = MultiChainSampler_L3_Variants(
            variant=variant,
            n_chains=n_chains,
            n_iterations=n_iterations,
            burn_in=burn_in,
            thin=thin,
            use_mle_tau2=use_mle_tau2,
            use_mle_g_y=use_mle_g_y,
            use_mle_theta_y=use_mle_theta_y,
            kernel_type=kernel_type,
            W_fixed=W_fixed,
            D=D if variant == 'No_W_Selective' else None,
            column_indices=column_indices,
            alpha1=alpha1_tau2,
            alpha2=alpha2_tau2,
            beta1=beta1_g,
            beta2=beta2_g,
            gamma1=gamma1_theta,
            gamma2_y=gamma2_theta_y,
            gamma2_q=gamma2_theta_q,
            gamma2_r=gamma2_theta_r,
            l=l_g,
            u=u_g,
            tau2_y_init=effective_tau2_y_init,
            tau2_q_init=effective_tau2_q_init,
            tau2_r_init=effective_tau2_r_init,
            g_y_init=effective_g_y_init,
            g_q_init=g_q_init,
            g_r_init=g_r_init,
            theta_y_init=effective_theta_y_init,
            theta_q_init=theta_q_init,
            theta_r_init=theta_r_init
        )
    elif D == 1:
        multichain = MultiChainSampler_D1(
            n_chains=n_chains,
            layer=layer,
            n_iterations=n_iterations,
            burn_in=burn_in,
            thin=thin,
            use_mle_tau2=use_mle_tau2,
            use_mle_g=use_mle_g,
            use_mle_theta=use_mle_theta,
            use_mle_g_y=use_mle_g_y,
            use_mle_theta_y=use_mle_theta_y,
            use_mle_all=use_mle_all,
            use_tf_gradients=use_tf_gradients,
            kernel_type=kernel_type,
            prior_M=prior_M,
            prior_V=prior_V,
            W_init=W_init,
            M_init=M_init,
            V_init=V_init,
            Lambda_init=Lambda_init,
            tau2_y_init=effective_tau2_y_init,
            tau2_q_init=effective_tau2_q_init,
            tau2_r_init=effective_tau2_r_init,
            g_y_init=effective_g_y_init,
            g_q_init=g_q_init,
            g_r_init=g_r_init,
            theta_y_init=effective_theta_y_init,
            theta_q_init=theta_q_init,
            theta_r_init=theta_r_init,
            Q_init=Q_init,
            R_init=R_init,
            tau2_init=tau2_init,
            g_init=g_init,
            theta_init=theta_init,
            mv_sampler=mv_sampler,
            rstiefel_rscol=rstiefel_rscol
        )
    else:  # D > 1
        multichain = MultiChainSampler_Dgeneral(
            D=D,
            n_chains=n_chains,
            layer=layer,
            n_iterations=n_iterations,
            burn_in=burn_in,
            thin=thin,
            use_mle_tau2=use_mle_tau2,
            use_mle_g=use_mle_g,
            use_mle_theta=use_mle_theta,
            use_mle_g_y=use_mle_g_y,
            use_mle_theta_y=use_mle_theta_y,
            use_mle_all=use_mle_all,
            use_tf_gradients=use_tf_gradients,
            kernel_type=kernel_type,
            prior_M=prior_M,
            prior_V=prior_V,
            W_init=W_init,
            M_init=M_init,
            V_init=V_init,
            Lambda_init=Lambda_init,
            tau2_y_init=effective_tau2_y_init,
            tau2_q_init=effective_tau2_q_init,
            tau2_r_init=effective_tau2_r_init,
            g_y_init=effective_g_y_init,
            g_q_init=g_q_init,
            g_r_init=g_r_init,
            theta_y_init=effective_theta_y_init,
            theta_q_init=theta_q_init,
            theta_r_init=theta_r_init,
            Q_init=Q_init,
            R_init=R_init,
            tau2_init=tau2_init,
            g_init=g_init,
            theta_init=theta_init,
            mv_sampler=mv_sampler,
            rstiefel_rscol=rstiefel_rscol
        )
    
    # Run chains
    results = multichain.run_chains(Y_train, X_train, Y_test, X_test, verbose=verbose)
    
    # Optional coda diagnostics for model parameters (including matrix-valued W)
    if compute_parameter_diagnostics:
        diag_burn = burn_in if diagnostics_burn is None else diagnostics_burn
        try:
            results['parameter_diagnostics'] = compute_multichain_parameter_diagnostics(
                chains=results['chains_samples'],
                burn=diag_burn,
                ci=diagnostics_ci,
                use_projection_for_W=diagnostics_use_projection_for_W,
                r_home=diagnostics_r_home,
                parameters=diagnostics_parameters
            )
            if verbose:
                analyzed = ", ".join(results['parameter_diagnostics'].keys())
                print(f"\nComputed coda diagnostics for: {analyzed}")
        except Exception as exc:
            results['parameter_diagnostics_error'] = str(exc)
            if verbose:
                print(f"\nWarning: parameter diagnostics skipped ({exc})")
    
    # Create diagnostic plots
    if save_plots:
        if verbose:
            print(f"\nCreating diagnostic plots in: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        multichain.create_all_diagnostics(output_dir=output_dir)
    
    # Save samples
    if save_samples:
        import pickle
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/mcmc_samples.pkl", "wb") as f:
            pickle.dump(results, f)
        if verbose:
            print(f"Samples saved to: {output_dir}/mcmc_samples.pkl")
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        print(f"\n{'CONVERGENCE DIAGNOSTICS':-^70}")
        conv = results['convergence']
        for key, val in conv.items():
            if 'r_hat' in key:
                status = "✓ Excellent" if val < 1.05 else "✓ Good" if val < 1.1 else "⚠ Check" if val < 1.2 else "✗ Poor"
                print(f"  {key:<20} {val:8.4f}  {status}")
        
        print(f"\n{'PERFORMANCE METRICS':-^70}")
        print(f"{'Metric':<12} {'Mean':<10} {'Median':<10} {'Std':<10} {'95% CI':<20}")
        print("-"*70)
        for metric_name, metric_vals in results['metrics_summary'].items():
            print(f"{metric_name.upper():<12} "
                  f"{metric_vals['mean']:>9.4f} "
                  f"{metric_vals['median']:>9.4f} "
                  f"{metric_vals['std']:>9.4f} "
                  f"[{metric_vals['ci_lower']:>6.3f}, {metric_vals['ci_upper']:>6.3f}]")
        
        print(f"\n{'COMPUTATION TIME':-^70}")
        for i, t in enumerate(results['computation_times']):
            print(f"  Chain {i+1}: {t:.2f}s")
        print(f"  Total: {sum(results['computation_times']):.2f}s")
        print(f"  Average per chain: {np.mean(results['computation_times']):.2f}s")
        
        print("\n" + "="*70)
        print("✅ Analysis complete!")
        print("="*70)
    
    return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXAMPLE: Using Configuration Functions")
    print("="*70)
    print("\nThree ways to configure the sampler:")
    print("1. Use preset config functions")
    print("2. Use get_config_for() with overrides")
    print("3. Specify all parameters manually")
    print("="*70)
    
    # ==========================================================================
    # Example 1: Using Preset Config (D=1, Layer=1)
    # ==========================================================================
    print("\n" + "="*70)
    print("Example 1: D=1, Layer=1 using create_config_D1_L1()")
    print("="*70)
    
    data = generate_case1_1d(n=100, seed=42)
    
    # Get preset configuration and override some values
    config = create_config_D1_L1(
        n_iterations=10,  # Short for demo
        burn_in=2,
        use_mle_all=True,
        output_dir='./example1_D1_L1',
        verbose=True
    )
    
    results_ex1 = run_multichain_analysis(
        Y_train=data['y_train'],
        X_train=data['X_train'],
        Y_test=data['y_test'],
        X_test=data['X_test'],
        **config
    )
    
    # ==========================================================================
    # Example 2: Using get_config_for() (D=2, Layer=2)
    # ==========================================================================
    print("\n" + "="*70)
    print("Example 2: D=2, Layer=2 using get_config_for()")
    print("="*70)
    
    data2 = generate_case1_2d(n=100, seed=42)
    
    # Use convenient wrapper function
    config2 = get_config_for(
        D=2, layer=2,
        n_iterations=10,
        use_mle_all=True,
        output_dir='./example2_D2_L2',
        verbose=True
    )
    
    results_ex2 = run_multichain_analysis(
        Y_train=data2['y_train'],
        X_train=data2['X_train'],
        Y_test=data2['y_test'],
        X_test=data2['X_test'],
        **config2
    )
    
    # ==========================================================================
    # Example 3: Manual Configuration (D=3, Layer=1)
    # ==========================================================================
    print("\n" + "="*70)
    print("Example 3: D=3, Layer=1 with manual configuration")
    print("="*70)
    
    # Generate D=3 data
    np.random.seed(42)
    n, p = 100, 5
    X_train = np.random.randn(n, p)
    X_test = np.random.randn(20, p)
    W_true = np.random.randn(p, 3)
    W_true, _ = np.linalg.qr(W_true)
    Y_train = np.sin((X_train @ W_true).sum(axis=1)) + 0.1 * np.random.randn(n)
    Y_test = np.sin((X_test @ W_true).sum(axis=1)) + 0.1 * np.random.randn(20)
    
    # Use preset config
    config3 = create_config_D3_L1(
        n_iterations=10,
        use_mle_all=True,
        output_dir='./example3_D3_L1',
        verbose=True
    )
    
    results_ex3 = run_multichain_analysis(
        Y_train=Y_train,
        X_train=X_train,
        Y_test=Y_test,
        X_test=X_test,
        **config3
    )
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nAvailable preset configurations:")
    print("  - D=1: create_config_D1_L1(), create_config_D1_L2(), create_config_D1_L3()")
    print("  - D=2: create_config_D2_L1(), create_config_D2_L2(), create_config_D2_L3()")
    print("  - D=3: create_config_D3_L1(), create_config_D3_L2(), create_config_D3_L3()")
    print("  - D=5: create_config_D5_L1(), create_config_D5_L2(), create_config_D5_L3()")
    print("\nOr use: get_config_for(D=2, layer=1, **overrides)")
    print("\nDiagnostic plots saved to:")
    print("  - ./example1_D1_L1/")
    print("  - ./example2_D2_L2/")
    print("  - ./example3_D3_L1/")
