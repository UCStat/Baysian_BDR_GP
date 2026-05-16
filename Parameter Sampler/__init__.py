"""Parameter Sampler Module for GP Bayesian Framework."""

# D=1 imports
from .parameter_sampler_D1 import (
    sample_tau2, sample_g, sample_theta_D, sample_W_HMC_stiefel,
    sample_M, sample_V, sample_Lambda_slice,
    estimate_tau2_MLE, estimate_g_MLE, estimate_theta_D_MLE,
    estimate_all_hyperparameters_MLE,
    covar_sep, log_likelihood_gp,
    NullC, rW, rmf_vector, rmf_matrix
)

# Make module accessible for both import styles
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

__all__ = [
    'sample_tau2', 'sample_g', 'sample_theta_D', 'sample_W_HMC_stiefel',
    'sample_M', 'sample_V', 'sample_Lambda_slice',
    'estimate_tau2_MLE', 'estimate_g_MLE', 'estimate_theta_D_MLE',
    'estimate_all_hyperparameters_MLE',
    'covar_sep', 'log_likelihood_gp'
]
