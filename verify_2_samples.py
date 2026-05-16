"""
Verify that all samplers produce exactly 2 posterior samples
"""

import numpy as np
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Add module paths
base_dir = Path(__file__).parent
for folder in ["Multichain", "Gibbs Sampling", "Parameter Sampler", "BDR Metrics and Plot", "Data Generation"]:
    sys.path.insert(0, str(base_dir / folder))

print("="*70)
print("VERIFICATION: 2 POSTERIOR SAMPLES")
print("="*70)

# Generate test data
np.random.seed(42)
n_train, n_test, p = 20, 10, 5

X_train = np.random.randn(n_train, p)
X_test = np.random.randn(n_test, p)

# Test D=1, Layer 1
print("\n" + "="*70)
print("TEST 1: D=1, Layer 1 (Full Model)")
print("="*70)

from multichain_sampler_D1 import MultiChainSampler

W_true_D1 = np.random.randn(p, 1)
W_true_D1 = W_true_D1 / np.linalg.norm(W_true_D1)
Y_train_D1 = np.sin((X_train @ W_true_D1).flatten()) + 0.1 * np.random.randn(n_train)
Y_test_D1 = np.sin((X_test @ W_true_D1).flatten()) + 0.1 * np.random.randn(n_test)

multichain_D1_L1 = MultiChainSampler(
    n_chains=2,
    layer=1,
    n_iterations=2,
    burn_in=0,
    thin=1,
    use_mle_all=True,
    kernel_type='isotropic_squared_exponential'
)

results_D1_L1 = multichain_D1_L1.run_chains(Y_train_D1, X_train, Y_test_D1, X_test, verbose=False)

# Check sample counts for all parameters
print("\nSample counts per chain:")
for chain_idx in range(2):
    samples = results_D1_L1['chains_samples'][chain_idx]
    print(f"\nChain {chain_idx + 1}:")
    # Check all possible parameter names
    param_names = ['tau2', 'g', 'theta', 'theta_D', 'W', 'M', 'V', 'Lambda']
    for param_name in param_names:
        if param_name in samples:
            param_samples = samples[param_name]
            if isinstance(param_samples, np.ndarray):
                if param_samples.ndim == 1:
                    count = len(param_samples)
                elif param_samples.ndim == 2:
                    count = param_samples.shape[0]
                elif param_samples.ndim == 3:
                    count = param_samples.shape[0]
                else:
                    count = param_samples.shape[0] if param_samples.size > 0 else 0
            else:
                count = len(param_samples) if hasattr(param_samples, '__len__') else 1
            status = "✅" if count == 2 else "❌"
            print(f"  {status} {param_name}: {count} samples")
    # Show all available keys
    print(f"  Available keys: {list(samples.keys())}")

# Test D>1, Layer 1
print("\n" + "="*70)
print("TEST 2: D=2, Layer 1 (Full Model)")
print("="*70)

from multichain_sampler_Dgeneral import MultiChainSampler as MultiChainSampler_Dgen

W_true_D2 = np.random.randn(p, 2)
W_true_D2, _ = np.linalg.qr(W_true_D2)
Y_train_D2 = np.sin((X_train @ W_true_D2).sum(axis=1)) + 0.1 * np.random.randn(n_train)
Y_test_D2 = np.sin((X_test @ W_true_D2).sum(axis=1)) + 0.1 * np.random.randn(n_test)

multichain_D2_L1 = MultiChainSampler_Dgen(
    n_chains=2,
    layer=1,
    D=2,
    n_iterations=2,
    burn_in=0,
    thin=1,
    use_mle_all=True,
    kernel_type='separable_squared_exponential'
)

results_D2_L1 = multichain_D2_L1.run_chains(Y_train_D2, X_train, Y_test_D2, X_test, verbose=False)

# Check sample counts for all parameters
print("\nSample counts per chain:")
for chain_idx in range(2):
    samples = results_D2_L1['chains_samples'][chain_idx]
    print(f"\nChain {chain_idx + 1}:")
    for param_name in ['tau2', 'g', 'theta_D', 'W', 'M', 'V', 'Lambda']:
        if param_name in samples:
            param_samples = samples[param_name]
            if isinstance(param_samples, np.ndarray):
                if param_samples.ndim == 1:
                    count = len(param_samples)
                elif param_samples.ndim == 2:
                    count = param_samples.shape[0]
                elif param_samples.ndim == 3:
                    count = param_samples.shape[0]
                else:
                    count = param_samples.shape[0] if param_samples.size > 0 else 0
            else:
                count = len(param_samples) if hasattr(param_samples, '__len__') else 1
            status = "✅" if count == 2 else "❌"
            print(f"  {status} {param_name}: {count} samples")
        else:
            print(f"  ⚠️  {param_name}: not found")

# Test Layer 2 variant
print("\n" + "="*70)
print("TEST 3: D=1, Layer 2, W_Known Variant")
print("="*70)

from multichain_sampler_L1_variants import MultiChainSampler_L1_Variants

# Use known W
W_known = W_true_D1

multichain_L1_WK = MultiChainSampler_L1_Variants(
    variant='W_Known',
    W_fixed=W_known,
    n_chains=2,
    n_iterations=2,
    burn_in=0,
    thin=1,
    use_mle_tau2=True,
    use_mle_g=False,
    use_mle_theta=True,
    kernel_type='isotropic_squared_exponential'
)

results_L1_WK = multichain_L1_WK.run_chains(Y_train_D1, X_train, Y_test_D1, X_test, verbose=False)

# Check sample counts
print("\nSample counts per chain:")
for chain_idx in range(2):
    samples = results_L1_WK['chains_samples'][chain_idx]
    print(f"\nChain {chain_idx + 1}:")
    # Check all possible parameter names
    param_names = ['tau2', 'g', 'theta', 'theta_D']
    for param_name in param_names:
        if param_name in samples:
            param_samples = samples[param_name]
            if isinstance(param_samples, np.ndarray):
                count = len(param_samples) if param_samples.ndim == 1 else param_samples.shape[0]
            else:
                count = len(param_samples) if hasattr(param_samples, '__len__') else 1
            status = "✅" if count == 2 else "❌"
            print(f"  {status} {param_name}: {count} samples")
    # Show all available keys
    print(f"  Available keys: {list(samples.keys())}")

print("\n" + "="*70)
print("✅ VERIFICATION COMPLETE")
print("="*70)
print("\nAll tests should show exactly 2 samples for each parameter.")
print("If any parameter shows ❌, it means it doesn't have exactly 2 samples.")

