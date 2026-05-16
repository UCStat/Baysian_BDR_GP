"""
Verify all imports are correct after folder reorganization
"""

import sys
from pathlib import Path

base_dir = Path(__file__).parent

print("="*70)
print("VERIFYING ALL IMPORTS")
print("="*70)

# Test 1: Data Generation
print("\n[1/6] Testing Data Generation imports...")
sys.path.insert(0, str(base_dir / "Data Generation"))
try:
    from Data_generation import generate_case1_1d, generate_case1_2d
    print("✅ Data Generation imports SUCCESS")
except Exception as e:
    print(f"❌ Data Generation imports FAILED: {e}")

# Test 2: Covariance Functions
print("\n[2/6] Testing Covariance Functions imports...")
sys.path.insert(0, str(base_dir / "Covariance Functions"))
try:
    from covariance_kernel_functions_and_gradients_W import (
        IsotropicSquaredExponentialKernel,
        SeparableSquaredExponentialKernel
    )
    print("✅ Covariance Functions imports SUCCESS")
except Exception as e:
    print(f"❌ Covariance Functions imports FAILED: {e}")

# Test 3: Parameter Sampler
print("\n[3/6] Testing Parameter Sampler imports...")
sys.path.insert(0, str(base_dir / "Parameter Sampler"))
try:
    from parameter_sampler_D1 import sample_tau2, sample_g, sample_W_HMC_stiefel
    from parameter_sampler_Dgeneral import sample_tau2 as sample_tau2_Dg
    print("✅ Parameter Sampler imports SUCCESS")
except Exception as e:
    print(f"❌ Parameter Sampler imports FAILED: {e}")

# Test 4: Gibbs Sampling
print("\n[4/6] Testing Gibbs Sampling imports...")
sys.path.insert(0, str(base_dir / "Gibbs Sampling"))
try:
    from gibbs_sampler_layers_D1 import GibbsSampler1Layer
    from gibbs_sampler_layers_Dgeneral import GibbsSampler1Layer as GS1_Dg
    print("✅ Gibbs Sampling imports SUCCESS")
except Exception as e:
    print(f"❌ Gibbs Sampling imports FAILED: {e}")

# Test 5: BDR Metrics and Plot
print("\n[5/6] Testing BDR Metrics and Plot imports...")
sys.path.insert(0, str(base_dir / "BDR Metrics and Plot"))
try:
    from BDR_metrics import compute_RMSPE, compute_BIC
    from BDR_plot import plot_trace, plot_density
    print("✅ BDR Metrics and Plot imports SUCCESS")
except Exception as e:
    print(f"❌ BDR Metrics and Plot imports FAILED: {e}")

# Test 6: Multichain
print("\n[6/6] Testing Multichain imports...")
sys.path.insert(0, str(base_dir / "Multichain"))
try:
    from multichain_sampler_D1 import MultiChainSampler
    from multichain_sampler_Dgeneral import MultiChainSampler as MC_Dg
    print("✅ Multichain imports SUCCESS")
except Exception as e:
    print(f"❌ Multichain imports FAILED: {e}")

# Test 7: Integration test
print("\n[7/7] Testing full integration...")
try:
    # Generate data
    data = generate_case1_1d(n=20, seed=42)
    print(f"  ✓ Data generated: {len(data['y_train'])} samples")
    
    # Test sampler initialization
    from multichain_sampler_D1 import MultiChainSampler
    sampler = MultiChainSampler(n_chains=2, layer=1, n_iterations=2, burn_in=0, thin=1, use_mle_all=True)
    print(f"  ✓ Multi-chain sampler initialized")
    
    print("✅ Full integration SUCCESS")
except Exception as e:
    print(f"❌ Full integration FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

