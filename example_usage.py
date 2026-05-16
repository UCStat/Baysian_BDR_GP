"""
Example Usage of Data Generation Module

This script demonstrates how to use the data generation module
to create synthetic datasets for GP models with dimension reduction.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

base_dir = Path(__file__).parent
data_gen_path = str(base_dir / "Data Generation")
if data_gen_path not in sys.path:
    sys.path.insert(0, data_gen_path)

from Data_generation import (
    Case1_PolynomialChaos,
    Case2_Piecewise,
    Case2_Exponential,
    generate_case1_1d,
    generate_case1_2d,
    generate_case2_piecewise,
    generate_case2_exponential
)


def _add_example_paths():
    """Ensure local module folders are importable when running this script directly."""
    base_dir = Path(__file__).parent
    for folder in ["Gibbs Sampling", "Parameter Sampler", "Data Generation"]:
        folder_path = str(base_dir / folder)
        if folder_path not in sys.path:
            sys.path.insert(0, folder_path)


def run_mle_options_examples():
    """Run concise examples for MLE hyperparameter options."""
    _add_example_paths()
    from gibbs_sampler_layers_D1 import GibbsSampler1Layer, GibbsSampler2Layer
    from parameter_sampler_D1 import estimate_all_hyperparameters_MLE

    print("\n" + "=" * 70)
    print("MLE Options for Hyperparameter Estimation")
    print("=" * 70)

    np.random.seed(42)
    n, p = 80, 8
    X = np.random.randn(n, p)
    W_true = np.random.randn(p, 1)
    W_true = W_true / np.linalg.norm(W_true)
    Z = X @ W_true
    C = np.exp(-0.5 * np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2))
    Y = np.random.multivariate_normal(np.zeros(n), C + 0.01 * np.eye(n))

    scenarios = [
        ("Full MCMC", dict(use_mle_tau2=False, use_mle_g=False, use_mle_theta=False)),
        ("MLE tau2 only", dict(use_mle_tau2=True, use_mle_g=False, use_mle_theta=False)),
        ("MLE all", dict(use_mle_all=True)),
    ]

    print(f"\n{'Method':<20} {'tau2_y':<12} {'g_y':<12} {'theta_y':<12}")
    print("-" * 60)
    for label, opts in scenarios:
        sampler = GibbsSampler1Layer(
            Y=Y,
            X=X,
            D=1,
            n_iterations=200,
            burn_in=50,
            thin=2,
            **opts
        )
        out = sampler.run(verbose=False)
        tau2_y = float(np.mean(out["tau2_y"]))
        g_y = float(np.mean(out["g_y"]))
        theta_y = float(np.mean(out["theta_D_y"]))
        print(f"{label:<20} {tau2_y:<12.6f} {g_y:<12.6f} {theta_y:<12.6f}")

    W_init = np.random.randn(p, 1)
    W_init = W_init / np.linalg.norm(W_init)
    mle_est = estimate_all_hyperparameters_MLE(
        Y=Y,
        input_matrix=X,
        W=W_init,
        tau2_init=0.005,
        g_init=0.00009,
        theta_init=1.0,
        n_iterations=8,
        n_grid=40,
        verbose=False
    )
    print("\nStandalone MLE:")
    print(f"  tau2={mle_est['tau2']:.6f}, g={mle_est['g']:.6f}, theta={mle_est['theta_D']:.6f}")

    sampler_2layer = GibbsSampler2Layer(
        Y=Y,
        X=X,
        D=1,
        n_iterations=120,
        burn_in=20,
        thin=2,
        use_mle_all=True
    )
    out2 = sampler_2layer.run(verbose=False)
    print("\n2-Layer with MLE-all:")
    print(
        f"  tau2_y={np.mean(out2['tau2_y']):.6f}, "
        f"g_y={np.mean(out2['g_y']):.6f}, g_q={np.mean(out2['g_q']):.6f}, "
        f"theta_y={np.mean(out2['theta_y']):.6f}, theta_q={np.mean(out2['theta_q']):.6f}"
    )


def example_1_quick_generation():
    """Example 1: Quick data generation using convenience functions."""
    print("\n" + "="*60)
    print("Example 1: Quick Data Generation")
    print("="*60)
    
    # Case 1: Polynomial chaos with 1D subspace
    print("\nGenerating Case 1 (1D) with n=350 samples...")
    data = generate_case1_1d(n=350, seed=42)
    print(f"  - Training samples: {data['n_train']}")
    print(f"  - Test samples: {data['n_test']}")
    print(f"  - Input dimension: {data['X_train'].shape[1]}")
    print(f"  - Reduced dimension: {data['z_train'].shape[1]}")
    print(f"  - Training response range: [{data['y_train'].min():.3f}, {data['y_train'].max():.3f}]")
    
    # Case 2: Piecewise function
    print("\nGenerating Case 2 (Piecewise) with n=300 samples...")
    data = generate_case2_piecewise(n=300, seed=42)
    print(f"  - Training samples: {data['n_train']}")
    print(f"  - Test samples: {data['n_test']}")
    print(f"  - Training response range: [{data['y_train'].min():.3f}, {data['y_train'].max():.3f}]")
    
    return data


def example_2_custom_generation():
    """Example 2: Custom data generation with class instances."""
    print("\n" + "="*60)
    print("Example 2: Custom Data Generation")
    print("="*60)
    
    # Create generator instance for 2D polynomial chaos
    print("\nGenerating Case 1 (2D) with custom parameters...")
    generator = Case1_PolynomialChaos(dimension=2, seed=123)
    
    # Generate with custom sample size and split ratio
    data = generator.generate_data(n=600, train_ratio=0.75)
    print(f"  - Training samples: {data['n_train']} (75%)")
    print(f"  - Test samples: {data['n_test']} (25%)")
    print(f"  - Projection matrix W shape: {data['W'].shape}")
    
    # Access the true parameters
    print(f"\nTrue Parameters:")
    print(f"  - a0 (intercept): {generator.a0:.5f}")
    print(f"  - a (linear coef): {generator.a}")
    print(f"  - A (quadratic matrix):\n{generator.A}")
    
    return generator, data


def example_3_multiple_sample_sizes():
    """Example 3: Generate data with multiple sample sizes."""
    print("\n" + "="*60)
    print("Example 3: Multiple Sample Sizes")
    print("="*60)
    
    results = {}
    
    # Case 1 with different sample sizes
    for n in [350, 600]:
        print(f"\nCase 1 (1D), n={n}:")
        data = generate_case1_1d(n=n, seed=42)
        results[f'case1_1d_n{n}'] = data
        print(f"  - Train/Test: {data['n_train']}/{data['n_test']}")
        print(f"  - Mean response: {data['y_train'].mean():.3f}")
        print(f"  - Std response: {data['y_train'].std():.3f}")
    
    # Case 2 with different sample sizes
    for n in [300, 500]:
        print(f"\nCase 2 (Piecewise), n={n}:")
        data = generate_case2_piecewise(n=n, seed=42)
        results[f'case2_piecewise_n{n}'] = data
        print(f"  - Train/Test: {data['n_train']}/{data['n_test']}")
        print(f"  - Mean response: {data['y_train'].mean():.3f}")
        print(f"  - Std response: {data['y_train'].std():.3f}")
    
    return results


def example_4_visualize_data():
    """Example 4: Generate and visualize data."""
    print("\n" + "="*60)
    print("Example 4: Data Visualization")
    print("="*60)
    
    # Use non-interactive backend
    plt.switch_backend('Agg')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Case 1: 1D Polynomial Chaos
    print("\nGenerating and plotting Case 1 (1D)...")
    data_1d = generate_case1_1d(n=350, seed=42)
    ax = axes[0, 0]
    ax.scatter(data_1d['z_train'], data_1d['y_train'], 
              alpha=0.6, s=30, label='Train', c='blue')
    ax.scatter(data_1d['z_test'], data_1d['y_test'], 
              alpha=0.6, s=30, label='Test', c='red')
    ax.set_xlabel('z (Reduced Dimension)', fontsize=10)
    ax.set_ylabel('y (Response)', fontsize=10)
    ax.set_title('Case 1: Polynomial Chaos (1D)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Case 1: 2D Polynomial Chaos
    print("Generating and plotting Case 1 (2D)...")
    data_2d = generate_case1_2d(n=350, seed=42)
    ax = axes[0, 1]
    scatter = ax.scatter(data_2d['z_train'][:, 0], data_2d['z_train'][:, 1], 
                        c=data_2d['y_train'], cmap='viridis', s=40, alpha=0.7)
    ax.set_xlabel('z₁', fontsize=10)
    ax.set_ylabel('z₂', fontsize=10)
    ax.set_title('Case 1: Polynomial Chaos (2D)', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Response (y)')
    ax.grid(True, alpha=0.3)
    
    # Case 2: Piecewise Function
    print("Generating and plotting Case 2 (Piecewise)...")
    data_piecewise = generate_case2_piecewise(n=300, seed=42)
    ax = axes[1, 0]
    # Sort by z for better visualization
    train_idx = np.argsort(data_piecewise['z_train'].flatten())
    ax.scatter(data_piecewise['z_train'][train_idx], 
              data_piecewise['y_train'][train_idx], 
              alpha=0.6, s=30, label='Train', c='blue')
    ax.scatter(data_piecewise['z_test'], data_piecewise['y_test'], 
              alpha=0.6, s=30, label='Test', c='red')
    ax.set_xlabel('z (Reduced Dimension)', fontsize=10)
    ax.set_ylabel('y (Response)', fontsize=10)
    ax.set_title('Case 2: Piecewise Function (1D)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Case 2: Exponential Function
    print("Generating and plotting Case 2 (Exponential)...")
    data_exp = generate_case2_exponential(n=300, seed=42)
    ax = axes[1, 1]
    scatter = ax.scatter(data_exp['z_train'][:, 0], data_exp['z_train'][:, 1], 
                        c=data_exp['y_train'], cmap='coolwarm', s=40, alpha=0.7)
    ax.set_xlabel('z₁ (scaled)', fontsize=10)
    ax.set_ylabel('z₂ (scaled)', fontsize=10)
    ax.set_title('Case 2: Exponential Function (2D)', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Response (y)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'data_generation_examples.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Optionally show the plot
    # plt.show()
    
    return fig


def example_5_save_data():
    """Example 5: Generate and save data to files."""
    print("\n" + "="*60)
    print("Example 5: Save Generated Data")
    print("="*60)
    
    import os
    
    # Create output directory
    output_dir = 'generated_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Case 1 (1D) data
    print("\nGenerating and saving Case 1 (1D) data...")
    data = generate_case1_1d(n=350, seed=42)
    
    # Save to CSV files
    np.savetxt(f'{output_dir}/case1_1d_X_train.csv', data['X_train'], delimiter=',')
    np.savetxt(f'{output_dir}/case1_1d_y_train.csv', data['y_train'], delimiter=',')
    np.savetxt(f'{output_dir}/case1_1d_X_test.csv', data['X_test'], delimiter=',')
    np.savetxt(f'{output_dir}/case1_1d_y_test.csv', data['y_test'], delimiter=',')
    np.savetxt(f'{output_dir}/case1_1d_z_train.csv', data['z_train'], delimiter=',')
    np.savetxt(f'{output_dir}/case1_1d_z_test.csv', data['z_test'], delimiter=',')
    np.savetxt(f'{output_dir}/case1_1d_W.csv', data['W'], delimiter=',')
    
    print(f"  - Files saved to '{output_dir}/' directory")
    print(f"  - Total files: 7 (X_train, y_train, X_test, y_test, z_train, z_test, W)")
    
    return output_dir


def example_6_compare_cases():
    """Example 6: Compare statistics across different cases."""
    print("\n" + "="*60)
    print("Example 6: Compare Cases")
    print("="*60)
    
    cases = {
        'Case 1 (1D, n=350)': generate_case1_1d(n=350, seed=42),
        'Case 1 (2D, n=350)': generate_case1_2d(n=350, seed=42),
        'Case 2 (Piecewise, n=300)': generate_case2_piecewise(n=300, seed=42),
        'Case 2 (Exponential, n=300)': generate_case2_exponential(n=300, seed=42),
    }
    
    print("\nComparative Statistics:")
    print("-" * 80)
    print(f"{'Case':<30} {'n_train':<10} {'n_test':<10} {'y_mean':<12} {'y_std':<12}")
    print("-" * 80)
    
    for case_name, data in cases.items():
        y_mean = data['y_train'].mean()
        y_std = data['y_train'].std()
        print(f"{case_name:<30} {data['n_train']:<10} {data['n_test']:<10} "
              f"{y_mean:<12.4f} {y_std:<12.4f}")
    
    print("-" * 80)


def run_data_generation_examples():
    """Run data-generation examples."""
    print("\n" + "="*70)
    print(" "*15 + "DATA GENERATION MODULE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_quick_generation()
    example_2_custom_generation()
    example_3_multiple_sample_sizes()
    example_6_compare_cases()
    
    # Visualization example (requires matplotlib)
    try:
        example_4_visualize_data()
    except ImportError:
        print("\nSkipping visualization example (matplotlib not installed)")
    except Exception as e:
        print(f"\nVisualization example failed: {e}")
    
    # Save data example
    try:
        example_5_save_data()
    except Exception as e:
        print(f"\nSave data example failed: {e}")
    
    print("\n" + "="*70)
    print(" "*20 + "ALL EXAMPLES COMPLETED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example scripts for BDR project.")
    parser.add_argument(
        "--mode",
        choices=["data", "mle", "all"],
        default="data",
        help="Choose data-generation examples, MLE examples, or both."
    )
    args = parser.parse_args()

    if args.mode in ("data", "all"):
        run_data_generation_examples()
    if args.mode in ("mle", "all"):
        run_mle_options_examples()
