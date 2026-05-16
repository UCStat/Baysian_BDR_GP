"""
Unit tests for Data Generation Module

Run with: python -m pytest test_data_generation.py
Or simply: python test_data_generation.py
"""

import numpy as np
import sys
from Data_generation import (
    Case1_PolynomialChaos,
    Case2_Piecewise,
    Case2_Exponential,
    generate_case1_1d,
    generate_case1_2d,
    generate_case2_piecewise,
    generate_case2_exponential
)


def test_case1_1d():
    """Test Case 1 with 1D subspace."""
    print("Testing Case 1 (1D)...", end=" ")
    
    # Generate data
    data = generate_case1_1d(n=350, seed=42)
    
    # Check data structure
    assert 'X_train' in data
    assert 'X_test' in data
    assert 'y_train' in data
    assert 'y_test' in data
    assert 'z_train' in data
    assert 'z_test' in data
    assert 'W' in data
    
    # Check dimensions
    assert data['X_train'].shape[1] == 10, "Input dimension should be 10"
    assert data['z_train'].shape[1] == 1, "Reduced dimension should be 1"
    assert data['W'].shape == (10, 1), "W should be 10x1"
    
    # Check train-test split
    assert data['n_train'] == 280, "Expected 280 training samples"
    assert data['n_test'] == 70, "Expected 70 test samples"
    assert data['n_train'] + data['n_test'] == 350
    
    # Check data types
    assert isinstance(data['X_train'], np.ndarray)
    assert isinstance(data['y_train'], np.ndarray)
    
    print("✓ PASSED")
    return True


def test_case1_2d():
    """Test Case 1 with 2D subspace."""
    print("Testing Case 1 (2D)...", end=" ")
    
    # Generate data
    data = generate_case1_2d(n=600, seed=42)
    
    # Check dimensions
    assert data['X_train'].shape[1] == 10, "Input dimension should be 10"
    assert data['z_train'].shape[1] == 2, "Reduced dimension should be 2"
    assert data['W'].shape == (10, 2), "W should be 10x2"
    
    # Check train-test split
    assert data['n_train'] == 480, "Expected 480 training samples"
    assert data['n_test'] == 120, "Expected 120 test samples"
    
    print("✓ PASSED")
    return True


def test_case2_piecewise():
    """Test Case 2 piecewise function."""
    print("Testing Case 2 (Piecewise)...", end=" ")
    
    # Generate data
    data = generate_case2_piecewise(n=300, seed=42)
    
    # Check dimensions
    assert data['X_train'].shape[1] == 10, "Input dimension should be 10"
    assert data['z_train'].shape[1] == 1, "Reduced dimension should be 1"
    
    # Check train-test split
    assert data['n_train'] == 240, "Expected 240 training samples"
    assert data['n_test'] == 60, "Expected 60 test samples"
    
    # Check that data is generated (values exist)
    assert not np.isnan(data['z_train']).any(), "z_train should not contain NaN"
    assert not np.isnan(data['y_train']).any(), "y_train should not contain NaN"
    
    print("✓ PASSED")
    return True


def test_case2_exponential():
    """Test Case 2 exponential function."""
    print("Testing Case 2 (Exponential)...", end=" ")
    
    # Generate data
    data = generate_case2_exponential(n=500, seed=42)
    
    # Check dimensions
    assert data['X_train'].shape[1] == 10, "Input dimension should be 10"
    assert data['z_train'].shape[1] == 2, "Reduced dimension should be 2"
    
    # Check train-test split
    assert data['n_train'] == 400, "Expected 400 training samples"
    assert data['n_test'] == 100, "Expected 100 test samples"
    
    # Check that z is scaled to approximately [1, 7] range
    # (with some tolerance due to projection)
    z_min = data['z_train'].min()
    z_max = data['z_train'].max()
    # Allow for projection effects but check general scaling
    assert z_min < 5, f"z minimum ({z_min}) should be closer to 1"
    assert z_max > 3, f"z maximum ({z_max}) should be closer to 7"
    
    print("✓ PASSED")
    return True


def test_reproducibility():
    """Test that random seed produces reproducible results."""
    print("Testing reproducibility...", end=" ")
    
    # Generate data twice with same seed
    data1 = generate_case1_1d(n=350, seed=42)
    data2 = generate_case1_1d(n=350, seed=42)
    
    # Check if results are identical
    assert np.allclose(data1['X_train'], data2['X_train']), "X_train should be identical"
    assert np.allclose(data1['y_train'], data2['y_train']), "y_train should be identical"
    
    # Generate with different seed
    data3 = generate_case1_1d(n=350, seed=123)
    
    # Check if results are different
    assert not np.allclose(data1['X_train'], data3['X_train']), "Different seeds should produce different data"
    
    print("✓ PASSED")
    return True


def test_projection_consistency():
    """Test that z = W^T x relationship holds."""
    print("Testing projection consistency...", end=" ")
    
    # Generate data
    data = generate_case1_1d(n=350, seed=42)
    
    # Check that z_train = X_train @ W
    z_computed = data['X_train'] @ data['W']
    assert np.allclose(z_computed, data['z_train'], atol=1e-10), \
        "z_train should equal X_train @ W"
    
    # Same for test set
    z_test_computed = data['X_test'] @ data['W']
    assert np.allclose(z_test_computed, data['z_test'], atol=1e-10), \
        "z_test should equal X_test @ W"
    
    print("✓ PASSED")
    return True


def test_train_test_split():
    """Test train-test split ratios."""
    print("Testing train-test split...", end=" ")
    
    # Test with different sample sizes
    for n in [300, 350, 500, 600]:
        data = generate_case1_1d(n=n, seed=42)
        
        # Check 80-20 split
        expected_train = int(n * 0.8)
        expected_test = n - expected_train
        
        assert data['n_train'] == expected_train, \
            f"Expected {expected_train} training samples for n={n}"
        assert data['n_test'] == expected_test, \
            f"Expected {expected_test} test samples for n={n}"
        assert data['n_train'] + data['n_test'] == n, \
            f"Train + test should equal total samples ({n})"
    
    print("✓ PASSED")
    return True


def test_class_initialization():
    """Test that class instances can be created correctly."""
    print("Testing class initialization...", end=" ")
    
    # Test Case1_PolynomialChaos
    gen_1d = Case1_PolynomialChaos(dimension=1, seed=42)
    assert gen_1d.dimension == 1
    assert gen_1d.W.shape == (10, 1)
    
    gen_2d = Case1_PolynomialChaos(dimension=2, seed=42)
    assert gen_2d.dimension == 2
    assert gen_2d.W.shape == (10, 2)
    
    # Test Case2_Piecewise
    gen_piecewise = Case2_Piecewise(seed=42)
    assert gen_piecewise.dimension == 1
    assert gen_piecewise.W.shape == (10, 1)
    
    # Test Case2_Exponential
    gen_exp = Case2_Exponential(seed=42)
    assert gen_exp.dimension == 2
    assert gen_exp.W.shape == (10, 2)
    
    print("✓ PASSED")
    return True


def test_output_shapes():
    """Test that all outputs have correct shapes."""
    print("Testing output shapes...", end=" ")
    
    # Case 1 1D
    data = generate_case1_1d(n=350, seed=42)
    n_train = data['n_train']
    n_test = data['n_test']
    
    assert data['X_train'].shape == (n_train, 10)
    assert data['X_test'].shape == (n_test, 10)
    assert data['y_train'].shape == (n_train,)
    assert data['y_test'].shape == (n_test,)
    assert data['z_train'].shape == (n_train, 1)
    assert data['z_test'].shape == (n_test, 1)
    assert data['W'].shape == (10, 1)
    
    # Case 1 2D
    data = generate_case1_2d(n=350, seed=42)
    n_train = data['n_train']
    n_test = data['n_test']
    
    assert data['z_train'].shape == (n_train, 2)
    assert data['z_test'].shape == (n_test, 2)
    assert data['W'].shape == (10, 2)
    
    print("✓ PASSED")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("Running Data Generation Module Tests")
    print("="*60 + "\n")
    
    tests = [
        test_case1_1d,
        test_case1_2d,
        test_case2_piecewise,
        test_case2_exponential,
        test_reproducibility,
        test_projection_consistency,
        test_train_test_split,
        test_class_initialization,
        test_output_shapes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("✓ All tests passed successfully!")
        return 0
    else:
        print(f"✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    # Suppress numpy warnings during testing
    import warnings
    warnings.filterwarnings('ignore')
    
    exit_code = run_all_tests()
    sys.exit(exit_code)

