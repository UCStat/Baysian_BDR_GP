"""
Data Generation Module for GP Bayesian Application

This module implements synthetic data generation scenarios for Gaussian Process models
with dimensionality reduction, as described in the research paper.

Cases:
    1. Polynomial chaos function with known structure (1D and 2D)
    2. Piecewise function (1D)
    3. Exponential function (2D)
"""

import numpy as np
from pyDOE import lhs
from typing import Tuple, Optional


class DataGenerator:
    """Base class for data generation scenarios."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                        train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.
        
        Args:
            X: Input features
            y: Response variable
            train_ratio: Proportion of data for training
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        n = X.shape[0]
        n_train = int(n * train_ratio)
        
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class Case1_PolynomialChaos(DataGenerator):
    """
    Case 1: Synthetic response surface of polynomial chaos function with known structure.
    
    The function is defined as:
        f(x) ≈ g(z), where z = W^T x
        g(z) = a0 + a^T z + z^T A z
        Y = g(z) + ε, where ε ~ N(0, σ²_ε)
    """
    
    def __init__(self, dimension: int = 1, seed: Optional[int] = None):
        """
        Initialize Case 1 data generator.
        
        Args:
            dimension: Dimension of the reduced space (1 or 2)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.dimension = dimension
        self.p = 10  # Original dimension
        self._set_parameters()
    
    def _set_parameters(self):
        """Set the true parameters for the data generation."""
        if self.dimension == 1:
            # 1D Input Subspace parameters
            self.W = np.array([
                -0.0091, -0.0579, -0.1877, 0.4774, 0.4559, 
                -0.6714, -0.1264, -0.0082, 0.0724, -0.2308
            ]).reshape(-1, 1)
            
            self.a0 = -0.16113
            self.a = np.array([-0.97483])
            self.A = np.array([[-1.66526]])
            self.sigma_epsilon = 0.1  # Standard deviation
            
        elif self.dimension == 2:
            # 2D Input Subspace parameters
            self.W = np.array([
                [0.00840, 0.0672],
                [-0.18426, -0.4148],
                [0.34300, 0.4821],
                [-0.05347, 0.0755],
                [0.08108, 0.2101],
                [0.06556, 0.5375],
                [-0.41219, 0.0781],
                [0.65424, -0.2002],
                [0.48483, -0.2912],
                [0.03966, 0.3480]
            ])
            
            self.a0 = -0.06976
            self.a = np.array([0.4376, 0.9870])
            self.A = np.array([
                [-0.9257, -0.3840],
                [-0.4174, -0.6766]
            ])
            self.sigma_epsilon = 0.1  # Standard deviation
        else:
            raise ValueError("Dimension must be 1 or 2")
    
    def g_function(self, z: np.ndarray) -> np.ndarray:
        """
        Quadratic function g(z) = a0 + a^T z + z^T A z.
        
        Args:
            z: Reduced dimensional input (n x D)
            
        Returns:
            Function values (n,)
        """
        if z.ndim == 1:
            z = z.reshape(1, -1)
        
        linear_term = z @ self.a
        quadratic_term = np.sum(z @ self.A * z, axis=1)
        
        return self.a0 + linear_term + quadratic_term
    
    def generate_data(self, n: int, train_ratio: float = 0.8) -> dict:
        """
        Generate synthetic data for Case 1.
        
        Args:
            n: Total number of samples
            train_ratio: Proportion of data for training
            
        Returns:
            Dictionary containing X_train, X_test, y_train, y_test, z_train, z_test, W
        """
        # Generate input samples from standard normal distribution
        X = np.random.randn(n, self.p)
        
        # Project to lower dimension
        z = X @ self.W
        
        # Compute response
        g_z = self.g_function(z)
        
        # Add Gaussian noise
        epsilon = np.random.normal(0, self.sigma_epsilon, n)
        y = g_z + epsilon
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, train_ratio)
        
        # Also compute z for train and test
        z_train = X_train @ self.W
        z_test = X_test @ self.W
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'z_train': z_train,
            'z_test': z_test,
            'W': self.W,
            'n_train': len(y_train),
            'n_test': len(y_test)
        }


class Case2_Piecewise(DataGenerator):
    """
    Case 2.1: Synthetic response surface of a piecewise function.
    
    The function is defined as:
        g(z) = 1.35 * cos(12πz),     if z < 0.333
        g(z) = 1.35,                  if 0.333 ≤ z ≤ 0.666
        g(z) = 1.35 * cos(6πz),      if z > 0.666
    
    where z = W^T x and x is sampled using Latin Hypercube Sampling.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Case 2.1 data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.p = 10  # Original dimension
        self.dimension = 1
        
        # Use the same W as Case 1 1D
        self.W = np.array([
            -0.0091, -0.0579, -0.1877, 0.4774, 0.4559, 
            -0.6714, -0.1264, -0.0082, 0.0724, -0.2308
        ]).reshape(-1, 1)
        
        self.noise_std = 0.1
    
    def g_function(self, z: np.ndarray) -> np.ndarray:
        """
        Piecewise function.
        
        Args:
            z: 1D input (n,) or (n, 1)
            
        Returns:
            Function values (n,)
        """
        z = z.flatten()
        result = np.zeros_like(z)
        
        # Regime 1: z < 0.333
        mask1 = z < 0.333
        result[mask1] = 1.35 * np.cos(12 * np.pi * z[mask1])
        
        # Regime 2: 0.333 ≤ z ≤ 0.666
        mask2 = (z >= 0.333) & (z <= 0.666)
        result[mask2] = 1.35
        
        # Regime 3: z > 0.666
        mask3 = z > 0.666
        result[mask3] = 1.35 * np.cos(6 * np.pi * z[mask3])
        
        return result
    
    def generate_data(self, n: int, train_ratio: float = 0.8) -> dict:
        """
        Generate synthetic data using Latin Hypercube Sampling.
        
        Args:
            n: Total number of samples
            train_ratio: Proportion of data for training
            
        Returns:
            Dictionary containing X_train, X_test, y_train, y_test, z_train, z_test, W
        """
        # Generate samples using Latin Hypercube Sampling in [0, 1]^10
        X = lhs(self.p, samples=n)
        
        # Project to lower dimension
        z = X @ self.W
        
        # Compute response
        g_z = self.g_function(z)
        
        # Add Gaussian noise
        epsilon = np.random.normal(0, self.noise_std, n)
        y = g_z + epsilon
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, train_ratio)
        
        # Also compute z for train and test
        z_train = X_train @ self.W
        z_test = X_test @ self.W
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'z_train': z_train,
            'z_test': z_test,
            'W': self.W,
            'n_train': len(y_train),
            'n_test': len(y_test)
        }


class Case2_Exponential(DataGenerator):
    """
    Case 2.2: Synthetic response surface of an exponential function.
    
    The function is defined as:
        f(z1, z2) = 10 * z1 * exp(-z1² - z2²)
    
    where z = [z1, z2]^T = W^T x, and x is sampled using Latin Hypercube Sampling.
    The components z1 and z2 are scaled to [1, 7] using:
        z_j = (z_j - 0.5) * 6 + 1
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Case 2.2 data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.p = 10  # Original dimension
        self.dimension = 2
        
        # Use the same W as Case 1 2D
        self.W = np.array([
            [0.00840, 0.0672],
            [-0.18426, -0.4148],
            [0.34300, 0.4821],
            [-0.05347, 0.0755],
            [0.08108, 0.2101],
            [0.06556, 0.5375],
            [-0.41219, 0.0781],
            [0.65424, -0.2002],
            [0.48483, -0.2912],
            [0.03966, 0.3480]
        ])
        
        self.noise_std = 0.1
    
    def scale_z(self, z: np.ndarray) -> np.ndarray:
        """
        Scale z components from [0, 1] to [1, 7].
        
        Args:
            z: Input in original scale (n, 2)
            
        Returns:
            Scaled z (n, 2)
        """
        return (z - 0.5) * 6 + 1
    
    def f_function(self, z: np.ndarray) -> np.ndarray:
        """
        Exponential function f(z1, z2) = 10 * z1 * exp(-z1² - z2²).
        
        Args:
            z: 2D input (n, 2) - should already be scaled
            
        Returns:
            Function values (n,)
        """
        z1 = z[:, 0]
        z2 = z[:, 1]
        
        return 10 * z1 * np.exp(-z1**2 - z2**2)
    
    def generate_data(self, n: int, train_ratio: float = 0.8) -> dict:
        """
        Generate synthetic data using Latin Hypercube Sampling.
        
        Args:
            n: Total number of samples
            train_ratio: Proportion of data for training
            
        Returns:
            Dictionary containing X_train, X_test, y_train, y_test, z_train, z_test, W
        """
        # Generate samples using Latin Hypercube Sampling in [0, 1]^10
        X = lhs(self.p, samples=n)
        
        # Project to lower dimension
        z = X @ self.W
        
        # Scale z to [1, 7]
        z_scaled = self.scale_z(z)
        
        # Compute response
        f_z = self.f_function(z_scaled)
        
        # Add Gaussian noise
        epsilon = np.random.normal(0, self.noise_std, n)
        y = f_z + epsilon
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, train_ratio)
        
        # Also compute z for train and test
        z_train = X_train @ self.W
        z_test = X_test @ self.W
        z_train_scaled = self.scale_z(z_train)
        z_test_scaled = self.scale_z(z_test)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'z_train': z_train_scaled,
            'z_test': z_test_scaled,
            'z_train_unscaled': z_train,
            'z_test_unscaled': z_test,
            'W': self.W,
            'n_train': len(y_train),
            'n_test': len(y_test)
        }


# Example usage functions
def generate_case1_1d(n: int = 350, seed: int = 42) -> dict:
    """
    Generate Case 1 data with 1D subspace.
    
    Args:
        n: Total number of samples (350 or 600)
        seed: Random seed
        
    Returns:
        Dictionary with train/test data
    """
    generator = Case1_PolynomialChaos(dimension=1, seed=seed)
    return generator.generate_data(n=n)


def generate_case1_2d(n: int = 350, seed: int = 42) -> dict:
    """
    Generate Case 1 data with 2D subspace.
    
    Args:
        n: Total number of samples (350 or 600)
        seed: Random seed
        
    Returns:
        Dictionary with train/test data
    """
    generator = Case1_PolynomialChaos(dimension=2, seed=seed)
    return generator.generate_data(n=n)


def generate_case2_piecewise(n: int = 300, seed: int = 42) -> dict:
    """
    Generate Case 2 piecewise function data.
    
    Args:
        n: Total number of samples (300 or 500)
        seed: Random seed
        
    Returns:
        Dictionary with train/test data
    """
    generator = Case2_Piecewise(seed=seed)
    return generator.generate_data(n=n)


def generate_case2_exponential(n: int = 300, seed: int = 42) -> dict:
    """
    Generate Case 2 exponential function data.
    
    Args:
        n: Total number of samples (300 or 500)
        seed: Random seed
        
    Returns:
        Dictionary with train/test data
    """
    generator = Case2_Exponential(seed=seed)
    return generator.generate_data(n=n)


# Convenience alias for the 1D Case 2 generator used by run_simulation.py.
def generate_case2_1d(n: int = 300, seed: int = 42) -> dict:
    """
    Generate Case 2 1D piecewise-function data.

    This wraps generate_case2_piecewise so callers can use the same
    case/dimension naming pattern as generate_case1_1d.
    """
    return generate_case2_piecewise(n=n, seed=seed)


# Convenience alias for the 2D Case 2 generator used by run_simulation.py.
def generate_case2_2d(n: int = 300, seed: int = 42) -> dict:
    """
    Generate Case 2 2D exponential-function data.

    This wraps generate_case2_exponential so callers can use the same
    case/dimension naming pattern as generate_case1_2d.
    """
    return generate_case2_exponential(n=n, seed=seed)


if __name__ == "__main__":
    # Example: Generate all cases
    print("=" * 60)
    print("Data Generation Examples")
    print("=" * 60)
    
    # Case 1: 1D (n=350)
    print("\nCase 1 - Polynomial Chaos (1D, n=350):")
    data_1d_350 = generate_case1_1d(n=350, seed=42)
    print(f"  Training samples: {data_1d_350['n_train']}")
    print(f"  Test samples: {data_1d_350['n_test']}")
    print(f"  X_train shape: {data_1d_350['X_train'].shape}")
    print(f"  z_train shape: {data_1d_350['z_train'].shape}")
    
    # Case 1: 1D (n=600)
    print("\nCase 1 - Polynomial Chaos (1D, n=600):")
    data_1d_600 = generate_case1_1d(n=600, seed=42)
    print(f"  Training samples: {data_1d_600['n_train']}")
    print(f"  Test samples: {data_1d_600['n_test']}")
    
    # Case 1: 2D (n=350)
    print("\nCase 1 - Polynomial Chaos (2D, n=350):")
    data_2d_350 = generate_case1_2d(n=350, seed=42)
    print(f"  Training samples: {data_2d_350['n_train']}")
    print(f"  Test samples: {data_2d_350['n_test']}")
    print(f"  X_train shape: {data_2d_350['X_train'].shape}")
    print(f"  z_train shape: {data_2d_350['z_train'].shape}")
    
    # Case 1: 2D (n=600)
    print("\nCase 1 - Polynomial Chaos (2D, n=600):")
    data_2d_600 = generate_case1_2d(n=600, seed=42)
    print(f"  Training samples: {data_2d_600['n_train']}")
    print(f"  Test samples: {data_2d_600['n_test']}")
    
    # Case 2: Piecewise (n=300)
    print("\nCase 2 - Piecewise Function (1D, n=300):")
    data_piecewise_300 = generate_case2_piecewise(n=300, seed=42)
    print(f"  Training samples: {data_piecewise_300['n_train']}")
    print(f"  Test samples: {data_piecewise_300['n_test']}")
    
    # Case 2: Piecewise (n=500)
    print("\nCase 2 - Piecewise Function (1D, n=500):")
    data_piecewise_500 = generate_case2_piecewise(n=500, seed=42)
    print(f"  Training samples: {data_piecewise_500['n_train']}")
    print(f"  Test samples: {data_piecewise_500['n_test']}")
    
    # Case 2: Exponential (n=300)
    print("\nCase 2 - Exponential Function (2D, n=300):")
    data_exp_300 = generate_case2_exponential(n=300, seed=42)
    print(f"  Training samples: {data_exp_300['n_train']}")
    print(f"  Test samples: {data_exp_300['n_test']}")
    print(f"  z_train shape: {data_exp_300['z_train'].shape}")
    
    # Case 2: Exponential (n=500)
    print("\nCase 2 - Exponential Function (2D, n=500):")
    data_exp_500 = generate_case2_exponential(n=500, seed=42)
    print(f"  Training samples: {data_exp_500['n_train']}")
    print(f"  Test samples: {data_exp_500['n_test']}")
    
    print("\n" + "=" * 60)
    print("All data generation scenarios completed successfully!")
    print("=" * 60)
