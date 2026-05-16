# 🚀 Quick Start Guide

Get up and running with the A Fully Bayesian Framework for Built-in Input Dimension Reduction for Gaussian Processes Modeling

## 📋 Prerequisites

- NumPy, SciPy, Matplotlib, Pandas
- Optional: TensorFlow (for automatic differentiation)
- Optional: Jupyter (for interactive notebooks)

## ⚡ Installation

```bash
# Navigate to the repository
cd github_results

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start (3 Steps)

### Step 1: Choose Your Interface

**Option A: Jupyter Notebook (Recommended for Beginners)**
```bash
jupyter notebook run_multichains.ipynb
```

**Option B: Python Script**
```python
from run_multichains import run_multichain_analysis
```

**Option C: Experiment Runners (Command Line Batch Runs)**

Use the experiment runners when you want the repository to handle data loading,
initialization, posterior sampling, diagnostics, metrics, plots, and summary
files for many combinations automatically.

```bash
# Synthetic simulations from Data_generation.py
python "Data Generation/run_simulation.py" --help

# Real application datasets
python "Application_Data/run_application.py" --help
```

The runner scripts internally call `run_multichain_analysis`. You do not need
to manually create a sampler or follow the direct Python examples below when
using these command-line runners.

### Step 2: Select Your Model

Choose one of these configurations:

#### **D=1, Layer 1 (Simplest)**
```python
from multichain_sampler_D1 import MultiChainSampler

sampler = MultiChainSampler(
    n_chains=2,
    layer=1,
    n_iterations=1000,  # Adjust as needed
    burn_in=200,
    thin=1,
    kernel_type='isotropic_squared_exponential'
)
```

#### **D>1, Layer 1 (Multi-dimensional)**
```python
from multichain_sampler_Dgeneral import MultiChainSampler

sampler = MultiChainSampler(
    n_chains=2,
    layer=1,
    D=2,  # Reduced dimension
    n_iterations=1000,
    burn_in=200,
    thin=1,
    kernel_type='separable_squared_exponential'
)
```

#### **Layer 2 or 3 (Deep GP)**
```python
# For Layer 2
sampler = MultiChainSampler(
    n_chains=2,
    layer=2,  # or layer=3
    D=1,  # or D>1
    n_iterations=1000,
    burn_in=200,
    thin=1,
    kernel_type='isotropic_squared_exponential'
)
```

### Step 3: Run Your Analysis

```python
# Prepare your data
import numpy as np
np.random.seed(42)

# Example: Generate synthetic data
n_train, n_test, p = 50, 20, 10
X_train = np.random.randn(n_train, p)
Y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(n_train)
X_test = np.random.randn(n_test, p)
Y_test = np.sin(X_test[:, 0]) + 0.1 * np.random.randn(n_test)

# Run the sampler
results = sampler.run_chains(
    Y_train, X_train,
    Y_test=Y_test,
    X_test=X_test,
    verbose=True
)

# Access results
samples = results['chains_samples']
metrics = results['metrics_summary']
convergence = results['convergence']
```

## 📚 Common Use Cases

### Use Case 1: Basic Analysis (D=1, Layer 1)

```python
from multichain_sampler_D1 import MultiChainSampler
import numpy as np

# Generate or load your data
X_train = np.random.randn(50, 10)
Y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(50)
X_test = np.random.randn(20, 10)
Y_test = np.sin(X_test[:, 0]) + 0.1 * np.random.randn(20)

# Create sampler
sampler = MultiChainSampler(
    n_chains=3,
    layer=1,
    n_iterations=2000,
    burn_in=500,
    thin=1,
    kernel_type='isotropic_squared_exponential'
)

# Run
results = sampler.run_chains(Y_train, X_train, Y_test, X_test, verbose=True)

# Check diagnostics
print(results['convergence'])
```

### Use Case 2: Experiment Runners

Use `run_simulation.py` for synthetic Case 1/Case 2 experiments:

```bash
python "Data Generation/run_simulation.py" \
  --sample-size 200 \
  --data-cases case1 case2 \
  --data-dimensions 1 2 \
  --posterior-dimensions 1 2 \
  --layers 1 2 3 \
  --w-variants full W_Known No_W No_W_Selective \
  --n-chains 3 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --output-dir ./simulation_outputs
```

Use `run_application.py` for the real application datasets:

```bash
python "Application_Data/run_application.py" \
  --applications elliptical_pde onera \
  --elliptical-outputs 1 2 \
  --onera-targets lift drag \
  --posterior-dimensions 1 2 \
  --layers 1 2 3 \
  --variants full No_W No_W_Selective \
  --n-chains 3 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --output-dir ./application_outputs
```

For smoke tests, add `--no-plots --no-save-samples` and reduce
`--n-iterations`; metrics are still computed and written to the summary files.
For full BDR models, add `--mv-sampler rstiefel --rstiefel-rscol 2` to use the
R `rstiefel` backend for posterior `M` and `V` updates.

### Use Case 3: Using MLE for Hyperparameters

```python
sampler = MultiChainSampler(
    n_chains=2,
    layer=1,
    n_iterations=1000,
    burn_in=200,
    thin=1,
    use_mle_tau2=True,   # MLE for tau²
    use_mle_g=True,      # MLE for nugget
    use_mle_theta=True,  # MLE for lengthscale
    kernel_type='isotropic_squared_exponential'
)
```

### Use Case 4: Layer Variants (W Known, No W, etc.)

```python
from multichain_sampler_L1_variants import MultiChainSampler_L1_Variants

# W is known
W_known = np.random.randn(10, 1)
W_known = W_known / np.linalg.norm(W_known)

sampler = MultiChainSampler_L1_Variants(
    variant='W_Known',
    W_fixed=W_known,
    n_chains=2,
    n_iterations=1000,
    burn_in=200,
    thin=1,
    kernel_type='isotropic_squared_exponential'
)
```

### Use Case 5: D>1 with Separable Kernel

```python
from multichain_sampler_Dgeneral import MultiChainSampler

sampler = MultiChainSampler(
    n_chains=2,
    layer=1,
    D=3,  # Reduced dimension
    n_iterations=1000,
    burn_in=200,
    thin=1,
    kernel_type='separable_squared_exponential'  # or 'separable_matern32'
)
```

## 🔧 Configuration Options

### MCMC Parameters
- `n_iterations`: Total MCMC iterations
- `burn_in`: Number of burn-in samples to discard
- `thin`: Thinning interval (keep every `thin`-th sample)
- `n_chains`: Number of independent chains

### Core Model Choices
- `D`: Any integer `>= 1` (presets available for `1`, `2`, `3`, `5`)
- `layer`: `1`, `2`, or `3`
- `variant`: `None` (full model), `'W_Known'`, `'No_W'`, `'No_W_Selective'` (for layers 1/2/3)
- `W_fixed`: Required when `variant='W_Known'`
- `column_indices`: Optional when `variant='No_W_Selective'` (`None` uses first `D` columns)

### Experiment Runner Options
- Synthetic runner: `Data Generation/run_simulation.py`
- Application runner: `Application_Data/run_application.py`
- `--posterior-dimensions`: one or more reduced dimensions `D`
- `--layers`: one or more model depths, selected from `1`, `2`, and `3`
- `--w-variants`: simulation-only selector for `full`, `W_Known`, `No_W`, `No_W_Selective`
- `--include-w-variants`: simulation-only shortcut for all simulation variants
- `--variants`: application-only selector for `full`, `No_W`, `No_W_Selective`
- `--no-plots`: skip plot PDFs; metrics are still computed
- `--no-save-samples`: skip `mcmc_samples.pkl`; summaries and metrics are still written
- `--mv-sampler rstiefel`: optional R `rstiefel` backend for posterior `M` and `V` in full models
- `--rstiefel-rscol`: optional `rscol` argument for `rstiefel::rmf.matrix.gibbs` when `D > 1`

### Kernel Types
- `'isotropic_squared_exponential'`: Isotropic SE kernel (D=1)
- `'separable_squared_exponential'`: Separable SE kernel (D>1)
- `'isotropic_matern32'`: Isotropic Matérn-3/2 (D=1)
- `'separable_matern32'`: Separable Matérn-3/2 (D>1)

### Estimation Flags
- Layer 1 full model: `use_mle_tau2`, `use_mle_g`, `use_mle_theta`, `use_mle_all`
- Layer 2/3 full model: `use_mle_tau2`, `use_mle_g_y`, `use_mle_theta_y`, `use_mle_all`
- Variants Layer 1: `use_mle_tau2`, `use_mle_g`, `use_mle_theta`
- Variants Layer 2/3: `use_mle_tau2`, `use_mle_g_y`, `use_mle_theta_y`

### HMC/Gradient Controls
- `eps_hmc`: Step size (default in interface: `0.09`)
- `T_step_hmc`: Number of leapfrog steps (default in interface: `15`)
- `M_hmc`: HMC proposals per iteration (default: `1`)
- `use_tf_gradients`: Use TensorFlow gradients (typically helpful for `D>1`)

### Optional Post-Sampling Parameter Diagnostics (R/coda)
- `compute_parameter_diagnostics`: `True`/`False`
- `diagnostics_burn`, `diagnostics_ci`, `diagnostics_use_projection_for_W`
- `diagnostics_r_home`, `diagnostics_parameters`

### Important Interface Note
- Preset/helper configs (`create_config_*`, `get_config_for`) are the safest way to run.
- The `run_multichain_analysis` signature includes advanced prior/HMC/initialization arguments for API compatibility.
- Layer-aware initialization controls are wired through the multichain wrappers:
  `W_init`, `M_init`, `V_init`, `Lambda_init`,
  `tau2_y_init`, `tau2_q_init`, `tau2_r_init`,
  `g_y_init`, `g_q_init`, `g_r_init`,
  `theta_y_init`, `theta_q_init`, `theta_r_init`, `Q_init`, `R_init`.

## 📊 Understanding Results

### Sample Structure
```python
results = sampler.run_chains(Y_train, X_train, Y_test, X_test)

# Access samples from chain 0
chain_0 = results['chains_samples'][0]
print(chain_0.keys())  # keys depend on layer/model variant

# Common full-model layer-1 keys:
tau2_y_samples = chain_0['tau2_y']
g_y_samples = chain_0['g_y']
theta_D_y_samples = chain_0['theta_D_y']
W_samples = chain_0['W']
```

### Metrics
```python
metrics = results['metrics_summary']

# Performance metrics
rmspe = metrics['rmspe']['mean']
nsme = metrics['nsme']['mean']
crps = metrics['crps']['mean']
bic = metrics['bic']['mean']

# Diagnostics
convergence = results['convergence']
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test all cases (2 samples each for quick verification)
python test_all_cases.py

# Verify 2-sample configuration
python verify_2_samples.py

# Test data generation
python test_data_generation.py
```

## 📖 Next Steps

1. **Read the main README.md** for detailed documentation
2. **Explore run_multichains.ipynb** for interactive examples
3. **Check folder-specific READMEs**:
   - `Data Generation/README.md` - Data generation
   - `Application_Data/README.md` - Application-data runner
   - `Covariance Functions/README.md` - Kernel functions
   - `Parameter Sampler/README.md` - MCMC sampling
   - `Gibbs Sampling/README.md` - Layer-wise sampling
   - `Multichain/README.md` - Multi-chain diagnostics
   - `BDR Metrics and Plot/README.md` - Metrics and plots

## ❓ Troubleshooting

### Import Errors
```python
# Make sure you're in the github_results directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Memory Issues
- Reduce `n_iterations`
- Increase `thin` to store fewer samples
- Use MLE for some hyperparameters to reduce sampling

### Convergence Issues
- Increase `n_iterations` and `burn_in`
- Check `results['convergence']` (`r_hat_*` values should be close to 1, typically `< 1.1`)
- Try different initialization

## 🎓 Example Workflow

```python
# 1. Generate or load data
from Data_generation import generate_case1_1d
data = generate_case1_1d(n=350, seed=42)
X_train, Y_train = data['X_train'], data['y_train']
X_test, Y_test = data['X_test'], data['y_test']

# 2. Create sampler
from multichain_sampler_D1 import MultiChainSampler
sampler = MultiChainSampler(
    n_chains=3,
    layer=1,
    n_iterations=2000,
    burn_in=500,
    thin=1
)

# 3. Run analysis
results = sampler.run_chains(Y_train, X_train, Y_test, X_test, verbose=True)

# 4. Check diagnostics
print("Convergence keys:", results['convergence'].keys())

# 5. Access posterior samples
tau2_mean = np.mean(results['chains_samples'][0]['tau2_y'])

# 6. View metrics
print("RMSPE:", results['metrics_summary']['rmspe']['mean'])
```

## 📝 Notes

- **Default Settings**: Conservative defaults for stability
- **Performance**: Use MLE for faster convergence when appropriate
- **Reproducibility**: Set random seeds for reproducible results

## 🔗 Quick Links

- **Main README**: `README.md`
- **Notebook Interface**: `run_multichains.ipynb`
- **Script Interface**: `run_multichains.py`
- **Synthetic Runner**: `Data Generation/run_simulation.py`
- **Application Runner**: `Application_Data/run_application.py`
- **Examples**: `example_usage.py` (use `--mode data|mle|all`), with `example_mle_options.py` kept as a compatibility wrapper

## 📚 Citation

If you use this repository, please cite the JUQ paper:

```bibtex
@article{gyamfi2026bdr,
  title   = {A Fully Bayesian Framework for Built-in Input Dimension Reduction for Gaussian Process Modeling},
  author  = {Gyamfi, Eric Herrison and Kang, Emily L. and Konomi, Bledar A. and Lin, Guang},
  journal = {Journal on Uncertainty Quantification},
  year    = {2026},
  note    = {Under Review}
}
```

---

**Ready to go?** Start with `run_multichains.ipynb` for the easiest experience!
