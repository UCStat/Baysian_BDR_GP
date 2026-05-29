# A Fully Bayesian Framework for Built-in Input Dimension Reduction for Gaussian Process Modeling

A comprehensive Python framework for A Fully Bayesian Framework for Built-in Input Dimension Reduction and Gaussian Process Modeling with full MCMC inference.



## 📁 Repository Structure

```

├── Data Generation/              # Synthetic data generation
│   ├── Data_generation.py
│   ├── __init__.py
│   └── README.md
│
├── Application_Data/             # Real application datasets
│   ├── Elliptical_PDE/
│   ├── Onera M6/
│   └── README.md
│
├── Scripts/                      # Command-line experiment runners
│   ├── run_simulation.py
│   ├── run_application.py
│   └── README.md
│
├── Run_Example/                  # Ready-to-edit example run scripts
│   ├── run_one_case.py
│   └── README.md
│
├── Covariance Functions/         # Kernel implementations  
│   ├── covariance_kernel_functions_and_gradients_W.py
│   ├── __init__.py
│   └── README.md
│
├── Parameter Sampler/            # MCMC sampling functions
│   ├── parameter_sampler_D1.py
│   ├── parameter_sampler_Dgeneral.py
│   ├── __init__.py
│   └── README.md
│
├── Gibbs Sampling/               # Layer-wise Gibbs samplers
│   ├── gibbs_sampler_layers_D1.py
│   ├── gibbs_sampler_layers_Dgeneral.py
│   ├── __init__.py
│   └── README.md
│
├── Multichain/                   # Multi-chain with diagnostics
│   ├── multichain_sampler_D1.py
│   ├── multichain_sampler_Dgeneral.py
│   ├── __init__.py
│   └── README.md
│
├── BDR Metrics and Plot/         # Performance & visualization
│   ├── BDR_metrics.py
│   ├── BDR_plot.py
│   ├── __init__.py
│   └── README.md
│
├── run_multichains.py            # Main interface (full + preset configuration API)
├── run_multichains.ipynb         # Jupyter notebook interface
│
├── requirements.txt              # Dependencies
├── QUICKSTART.md                 # Quick start guide
├── SAMPLING_README.md            # Sampling details
├── DGENERAL_README.md            # D>1 documentation
├── KERNELS_README.md             # Kernel documentation

```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd github_results

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

#### Paper-Style Replication

Use the ready-to-edit Case 1a wrapper when you want to reproduce the paper-style
simulation outputs first:

```bash
python3 Run_Example/run_one_case.py
```

This calls `Scripts/run_simulation.py` with Case 1, one-dimensional input
subspace, layers 1/2/3, `full`, `No_W_Selective`, and `W_Known` variants, and
the MCMC settings used in the example file. Outputs are written under
`simulation_outputs/`, including run folders, posterior summaries,
time-complexity summaries, diagnostic plots, metric comparison tables, and
`metric_boxplots_by_layer/`.

To check the wrapper without launching the full 2000-iteration run:

```bash
python3 -m py_compile Run_Example/run_one_case.py
```

Edit `SAMPLE_SIZE`, `DATA_CASES`, and `DATA_DIMENSIONS` in
`Run_Example/run_one_case.py` to run Case 1b, Case 2a, or Case 2b. See
`Run_Example/README.md` for the exact settings and for application-data run
examples.

#### Other Option 1: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook run_multichains.ipynb
```

The notebook provides:
- Step-by-step instructions
- Interactive examples
- Visual explanations
- All preset configurations

**Follow the notebook cells in order:**
1. Run the import cell
2. Review initialization examples
3. Choose a configuration method
4. Run your analysis
5. Review results

#### Other Option 2: Direct Python API

```python
from run_multichains import run_multichain_analysis
from Data_generation import generate_case1_1d

# Generate data
data = generate_case1_1d(n=200, seed=42)

# Run complete analysis
results = run_multichain_analysis(
    Y_train=data['y_train'],
    X_train=data['X_train'],
    Y_test=data['y_test'],
    X_test=data['X_test'],
    D=1,              # Reduced dimension
    layer=1,          # 1-layer GP
    n_chains=3,       # Number of chains
    n_iterations=2000,
    burn_in=500,
    use_mle_all=True, # Fast MLE mode
    output_dir='./diagnostics',
    verbose=True
)

# Access results
print(results['convergence'])        # R-hat, Heidelberg-Welch
print(results['metrics_summary'])    # RMSPE, NSME, CRPS, BIC, MLPPD, CP, ALCI, Score
```

#### Other Option 3: Direct Experiment Runners

Use the two runner scripts when you want complete repeated experiments rather
than a single direct `run_multichain_analysis` call. Both runners use the same
posterior sampling, initialization, diagnostics, metrics, plots, and summary
outputs.

The runner scripts internally call `run_multichain_analysis`, so you do not
need to follow the Quick Start Guide when using them from the command line. Use
this Experiment Runners section for batch command-line runs; use the Quick
Start Guide below when you want to call `run_multichain_analysis` directly from
Python with your own prepared arrays and configuration.

| Script | Data source | Kernel | W variants | Summary file |
| --- | --- | --- | --- | --- |
| `Scripts/run_simulation.py` | Synthetic Case 1/Case 2 from `Data_generation.py` | `isotropic_squared_exponential` by default; choose with `--kernel-type` | choose `full`, `W_Known`, `No_W`, `No_W_Selective` with `--w-variants`; `--include-w-variants` is a shortcut for all | `simulation_summary.csv` |
| `Scripts/run_application.py` | Real `Elliptical_PDE` and `Onera M6` data | `separable_matern32` for Elliptical_PDE; `separable_squared_exponential` for Onera | `full`, `No_W`, `No_W_Selective`; no `W_Known` because true `W` is unavailable | `application_summary.csv` |

Both runners expose `--posterior-dimensions`, `--variant-dimensions`,
`--layers`, `--n-chains`, `--n-iterations`, `--burn-in`, `--thin`,
`--mv-sampler`, `--no-plots`, and `--no-save-samples`. Defaults are small
enough for smoke tests; use larger iteration counts for real experiments. The
scripts use `seed=42` and `thin=3` by default.

Important shared options:

- `--posterior-dimensions`: reduced posterior dimension(s) `D` used by the BDR sampler.
- `--variant-dimensions`: optional per-variant D grid using `VARIANT=D[,D...]`; variants not listed use `--posterior-dimensions`.
- `--layers`: model depth(s), selected from `1`, `2`, and `3`.
- `--kernel-type`: simulation-only covariance kernel. Choices are `isotropic_squared_exponential`, `separable_squared_exponential`, `isotropic_matern32`, and `separable_matern32`.
- `--n-chains`: number of MCMC chains.
- `--n-iterations`: MCMC iterations per chain.
- `--burn-in`: number of iterations discarded before summaries.
- `--thin`: thinning interval for saved posterior samples.
- `--mv-sampler`: `python` by default, or `rstiefel` for posterior `M` and `V` updates in full models.
- `--rstiefel-rscol`: optional number of columns updated together by `rstiefel::rmf.matrix.gibbs` for D > 1.
- `--no-plots`: skips diagnostic and metric plot files; metrics are still computed.
- `--no-save-samples`: skips `mcmc_samples.pkl`; summaries and metrics are still written.
- `--parameter-diagnostics`: attempts extra R/coda diagnostics.
- `--continue-on-error`: keeps running later combinations if one run fails.
- `--verbose`: prints sampler progress.

Simulation-only options:

- `--sample-size`: total synthetic observations before the train/test split.
- `--data-cases`: synthetic generator family, `case1` and/or `case2`.
- `--data-dimensions`: true synthetic generator dimension, currently `1` and/or `2`.
- `--w-variants`: choose specific simulation variants from `full`, `W_Known`, `No_W`, and `No_W_Selective`.
- Simulation `--variant-dimensions` accepts `W_known` as an alias for `W_Known`, for example `full=1,2,3 No_W_Selective=1,2,3 W_known=1,2`.
- `--include-w-variants`: shortcut flag that runs `full`, `W_Known`, `No_W`, and `No_W_Selective`.

Application-only options:

- `--applications`: choose `elliptical_pde`, `onera`, or both.
- `--elliptical-outputs`: choose Elliptical_PDE output pairs `1` and/or `2`; output `1` uses `X_1.npy` and `Y_1.npy`.
- `--onera-targets`: choose Onera response `lift`, `drag`, or both.
- `--variants`: choose `full`, `No_W`, and/or `No_W_Selective`.
- Application `--variant-dimensions` example: `full=1,2,3 No_W_Selective=1,2,3`.
- `--train-fraction`: training fraction for the shuffled split; default is `0.8`.
- `--max-rows`: optional row cap for smoke tests before the split.

Synthetic simulation example:

```bash
python "Scripts/run_simulation.py" \
  --sample-size 200 \
  --data-cases case1 case2 \
  --data-dimensions 1 2 \
  --posterior-dimensions 1 2 \
  --layers 1 2 3 \
  --kernel-type isotropic_squared_exponential \
  --n-chains 3 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --w-variants full W_Known No_W No_W_Selective \
  --output-dir ./simulation_outputs
```

Application-data example:

```bash
python "Scripts/run_application.py" \
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

Per-variant D grids can be supplied to either runner. For example, simulation
runs can use `--variant-dimensions full=1,2,3 No_W_Selective=1,2,3 W_known=1,2`,
while application runs can use
`--variant-dimensions full=1,2,3 No_W_Selective=1,2,3`.

Use `--mv-sampler rstiefel --rstiefel-rscol 2` with either runner to sample
posterior `M` and `V` through the R `rstiefel` backend. Each run folder writes
`config_used.json`, `initial_values_and_priors.npz`, `results_summary.json`,
posterior and time-complexity summary tables, optional posterior samples,
optional diagnostic/metric plots, top-level metric comparison tables, and
per-layer metric model boxplots under `metric_boxplots_by_layer/`. Per-metric
y-axis limits are shared across layers in those boxplots, and very small metric
values use a scientific y-axis multiplier. ALCI, Score, BIC, and MLPPD use a
symmetric-log y-axis scale; MLPPD also gets an additional unscaled linear plot.
The same folder also includes grouped metric boxplots with layer 1, 2, and 3
side by side in one figure. More detailed runner notes live in
`Data Generation/README.md` and
`Application_Data/README.md`.

### Quick Start Guide

#### Step 1: Choose Your Configuration Method

**Method A: Preset Configurations (Easiest)**
```python
from run_multichains import create_config_D1_L1, run_multichain_analysis

# Create config with automatic initialization
config = create_config_D1_L1(p=10, seed=42, n_iterations=2000)

# Run analysis
results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    **config
)
```

**Method B: Helper Function**
```python
from run_multichains import get_config_for, run_multichain_analysis

# Get config with overrides
config = get_config_for(D=2, layer=2, n_iterations=1000, use_mle_all=True)

# Run analysis
results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    **config
)
```

**Method C: Manual Configuration**
```python
# Specify all parameters directly
results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    D=1, layer=1,
    n_chains=3, n_iterations=2000, burn_in=500,
    use_mle_all=True,
    output_dir='./results'
)
```

#### Step 2: Prepare Your Data

Your data should be:
- **X_train**: (n, p) - Training inputs
- **Y_train**: (n,) - Training responses
- **X_test**: (n_test, p) - Test inputs
- **Y_test**: (n_test,) - Test responses

Or use the provided data generation:
```python
from Data_generation import generate_case1_1d, generate_case1_2d

# For D=1
data = generate_case1_1d(n=200, seed=42)

# For D>1
data = generate_case1_2d(n=200, seed=42)
```

#### Step 3: Run Analysis

```python
results = run_multichain_analysis(
    Y_train=data['y_train'],
    X_train=data['X_train'],
    Y_test=data['y_test'],
    X_test=data['X_test'],
    **config  # Your configuration
)
```

#### Step 4: Check Results

```python
# Convergence diagnostics
print("R-hat values:")
for param, rhat in results['convergence'].items():
    print(f"  {param}: {rhat:.4f} {'✓' if rhat < 1.1 else '⚠'}")

# Performance metrics
print("\nPerformance Metrics:")
for metric, stats in results['metrics_summary'].items():
    print(f"  {metric}: {stats['mean']:.4f} (95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")
```

#### Step 5: Review Diagnostic Plots

Check the `output_dir` directory for:
- Trace plots (convergence)
- Density plots (posterior distributions)
- Autocorrelation plots (mixing)
- Actual vs Predicted plots (model fit)
- Convergence diagnostics table

See `run_multichains.ipynb` for detailed step-by-step instructions with examples!

## 🎯 Features

### Model Architectures

- **1-Layer GP:** X → Z → Y
- **2-Layer Deep GP:** X → Z → Q → Y
- **3-Layer Deep GP:** X → Z → R → Q → Y

### Reduced Dimensions

- **D=1:** Single latent dimension
- **D>1:** Multi-dimensional (D=2, 3, 5, ...)

### Covariance Kernels

1. Isotropic Squared Exponential
2. Separable Squared Exponential
3. Isotropic Matérn-3/2
4. Separable Matérn-3/2

All with **NumPy** and **TensorFlow** gradient implementations!

### Parameter Estimation

- **Full MCMC:** Sample all parameters
- **Hybrid MLE-MCMC:** MLE for hyperparameters, MCMC for others
- **All MLE:** Fastest option

### Sampling Algorithms

- **τ²:** Gibbs sampling (Inverse-Gamma posterior)
- **g, θ:** Metropolis-Hastings (Gamma priors)
- **W:** HMC on Stiefel manifold with geodesic flows
- **M, V:** Conjugate updates
- **Λ:** Elliptical slice sampling
- **Q, R:** Latent variable sampling for Deep GPs

### Convergence Diagnostics

- **Gelman-Rubin (R̂):** < 1.1 indicates convergence
- **Heidelberg-Welch:** Stationarity and halfwidth tests

### Performance Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| RMSPE | Root Mean Square Predictive Error | Lower |
| NSME | Nash-Sutcliffe Model Efficiency | Higher (max 1) |
| CRPS | Continuous Ranked Probability Score | Lower |
| BIC | Bayesian Information Criterion | Higher |
| MLPPD | Mean Log Pointwise Predictive Density | Higher |
| CP | Coverage Probability | Closer to target coverage |
| ALCI | Average Length of Credible Intervals | Lower for same coverage |
| Score | Predictive Log-Likelihood | Higher |

All metrics include: **mean, median, std, 95% credible intervals**

### Diagnostic Plots

1. Trace plots
2. Autocorrelation plots
3. W parameter traces (multi-chain)
4. Actual vs. Predicted
5. Convergence diagnostics table
6. Metrics boxplots
7. Metrics comparison table (`GP (D) BDR` labels for full layer-1 BDR runs; `DGP 2-layer (D) BDR` and `DGP 3-layer (D) BDR` labels for full deeper BDR runs; Oracle labels for `W_Known` runs; `W/o` labels for `No_W_Selective` runs)
8. Single-layer RMSPE boxplot by dimension or method
9. Grouped layer RMSPE boxplot by dimension or method

## 📖 Documentation

| File | Description |
|------|-------------|
| [SAMPLING_README.md](SAMPLING_README.md) | Detailed sampling algorithms |
| [DGENERAL_README.md](DGENERAL_README.md) | D>1 specific documentation |
| [KERNELS_README.md](KERNELS_README.md) | Covariance kernel details |


**Folder READMEs:**
- [Run_Example/README.md](Run_Example/README.md)
- [Scripts/README.md](Scripts/README.md)
- [Application_Data/README.md](Application_Data/README.md)
- [Data Generation/README.md](Data%20Generation/README.md)
- [Covariance Functions/README.md](Covariance%20Functions/README.md)
- [Parameter Sampler/README.md](Parameter%20Sampler/README.md)
- [Gibbs Sampling/README.md](Gibbs%20Sampling/README.md)
- [Multichain/README.md](Multichain/README.md)
- [BDR Metrics and Plot/README.md](BDR%20Metrics%20and%20Plot/README.md)

## 📖 Detailed Usage Guide

### Understanding the Framework

The framework supports three types of models:

1. **Full Models**: Sample all parameters including W, M, Lambda, V
   - Use when: You need to estimate the projection matrix W
   - Example: `create_config_D1_L1()`, `create_config_D2_L2()`, etc.

2. **Layer Variants**: Skip W, M, Lambda, V sampling
   - Use when: W is known, you want all columns of `X`, or you want fixed
     selected columns of `X`
   - Example: `create_config_L1_W_Known()`, `create_config_L2_No_W()`, etc.

3. **Custom Models**: Full control over all parameters
   - Use when: You need specific hyperparameter settings
   - Example: Manual parameter specification

### Configuration Methods Explained

#### Method 1: Preset Configurations (Recommended)

**Advantages:**
- Easiest to use
- Sensible defaults
- Automatic initialization when `p` is provided
- Type-safe parameter names

**Example:**
```python
# D=1, Layer 1 with automatic initialization
config = create_config_D1_L1(
    p=10,              # Input dimension (triggers auto-init)
    seed=42,           # Random seed
    n_iterations=2000, # Override default
    use_mle_all=True   # Override default
)
```

**Available Presets:**
- `create_config_D1_L1/L2/L3(p, seed, **kwargs)`
- `create_config_D2_L1/L2/L3(p, seed, **kwargs)`
- `create_config_D3_L1/L2/L3(p, seed, **kwargs)`
- `create_config_D5_L1/L2/L3(p, seed, **kwargs)`

#### Method 2: Helper Function

**Advantages:**
- Flexible for any D and layer
- Quick parameter overrides

**Example:**
```python
config = get_config_for(
    D=2,
    layer=2,
    n_iterations=1000,
    use_mle_all=True,
    use_tf_gradients=True
)
```

#### Method 3: Manual Configuration

**Advantages:**
- Complete control
- No hidden defaults

**Example:**
```python
results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    D=1, layer=1,
    n_chains=3,
    n_iterations=2000,
    burn_in=500,
    thin=2,
    # ... all other parameters
)
```

### Using Layer Variants

Layer variants are simplified models that skip W, M, Lambda, V sampling:

**When to use variants:**
- You have a known projection matrix W → Use `W_Known` variant
- You do not need dimensionality reduction and want all columns of `X` → Use `No_W` variant
- You want fixed column-selection reduction → Use `No_W_Selective` variant

**Example:**
```python
# Layer 1: W is known
W_fixed = np.random.randn(p, 2)
W_fixed, _ = np.linalg.qr(W_fixed)

config = create_config_L1_W_Known(
    W_fixed=W_fixed,
    n_iterations=1000,
    use_mle_tau2=True,
    use_mle_g=False,
    use_mle_theta=True
)

results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    layer=1,
    variant='W_Known',
    **config
)
```

## 🔧 Configuration

### All Configurable Parameters

Use [`run_multichain_analysis`](run_multichains.py) for the most complete interface.  
Recommended workflow is:
1. start from a preset (`create_config_D*_L*` or `get_config_for`)
2. override only what you need.

### Option Matrix (Valid Values)

- `D`: any integer `>= 1` (presets provided for `1`, `2`, `3`, `5`)
- `layer`: `1`, `2`, or `3`
- `variant`: `None` (full model), `'W_Known'`, `'No_W'`, `'No_W_Selective'` (supported for layers `1/2/3`)
- `kernel_type`: `'isotropic_squared_exponential'`, `'separable_squared_exponential'`, `'isotropic_matern32'`, `'separable_matern32'`
- `mv_sampler`: `'python'` (default) or `'rstiefel'` for posterior `M`/`V` updates in full models
- `rstiefel_rscol`: optional integer passed as `rscol` to `rstiefel::rmf.matrix.gibbs` when `mv_sampler='rstiefel'`

Variant-specific requirements:
- `'W_Known'`: must pass `W_fixed` with shape `(p, D)`
- `'No_W'`: uses all original input columns; changing `D` does not change the
  model input
- `'No_W_Selective'`: pass `D`; `column_indices` optional (`None` uses first `D` columns)
- Runner `--variant-dimensions` lets each selected variant use its own D grid.
- `Scripts/run_simulation.py --w-variants ...` chooses specific simulation variants; `--include-w-variants` is a shortcut for all variants.

MLE flag combinations:
- Full model, layer 1: `use_mle_tau2`, `use_mle_g`, `use_mle_theta`, `use_mle_all`
- Full model, layers 2/3: `use_mle_tau2`, `use_mle_g_y`, `use_mle_theta_y`, `use_mle_all`
- Variants, layer 1: `use_mle_tau2`, `use_mle_g`, `use_mle_theta`
- Variants, layers 2/3: `use_mle_tau2`, `use_mle_g_y`, `use_mle_theta_y`

### Forwarded vs Reserved Arguments

Currently forwarded in the multichain wrappers:
- model routing: `D`, `layer`, `variant`, `W_fixed`, `column_indices`
- sampling controls: `n_chains`, `n_iterations`, `burn_in`, `thin`
- M/V sampler controls: `mv_sampler`, `rstiefel_rscol`
- estimation flags: `use_mle_*`, `use_mle_all`, `use_tf_gradients`
- kernel/priors: `kernel_type`, `prior_M`, `prior_V`
- initialization: `W_init`, `M_init`, `V_init`, `Lambda_init`,
  `tau2_y_init`, `tau2_q_init`, `tau2_r_init`,
  `g_y_init`, `g_q_init`, `g_r_init`,
  `theta_y_init`, `theta_q_init`, `theta_r_init`, `Q_init`, `R_init`
- outputs/diagnostics: `output_dir`, `save_samples`, `save_plots`, `verbose`,
  `compute_parameter_diagnostics`, `diagnostics_*`

Accepted but currently reserved in full-model wrappers (API compatibility / future expansion):
- advanced HMC/prior/init knobs like `eps_hmc`, `T_step_hmc`, `M_hmc`,
  `alpha*`, `beta*`, `gamma*`, `l_*`, `u_*`, `nu_lambda`, `epsilon_lambda`,
  `max_iter_lambda`

Example call template:

```python
results = run_multichain_analysis(
    # Model
    D=1, layer=1,
    
    # MCMC
    n_chains=3, n_iterations=2000, burn_in=500, thin=3,
    
    # Estimation
    use_mle_all=False, use_tf_gradients=False,
    
    # HMC for W
    eps_hmc=0.001, T_step_hmc=17, M_hmc=1,
    
    # Priors: tau2
    alpha1_tau2=0.001, alpha2_tau2=0.001,
    
    # Priors: g (nugget)
    beta1_g=3/2, beta2_g=3.9,
    l_g=1.0, u_g=2.0,  # MH bounds
    
    # Priors: theta (lengthscale)
    gamma1_theta=0.01, gamma2_theta=0.01/3,
    l_theta=1.0, u_theta=2.0,  # MH bounds
    
    
    
    # Initialization (layer-aware)
    W_init=None, tau2_y_init=0.005, g_y_init=0.00009, theta_y_init=1.0,
    
    # Output
    output_dir='./diagnostics',
    save_samples=True,
    save_plots=True,
    verbose=True
)
```

## 📊 Complete Examples

### Example 1: D=1, 1-Layer (Simplest Case)

```python
from run_multichains import create_config_D1_L1, run_multichain_analysis
from Data_generation import generate_case1_1d

# Generate data
data = generate_case1_1d(n=200, seed=42)

# Create config with automatic initialization
config = create_config_D1_L1(
    p=data['X_train'].shape[1],
    seed=42,
    n_iterations=2000,
    use_mle_all=True
)

# Run analysis
results = run_multichain_analysis(
    Y_train=data['y_train'],
    X_train=data['X_train'],
    Y_test=data['y_test'],
    X_test=data['X_test'],
    **config
)

# Check results
print("Convergence:", results['convergence'])
print("RMSPE:", results['metrics_summary']['rmspe']['mean'])
```

### Example 2: D=2, 2-Layer Deep GP

```python
from run_multichains import create_config_D2_L2, run_multichain_analysis
from Data_generation import generate_case1_2d

# Generate data
data = generate_case1_2d(n=200, seed=42)

# Create config with automatic initialization
config = create_config_D2_L2(
    p=data['X_train'].shape[1],
    seed=42,
    n_iterations=2000,
    use_mle_all=True,
    use_tf_gradients=True  # Recommended for D>1
)

# Run analysis
results = run_multichain_analysis(
    Y_train=data['y_train'],
    X_train=data['X_train'],
    Y_test=data['y_test'],
    X_test=data['X_test'],
    **config
)
```

### Example 3: Using Layer Variants

```python
from run_multichains import create_config_L1_W_Known, run_multichain_analysis
import numpy as np

# Create a known W matrix
p, D = 10, 2
W_fixed = np.random.randn(p, D)
W_fixed, _ = np.linalg.qr(W_fixed)

# Create variant config
config = create_config_L1_W_Known(
    W_fixed=W_fixed,
    n_iterations=1000,
    use_mle_tau2=True,
    use_mle_g=False,
    use_mle_theta=True
)

# Run analysis
results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    layer=1,
    variant='W_Known',
    **config
)
```

### Example 4: Custom Hyperparameters

```python
from run_multichains import run_multichain_analysis

results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    D=1, layer=1,
    n_chains=3, n_iterations=2000,
    
    # Custom HMC settings
    eps_hmc=0.09, T_step_hmc=15,
    
    # Custom priors
    alpha1_tau2=0.001, alpha2_tau2=0.001,
    beta1_g=3/2, beta2_g=3.9,
    gamma1_theta=3/2, gamma2_theta=3.9,
    
    # Custom MH bounds
    l_g=1, u_g=2,
    l_theta=1, u_theta=2,
    
    output_dir='./custom'
)
```

### Example 5: 3-Layer Deep GP with Individual MLE Options

```python
from run_multichains import create_config_D1_L3, run_multichain_analysis

config = create_config_D1_L3(
    p=10, seed=42,
    n_iterations=2000,
    # Individual MLE flags for Y layer only
    use_mle_tau2=False,   # MLE for tau2_y
    use_mle_g_y=False,   # MCMC for g_y
    use_mle_theta_y=False # MLE for theta_y
    # Note: Q and R layers always use MCMC
)

results = run_multichain_analysis(
    Y_train=Y_train, X_train=X_train,
    Y_test=Y_test, X_test=X_test,
    **config
)
```

## 🎓 Step-by-Step Tutorial

### For Beginners: Start with the Notebook

1. **Open the notebook:**
   ```bash
   jupyter notebook run_multichains.ipynb
   ```

2. **Follow the cells in order:**
   - Cell 1: Import modules
   - Cell 2: Understand initialization
   - Cell 3: Run your first analysis
   - Continue through examples

3. **Modify examples:**
   - Change `D` and `layer` values
   - Adjust `n_iterations` and `burn_in`
   - Try different MLE options

### For Advanced Users: Use Python Scripts

1. **Import the main function:**
   ```python
   from run_multichains import run_multichain_analysis
   ```

2. **Choose configuration method:**
   - Preset configs (easiest)
   - Helper function (flexible)
   - Manual (full control)

3. **Run analysis:**
   ```python
   results = run_multichain_analysis(...)
   ```

4. **Analyze results:**
   - Check convergence diagnostics
   - Review performance metrics
   - Examine diagnostic plots

## ✅ Sanity Checks

```bash
# Check the example wrapper without running the full simulation
python3 -m py_compile Run_Example/run_one_case.py

# Confirm the command-line runners load and show their options
python3 "Scripts/run_simulation.py" --help
python3 "Scripts/run_application.py" --help

```

## 💡 Tips and Best Practices

1. **Start Simple:**
   - Begin with D=1, Layer=1
   - Use `use_mle_all=True` for quick results
   - Use preset configurations

2. **Check Convergence:**
   - R-hat < 1.1 indicates convergence
   - Increase `n_iterations` if not converged
   - Check trace plots for good mixing

3. **For D>1:**
   - Use `use_tf_gradients=True` (recommended)
   - Use separable or isotropic kernels
   - Provide `p` for automatic initialization

4. **For Multi-Layer Models:**
   - Layer 2: Q layer uses ESS, theta_q uses MCMC
   - Layer 3: R and Q layers use ESS, theta_r and theta_q use MCMC
 

5. **Performance:**
   - Variants are faster than full models
   - TensorFlow gradients help for D>1

6. **Reproducibility:**
   - Always set `seed` parameter
   - Save your configurations
   - Document your hyperparameter choices

## 📦 Dependencies

- numpy
- scipy
- tensorflow
- matplotlib
- seaborn
- pandas
- pyDOE

Install all: `pip install -r requirements.txt`




## 🤝 Contributing

This is a research codebase for the paper. For questions or issues, please contact gyamfien@mail.uc.edu.

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



**Last Updated:** April 2026
