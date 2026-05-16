# Data Generation Module

This module provides synthetic data generation for testing and validation of Bayesian Dimensionality Reduction models.

## Contents

- `Data_generation.py` - Main data generation module
- `run_simulation.py` - End-to-end multichain simulation runner
- `__init__.py` - Package initialization

## Available Scenarios

### Case 1: Polynomial Chaos Expansion

**1D Version (D=1):**
```python
from Data_generation import generate_case1_1d

data = generate_case1_1d(n=200, seed=42)
# Returns dictionary with:
#   - y_train, y_test: Response vectors
#   - X_train, X_test: Input matrices (n, p)
#   - z_train, z_test: Latent projections (n, D) where D=1
#   - W: True projection matrix (p, D)
```

**2D Version (D=2):**
```python
from Data_generation import generate_case1_2d

data = generate_case1_2d(n=200, seed=42)
# Returns dictionary with:
#   - y_train, y_test: Response vectors
#   - X_train, X_test: Input matrices (n, p)
#   - z_train, z_test: Latent projections (n, D) where D=2
#   - W: True projection matrix (p, D)
```

### Case 2: Piecewise Functions

```python
from Data_generation import generate_case2_piecewise

data = generate_case2_piecewise(n=200, seed=42)
```

### Case 3: Exponential Functions

```python
from Data_generation import generate_case2_exponential

data = generate_case2_exponential(n=200, seed=42)
```

## Usage

### Basic Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "Data Generation"))

from Data_generation import generate_case1_1d

# Generate data
data = generate_case1_1d(n=200, seed=42)

# Access components
Y_train = data['y_train']      # (n,)
X_train = data['X_train']      # (n, p)
Y_test = data['y_test']        # (n_test,)
X_test = data['X_test']        # (n_test, p)
Z_train = data['z_train']      # (n, D)
Z_test = data['z_test']        # (n_test, D)
W_true = data['W']             # (p, D)
```

### End-to-End Multichain Simulation Runner

`run_simulation.py` generates Case 1 or Case 2 data, builds initial values and prior
objects through the main configuration helpers, and runs three-chain posterior
sampling across 1-layer, 2-layer, and 3-layer models. It uses the isotropic
squared exponential kernel for every run, `seed=42`, and `thin=3`.

The data case, data-generation dimension, and posterior-sampling dimension are separate:

- `--data-cases`: generator case, currently `case1` or `case2`
- `--data-dimensions`: true generator dimension (`1` or `2`)
- `--posterior-dimensions`: reduced dimension `D` used by the BDR posterior sampler
- `--dimensions`: backward-compatible alias for `--posterior-dimensions`

`run_simulation.py` imports `generate_case1_1d`, `generate_case1_2d`,
`generate_case2_1d`, and `generate_case2_2d` from `Data_generation.py`.
The Case 2 names are convenience aliases for the existing Case 2 generators:
`generate_case2_1d` wraps the piecewise-function case, and
`generate_case2_2d` wraps the exponential-function case.

```bash
python "Data Generation/run_simulation.py"
```

#### Runner Options

Data selection:

- `--sample-size`: total number of generated observations before the train/test split.
- `--data-cases`: choose `case1`, `case2`, or both.
- `--data-dimensions`: choose true generator dimension `1`, `2`, or both.
- `--posterior-dimensions`: choose posterior sampler dimension(s) `D`.
- `--dimensions`: alias for `--posterior-dimensions`.

Model selection:

- `--layers`: model depth(s), selected from `1`, `2`, and `3`.
- `--w-variants`: choose specific variants from `full`, `W_Known`, `No_W`,
  and `No_W_Selective`.
- `--include-w-variants`: flag that adds `W_Known`, `No_W`, and
  `No_W_Selective` alongside the default full model.
- `full`: samples `W`, `M`, `V`, `Lambda`, and available GP hyperparameters.
- `W_Known`: fixes `W` to the true projection matrix from `Data_generation.py`.
- `No_W`: does not sample `W`; uses all original columns of `X`.
- `No_W_Selective`: does not sample `W`; uses the first `D` columns of `X`.

MCMC controls:

- `--n-chains`: number of MCMC chains.
- `--n-iterations`: MCMC iterations per chain.
- `--burn-in`: burn-in iterations discarded before summaries.
- `--thin`: thinning interval for saved posterior samples.
- `seed=42`: fixed internally by the runner for reproducibility.

M/V sampler controls:

- `--mv-sampler python`: default local Python sampler for posterior `M` and `V`.
- `--mv-sampler rstiefel`: uses the R `rstiefel` backend for posterior `M`
  and `V` in the `full` model.
- `--rstiefel-rscol`: optional `rscol` value for
  `rstiefel::rmf.matrix.gibbs` when `D > 1`.

Output controls:

- `--output-dir`: directory where run folders and `simulation_summary.csv`
  are written.
- `--no-plots`: skips diagnostic and metric plot files; metrics are still computed.
- `--no-save-samples`: skips `mcmc_samples.pkl`; summaries and metrics are
  still written.
- `--parameter-diagnostics`: attempts extra R/coda diagnostics.
- `--continue-on-error`: keeps running later combinations after one failure.
- `--verbose`: prints sampler progress.

By default, only the full BDR models are run. Use `--w-variants` when you want
to choose exactly which variants run. For example, this runs only `full` and
`No_W_Selective`:

```bash
python "Data Generation/run_simulation.py" \
  --w-variants full No_W_Selective
```

Add `--include-w-variants` as a shortcut when you want all W variants alongside
the full model:

- `W_Known`: fixes `W` at the true projection matrix returned by `Data_generation.py`
- `No_W`: omits `W` and uses the full input matrix directly
- `No_W_Selective`: omits `W` and uses the first `D` columns of `X`

Do not combine `--w-variants` and `--include-w-variants` in the same command.

All variants use the same `seed=42`, `thin=3`, isotropic squared exponential
kernel, initial kernel hyperparameters, and prior values as the full model.
For W-not-sampled variants, posterior samples contain the parameters that are
active in that variant; `W` is fixed or absent by construction.
Because `W_Known` uses the true generator `W`, it only runs when
`--posterior-dimensions` matches the selected `--data-dimensions`; mismatched
`W_Known` combinations are marked as skipped in `simulation_summary.csv`.

By default, the runner uses a small smoke-test setup. Increase the sample size
and MCMC iterations for real analysis:

```bash
python "Data Generation/run_simulation.py" \
  --sample-size 200 \
  --data-cases case1 case2 \
  --data-dimensions 1 2 \
  --posterior-dimensions 1 2 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --mv-sampler python \
  --w-variants full W_Known No_W No_W_Selective \
  --output-dir ./simulation_outputs
```

`--mv-sampler` controls posterior Gibbs updates for `M` and `V` in the full
BDR models:

- `python`: use the repository's local matrix-von-Mises-Fisher Gibbs sampler
  (default)
- `rstiefel`: call `rstiefel::rmf.matrix.gibbs(M, X)` through `rpy2`

When using `--mv-sampler rstiefel`, R, `rpy2`, and the R package `rstiefel`
must be installed. You can also pass `--rstiefel-rscol <k>` to forward the
optional `rscol` argument to `rmf.matrix.gibbs`.

The runner uses the SVD-based initialization from the paper: sample
`F_ij iid N(0,1)`, compute `F = M_full L V^T`, set
`M_init = M_full[:, :D]`, `Lambda_init = diag(L)`, `V_init = V`, then sample
`W_init ~ ML(M_init Lambda_init V_init^T)`, `F_M ~ ML(M_init)`, and
`F_V ~ ML(V_init)`. Here `D` is the posterior-sampling dimension from
`--posterior-dimensions`, not necessarily the generator dimension from
`--data-dimensions`. It uses `lambda ~ Gamma(5/2, 10/3)`, `l=1`, `u=2`,
`tau2 ~ InvGamma(0.001, 0.001)`, `g ~ Gamma(3/2, 3.9)`, and layer-specific
lengthscale rates `theta_y=3.9`, `theta_q=3.9/3`, and `theta_r=3.9/6`.
Initial kernel hyperparameters are `theta=1`, `g=9e-5`, and `tau2=0.005`.

Each run writes:
- `config_used.json`
- `initial_values_and_priors.npz`
- `results_summary.json`
- `mcmc_samples.pkl`, unless `--no-save-samples` is used
- diagnostic trace, density, autocorrelation, W, and prediction plots, unless
  `--no-plots` is used
- `metrics_boxplot.pdf` and `metrics_summary_table.pdf`, unless `--no-plots`
  is used

The top-level output directory also includes `simulation_summary.csv` with
`D`, `layer`, `variant`, status, metric means, and output paths.

### Integration with Main Framework

```python
from Data_generation import generate_case1_1d
from run_multichains import run_multichain_analysis, create_config_D1_L1

# Generate data
data = generate_case1_1d(n=200, seed=42)

# Create config
config = create_config_D1_L1(
    p=data['X_train'].shape[1],
    seed=42
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

## Parameters

- **n**: Number of training samples (default: 200)
- **seed**: Random seed for reproducibility (default: None)
- **p**: Input dimension (automatically determined from scenario)
- **D**: Reduced dimension (1 for 1D, 2 for 2D scenarios)

## Output Format

All generation functions return a dictionary with:
- `y_train`, `y_test`: Response vectors
- `X_train`, `X_test`: Input matrices
- `z_train`, `z_test`: Latent projections (if applicable)
- `w_true`: True projection matrix (if applicable)

## See Also

- `../run_multichains.py` - Main interface that uses generated data
- `../run_multichains.ipynb` - Jupyter notebook with examples
- `../example_usage.py` - Examples of all data generation scenarios
- `../test_data_generation.py` - Comprehensive tests
