# Data Generation Module

This module provides synthetic data generation for testing and validation of Bayesian Dimensionality Reduction models.

## Contents

- `Data_generation.py` - Main data generation module
- `../Scripts/run_simulation.py` - End-to-end multichain simulation runner
- `__init__.py` - Package initialization

Detailed command-line runner documentation now lives in
[`../Scripts/README.md`](../Scripts/README.md). This README keeps the
synthetic data generator notes and related simulation context.

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

### Case 2a: Piecewise Functions

```python
from Data_generation import generate_case2_piecewise

data = generate_case2_piecewise(n=200, seed=42)
```

### Case 3b: Exponential Functions

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

`Scripts/run_simulation.py` generates Case 1 or Case 2 data, builds initial values and prior
objects through the main configuration helpers, and runs three-chain posterior
sampling across 1-layer, 2-layer, and 3-layer models. It defaults to the
isotropic squared exponential kernel, uses `seed=42`, and uses `thin=3`.

The data case, data-generation dimension, and posterior-sampling dimension are separate:

- `--data-cases`: generator case, currently `case1` or `case2`
- `--data-dimensions`: true generator dimension (`1` or `2`)
- `--posterior-dimensions`: reduced dimension `D` used by the BDR posterior sampler
- `--dimensions`: backward-compatible alias for `--posterior-dimensions`
- `--kernel-type`: covariance kernel used by the BDR sampler

`Scripts/run_simulation.py` imports `generate_case1_1d`, `generate_case1_2d`,
`generate_case2_1d`, and `generate_case2_2d` from `Data_generation.py`.
The Case 2 names are convenience aliases for the existing Case 2 generators:
`generate_case2_1d` wraps the piecewise-function case, and
`generate_case2_2d` wraps the exponential-function case.

```bash
python "Scripts/run_simulation.py"
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
- `--kernel-type`: choose one of `isotropic_squared_exponential`,
  `separable_squared_exponential`, `isotropic_matern32`, or
  `separable_matern32`. The default is `isotropic_squared_exponential`.
- `--w-variants`: choose specific variants from `full`, `W_Known`, `No_W`,
  and `No_W_Selective`.
- `--variant-dimensions`: optional per-variant posterior dimensions using
  `VARIANT=D[,D...]`, for example
  `full=1,2,3 No_W_Selective=1,2,3 W_known=1,2`. Variants not listed here
  use `--posterior-dimensions`. `W_known` is accepted as an alias for
  `W_Known` in this option.
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
python "Scripts/run_simulation.py" \
  --w-variants full No_W_Selective
```

Add `--include-w-variants` as a shortcut when you want all W variants alongside
the full model:

- `W_Known`: fixes `W` at the true projection matrix returned by `Data_generation.py`
- `No_W`: omits `W` and uses the full input matrix directly
- `No_W_Selective`: omits `W` and uses the first `D` columns of `X`

Do not combine `--w-variants` and `--include-w-variants` in the same command.

Use `--variant-dimensions` when each selected variant should use its own D
grid:

```bash
python "Scripts/run_simulation.py" \
  --sample-size 200 \
  --data-cases case1 case2 \
  --data-dimensions 1 2 \
  --layers 1 2 3 \
  --kernel-type isotropic_squared_exponential \
  --w-variants full No_W_Selective W_Known \
  --variant-dimensions full=1,2,3 No_W_Selective=1,2,3 W_known=1,2 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --output-dir ./simulation_outputs
```

If `--variant-dimensions` is omitted, every selected variant uses every value
from `--posterior-dimensions`. If a selected variant is omitted from
`--variant-dimensions`, only that variant falls back to `--posterior-dimensions`.
`No_W` uses all input columns regardless of `D`, so normally give `No_W` only
one D value if you include it in `--variant-dimensions`.

All variants use the same `seed=42`, `thin=3`, selected kernel type, initial
kernel hyperparameters, and prior values as the full model.
For W-not-sampled variants, posterior samples contain the parameters that are
active in that variant. `W_Known` keeps `W` fixed, `No_W` has no `W`, and
`No_W_Selective` keeps the selected input columns fixed rather than storing a
fixed `W` matrix.
Because `W_Known` uses the true generator `W`, it only runs when the selected
posterior dimension for that `W_Known` row matches the selected
`--data-dimensions` value. Mismatched `W_Known` combinations are marked as
skipped in `simulation_summary.csv`.

By default, the runner uses a small quick-test setup. Increase the sample size
and MCMC iterations for real analysis:

```bash
python "Scripts/run_simulation.py" \
  --sample-size 200 \
  --data-cases case1 case2 \
  --data-dimensions 1 2 \
  --posterior-dimensions 1 2 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --kernel-type isotropic_squared_exponential \
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
- `posterior_parameter_summary.csv`; `posterior_parameter_summary.pdf` is
  also written unless `--no-plots` is used
- `time_complexity_summary.csv`; `time_complexity_summary.pdf` is also written
  unless `--no-plots` is used
- diagnostic trace, density, autocorrelation, W-entry, WWT-entry, `Lambda`,
  `M`, `V`, `Q`, `R`, and prediction plots, unless `--no-plots` is used.
  Full model runs where `W` is sampled include trace, density, and
  autocorrelation plots for both `W` and `W W^T`, plus `Lambda`, `M`, and `V`.
  `Q` plots are produced for layer 2 and 3 runs; `R` plots are produced for
  layer 3 runs. High-dimensional matrices are flattened and limited to the
  first 12 entries for readability. Diagnostic plots requested as `.png` are
  also written as matching `.pdf` files.
- `metrics_boxplot.pdf` and `metrics_summary_table.pdf`, unless `--no-plots`
  is used. These include RMSPE, NSME, CRPS, BIC, MLPPD, CP, ALCI, Score.
- Top-level `metric_boxplots_by_layer/*.png` and matching `.pdf` files,
  unless `--no-plots` is used. For each metric and layer, these compare compact
  model labels such as `1-BDR`, `2-BDR`, `1-W/o`, and `1-Oracle`, where the
  number is the posterior dimension selected for that variant. 
`posterior_parameter_summary.csv` contains one row per parameter component and
chain:

```text
parameter | component | chain | mean | median | sd | 2.5% | 97.5% | ESS | Rhat | stest | pvalue
```

The table is built for all numeric posterior sample arrays present in the run,
including `W`, derived `W W^T`, `Lambda`, `M`, `V`, `Q`, `R`, and sampled
hyperparameters. `ESS` and `Rhat` are computed from the saved chains. For
matching parameters where R/coda diagnostics are available, `stest` and
`pvalue` use the coda Heidelberger-Welch stationarity result; otherwise the
table still writes a Python stationarity fallback.

The same table also includes complexity rows as pseudo-parameters. For example,
`overall_time_complexity`, `time_complexity_W`, `time_complexity_Q`,
`time_complexity_R`, `time_complexity_Lambda`, `time_complexity_M`, and
`time_complexity_V` appear when those quantities are selected/present in the
run. For these rows, the complexity formula is written in `component`,
`chain=all`, and the posterior-statistic columns are blank.
`time_complexity_summary.csv` records the overall run complexity scale and the
per-sample/per-iteration complexity for each selected sampled parameter only:
HMC for `W`, derived `W W^T`, latent `Q`/`R` elliptical-slice updates,
`Lambda` slice sampling, matrix-von-Mises-Fisher updates for `M`/`V`, and GP
hyperparameter updates.

The top-level output directory also includes `simulation_summary.csv` with
`D`, `layer`, `variant`, `kernel_type`, status, metric means,
`total_seconds`, and output paths. It also writes
`metrics_comparison_table.csv`; when plots are enabled, it writes
`metrics_comparison_table.pdf`. The comparison table uses
publication labels for full layer-1, layer-2, and layer-3 BDR runs:

```text
posterior_D=1, layer=1, variant=full -> GP (1) BDR
posterior_D=2, layer=1, variant=full -> GP (2) BDR
posterior_D=3, layer=1, variant=full -> GP (3) BDR
posterior_D=1, layer=2, variant=full -> DGP 2-layer (1) BDR
posterior_D=2, layer=2, variant=full -> DGP 2-layer (2) BDR
posterior_D=3, layer=2, variant=full -> DGP 2-layer (3) BDR
posterior_D=1, layer=3, variant=full -> DGP 3-layer (1) BDR
posterior_D=2, layer=3, variant=full -> DGP 3-layer (2) BDR
posterior_D=3, layer=3, variant=full -> DGP 3-layer (3) BDR
posterior_D=1, layer=1, variant=W_Known -> GP (1) Oracle
posterior_D=1, layer=2, variant=W_Known -> DGP 2-layer (1) Oracle
posterior_D=1, layer=3, variant=W_Known -> DGP 3-layer (1) Oracle
posterior_D=2, layer=1, variant=W_Known -> GP (2) Oracle
posterior_D=2, layer=2, variant=W_Known -> DGP 2-layer (2) Oracle
posterior_D=2, layer=3, variant=W_Known -> DGP 3-layer (2) Oracle
posterior_D=D, layer=1, variant=No_W_Selective -> GP (D) W/o
posterior_D=D, layer=2, variant=No_W_Selective -> DGP 2-layer (D) W/o
posterior_D=D, layer=3, variant=No_W_Selective -> DGP 3-layer (D) W/o
```

The comparison table reports RMSPE, NSME, CRPS, BIC, MLPPD, CP, ALCI, and
Score in that order.

#### Run Count and Runtime Accounting

The runner creates one summary row for every selected combination:

```text
len(data_cases) * len(data_dimensions) *
sum(number of selected D values for each selected variant) * len(layers)
```

Without `--variant-dimensions`, the selected-D count is
`len(posterior_dimensions) * len(w_variants)`.

Rows are marked as skipped instead of run when `W_Known` is selected but
`data_dim != posterior_D`, because the true generator `W` has a different
number of columns than the requested posterior model.

For the full example above:

```text
2 data cases * 2 data dimensions * 2 posterior dimensions *
3 layers * 4 W variants = 96 summary rows
```

Of those, `12` `W_Known` rows are skipped, so `84` model runs are attempted.
Each successful row records its measured runtime in `total_seconds`; total
model time is the sum of `total_seconds` for rows with `status=ok`. Higher
`posterior_D`, deeper layers, more W variants, more chains, and more MCMC
iterations all increase runtime. `thin` changes how many samples are saved
after burn-in, but the sampler still performs `--n-iterations` iterations.




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
