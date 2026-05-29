# Experiment Runner Scripts

This folder contains the command-line entry points for end-to-end BDR
experiments:

- `run_simulation.py`: synthetic Case 1/Case 2 experiments from
  `Data Generation/Data_generation.py`
- `run_application.py`: real application experiments from `Application_Data/`

Run commands from the repository root so quoted paths and relative output
directories resolve correctly:

```bash
python "Scripts/run_simulation.py" --help
python "Scripts/run_application.py" --help
```

## Synthetic Simulation Runner

`run_simulation.py` generates synthetic data, initializes priors and starting
values, runs multichain posterior sampling, and writes diagnostics, posterior
summaries, metrics, and plots.

Simulation-specific inputs:

- `--sample-size`: total generated observations before the train/test split.
- `--data-cases`: synthetic generator case, `case1`, `case2`, or both.
- `--data-dimensions`: true synthetic generator dimension, `1`, `2`, or both.
- `--w-variants`: selected model variants from `full`, `W_Known`, `No_W`, and
  `No_W_Selective`.
- `--include-w-variants`: shortcut for running `full`, `W_Known`, `No_W`, and
  `No_W_Selective`.
- `--kernel-type`: covariance kernel for all synthetic simulation runs. Choices
  are `isotropic_squared_exponential`, `separable_squared_exponential`,
  `isotropic_matern32`, and `separable_matern32`.

The synthetic runner defaults to `isotropic_squared_exponential`. `W_Known`
fixes `W` at the true generator projection and only runs when `posterior_D ==
data_dimension`; mismatches are marked as skipped in `simulation_summary.csv`.

```bash
python "Scripts/run_simulation.py" \
  --sample-size 200 \
  --data-cases case1 case2 \
  --data-dimensions 1 \
  --layers 1 2 3 \
  --kernel-type isotropic_squared_exponential \
  --w-variants full No_W_Selective W_Known \
  --variant-dimensions full=1,2,3 No_W_Selective=1,2,3 W_known=1,2 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --output-dir ./simulation_outputs
```

## Application Data Runner

`run_application.py` loads real datasets, creates an 80/20 train/test split by
default, initializes priors and starting values, runs multichain posterior
sampling, and writes the same summary and plot artifacts as the simulation
runner.

Application-specific inputs:

- `--applications`: `elliptical_pde`, `onera`, or both.
- `--elliptical-outputs`: Elliptical_PDE output pairs `1`, `2`, or both.
- `--onera-targets`: Onera response column, `lift`, `drag`, or both.
- `--variants`: selected model variants from `full`, `No_W`, and
  `No_W_Selective`.
- `--train-fraction`: training split fraction; default is `0.8`.
- `--max-rows`: optional row cap before splitting for smoke tests.

Application kernels are fixed by dataset: Elliptical_PDE uses the separable
Matern-3/2 kernel, and Onera M6 uses the separable squared exponential kernel.
`W_Known` is not available because the real application datasets do not provide
a true known projection matrix.

```bash
python "Scripts/run_application.py" \
  --applications elliptical_pde onera \
  --elliptical-outputs 1 \
  --onera-targets lift \
  --layers 1 2 3 \
  --variants full No_W_Selective \
  --variant-dimensions full=1,2,3 No_W_Selective=1,2,3,10 \
  --n-chains 3 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --output-dir ./application_outputs
```



## Shared Model Options

Both runners support these model and MCMC options:

- `--posterior-dimensions`: posterior/model reduced dimensions `D`.
- `--dimensions`: backward-compatible alias for `--posterior-dimensions`.
- `--variant-dimensions`: optional per-variant D grid using
  `VARIANT=D[,D...]`; variants not listed here use `--posterior-dimensions`.
- `--layers`: model depths, selected from `1`, `2`, and `3`.
- `--kernel-type`: simulation runner only; covariance kernel for synthetic
  runs. Application kernels are fixed by dataset.
- `--n-chains`: number of MCMC chains.
- `--n-iterations`: MCMC iterations per chain.
- `--burn-in`: iterations discarded before summaries.
- `--thin`: thinning interval for saved posterior samples.
- `--mv-sampler python`: default local matrix-von-Mises-Fisher sampler for
  posterior `M` and `V`.
- `--mv-sampler rstiefel`: use R `rstiefel::rmf.matrix.gibbs` through `rpy2`
  for posterior `M` and `V` in full W-sampled models.
- `--rstiefel-rscol`: optional `rscol` argument for the R `rstiefel` backend.
- `--output-dir`: directory for run folders and top-level summary files.
- `--no-plots`: skip diagnostic and metric plot files; metrics are still
  computed.
- `--no-save-samples`: skip `mcmc_samples.pkl`; summaries and metrics are still
  written.
- `--parameter-diagnostics`: attempt extra R/coda diagnostics.
- `--continue-on-error`: continue later combinations after one failed run.
- `--verbose`: print sampler progress.

Variant behavior:

- `full`: samples `W`, `M`, `V`, `Lambda`, and available GP hyperparameters.
- `W_Known`: simulation only; fixes `W` at the true synthetic projection.
- `No_W`: does not sample `W`; uses all original columns of `X`.
- `No_W_Selective`: does not sample `W`; uses the first `D` columns of `X`.

`No_W` uses all input columns regardless of `D`, so normally give it only one D
value in `--variant-dimensions`.

## Output Structure

Each runner writes a top-level summary file:

- Simulation: `simulation_summary.csv`
- Application: `application_summary.csv`

Each successful run folder contains:

- `config_used.json`
- `initial_values_and_priors.npz`
- `results_summary.json`
- `mcmc_samples.pkl`, unless `--no-save-samples` is used
- `posterior_parameter_summary.csv`
- `posterior_parameter_summary.pdf`, unless `--no-plots` is used
- `time_complexity_summary.csv`
- `time_complexity_summary.pdf`, unless `--no-plots` is used

The top-level output directory also contains:

- `metrics_comparison_table.csv`
- `metrics_comparison_table.pdf`, unless `--no-plots` is used
- `metric_boxplots_by_layer/`, unless `--no-plots` is used

Summary rows include the selected data/application, `posterior_D`, `layer`,
`variant`, `kernel_type`, status, metric means, `total_seconds`, and output
paths. Successful rows record measured runtime in `total_seconds`; total model
time is the sum of `total_seconds` over rows with `status=ok`.

## Posterior Summary Tables

`posterior_parameter_summary.csv` has this column layout:

```text
parameter | component | chain | mean | median | sd | 2.5% | 97.5% | ESS | Rhat | stest | pvalue
```

It includes all numeric sampled parameter arrays present in the run, including
sampled hyperparameters, `W`, derived `W W^T`, `Lambda`, `M`, `V`, `Q`, and
`R` when those quantities exist for the selected model and layer.

For matching parameters where R/coda diagnostics are available, `stest` and
`pvalue` use the coda Heidelberger-Welch stationarity result. Otherwise the
table still writes a Python stationarity fallback.

The same table also includes complexity rows as pseudo-parameters:

- `overall_time_complexity`
- `time_complexity_W`
- `time_complexity_Q`
- `time_complexity_R`
- `time_complexity_Lambda`
- `time_complexity_M`
- `time_complexity_V`

For these rows, the complexity formula is written in `component`, `chain=all`,
and the posterior-statistic columns are blank. `time_complexity_summary.csv`
records the same complexity information in a compact table.

## Metrics And Labels

Metric summaries and comparison tables include:

```text
RMSPE | NSME | CRPS | BIC | MLPPD | CP | ALCI | Score
```

Publication labels in `metrics_comparison_table.*` use:

- `full`, layer 1: `GP (D) BDR`
- `full`, layer 2: `DGP 2-layer (D) BDR`
- `full`, layer 3: `DGP 3-layer (D) BDR`
- `W_Known`, layer 1: `GP (D) Oracle`
- `W_Known`, layer 2: `DGP 2-layer (D) Oracle`
- `W_Known`, layer 3: `DGP 3-layer (D) Oracle`
- `No_W_Selective`, layer 1: `GP (D) W/o`
- `No_W_Selective`, layer 2: `DGP 2-layer (D) W/o`
- `No_W_Selective`, layer 3: `DGP 3-layer (D) W/o`

Compact labels in `metric_boxplots_by_layer/` use:

- `full`: `1-BDR`, `2-BDR`, `3-BDR`, ...
- `W_Known`: `1-Oracle`, `2-Oracle`, ...
- `No_W_Selective`: `1-W/o`, `2-W/o`, `10-W/o`, ...

The number comes from the selected `D` for that variant.

## Metric Plots

The `metric_boxplots_by_layer/` folder contains one single-layer boxplot per
metric and layer:

```text
rmspe_layer1_model_boxplot.png
rmspe_layer1_model_boxplot.pdf
rmspe_layer2_model_boxplot.png
rmspe_layer2_model_boxplot.pdf
```

It also contains grouped layer plots with layer 1, 2, and 3 side by side in one
figure for each metric:

```text
rmspe_layers_grouped_model_boxplot.png
rmspe_layers_grouped_model_boxplot.pdf
```


## Diagnostic Plots

When plots are enabled, run folders include trace, density, and autocorrelation
diagnostics for sampled quantities. Full W-sampled models include diagnostics
for `W`, derived `W W^T`, `Lambda`, `M`, and `V`. Layer 2 and 3 runs include
`Q` diagnostics; layer 3 runs include `R` diagnostics.

High-dimensional arrays are flattened and limited to the first 12 entries for
readability. Diagnostic plots requested as `.png` are also written as matching
`.pdf` files.





## Related Documentation

- [`../Data Generation/README.md`](../Data%20Generation/README.md): synthetic
  data generator details.
- [`../Application_Data/README.md`](../Application_Data/README.md): real
  application dataset details.
- [`../BDR Metrics and Plot/README.md`](../BDR%20Metrics%20and%20Plot/README.md):
  metric and plotting helper details.
