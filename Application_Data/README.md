# Application Data Runner

Use `../Scripts/run_application.py` to run BDR posterior sampling on the real
application datasets in this folder.

Detailed command-line runner documentation now lives in
[`../Scripts/README.md`](../Scripts/README.md). This README keeps the
application dataset notes and related runner context.

## Datasets

- `Elliptical_PDE`: uses `X_1.npy`/`Y_1.npy` and/or `X_2.npy`/`Y_2.npy` with
  the `separable_matern32` kernel.
- `Onera M6`: uses `onera_m6.csv`, `x_*` columns as inputs, and `lift` or
  `drag` as the response with the `separable_squared_exponential` kernel.

The runner uses an 80/20 train/test split by default with `seed=42`.

## Options

Dataset selection:

- `--applications`: choose `elliptical_pde`, `onera`, or both.
- `--elliptical-outputs`: choose `1`, `2`, or both. Output `1` uses
  `X_1.npy` and `Y_1.npy`; output `2` uses `X_2.npy` and `Y_2.npy`.
- `--onera-targets`: choose `lift`, `drag`, or both.

Model selection:

- `--posterior-dimensions`: reduced posterior dimension(s) `D`.
- `--layers`: model depth(s), selected from `1`, `2`, and `3`.
- `--variants`: choose `full`, `No_W`, and/or `No_W_Selective`.
- `--variant-dimensions`: optional per-variant posterior dimensions using
  `VARIANT=D[,D...]`, for example `full=1,2,3 No_W_Selective=1,2,3`.
  Variants not listed here use `--posterior-dimensions`.
- `full`: samples `W`, `M`, `V`, `Lambda`, and available GP hyperparameters.
- `No_W`: does not sample `W`; uses all original columns of `X`.
- `No_W_Selective`: does not sample `W`; uses the first `D` columns of `X`.

MCMC controls:

- `--n-chains`: number of MCMC chains.
- `--n-iterations`: MCMC iterations per chain.
- `--burn-in`: burn-in iterations discarded before summaries.
- `--thin`: thinning interval for saved posterior samples.
- `--seed`: random seed for the train/test split and initialization; default is `42`.
- `--train-fraction`: training fraction; default is `0.8`.
- `--max-rows`: optional row cap before splitting, useful for smoke tests.

M/V sampler controls:

- `--mv-sampler python`: default local Python sampler for posterior `M` and `V`.
- `--mv-sampler rstiefel`: uses the R `rstiefel` backend for posterior `M`
  and `V` in the `full` model.
- `--rstiefel-rscol`: optional `rscol` value for
  `rstiefel::rmf.matrix.gibbs` when `D > 1`.

Output controls:

- `--output-dir`: directory where run folders and `application_summary.csv`
  are written.
- `--no-plots`: skips plot PDFs; metrics are still computed.
- `--no-save-samples`: skips `mcmc_samples.pkl`; summaries and metrics are
  still written.
- `--parameter-diagnostics`: attempts extra R/coda diagnostics.
- `--continue-on-error`: keeps running later combinations after one failure.
- `--verbose`: prints sampler progress.

## Quick Test

```bash
python "Scripts/run_application.py" \
  --applications elliptical_pde \
  --elliptical-outputs 1 \
  --posterior-dimensions 1 \
  --layers 1 \
  --variants full \
  --max-rows 12 \
  --n-iterations 4 \
  --burn-in 1 \
  --thin 3 \
  --no-plots \
  --no-save-samples \
  --output-dir ./application_smoke_test
```

## Full Application Run

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

Use `--variant-dimensions` when each selected variant should use its own D
grid:

```bash
python "Scripts/run_application.py" \
  --applications elliptical_pde onera \
  --elliptical-outputs 1 \
  --onera-targets lift \
  --layers 1 2 3 \
  --variants full No_W_Selective \
  --variant-dimensions full=1,2,3 No_W_Selective=1,2,3 \
  --n-chains 3 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --output-dir ./application_outputs
```

If `--variant-dimensions` is omitted, every selected variant uses every value
from `--posterior-dimensions`. If a selected variant is omitted from
`--variant-dimensions`, only that variant falls back to `--posterior-dimensions`.
`No_W` uses all input columns regardless of `D`, so normally give `No_W` only
one D value if you include it in `--variant-dimensions`.

`W_Known` is not included because the application datasets do not provide a
true known projection matrix. Use `full` to sample `W`, or use `No_W` and
`No_W_Selective` for W-not-sampled comparisons.

The top-level output directory contains `application_summary.csv`. Each row
includes `posterior_D`, `layer`, `variant`, status, metric means,
`total_seconds`, and the run output path. It also writes
`metrics_comparison_table.csv`; when plots are enabled, it writes
`metrics_comparison_table.pdf`. Full layer-1 BDR runs are labeled as
`GP (1) BDR`, `GP (2) BDR`, or `GP (3) BDR` according to posterior dimension.
Full layer-2 BDR runs are labeled as `DGP 2-layer (1) BDR`,
`DGP 2-layer (2) BDR`, or `DGP 2-layer (3) BDR`.
Full layer-3 BDR runs are labeled as `DGP 3-layer (1) BDR`,
`DGP 3-layer (2) BDR`, or `DGP 3-layer (3) BDR`.
`No_W_Selective` runs are labeled as `GP (D) W/o`,
`DGP 2-layer (D) W/o`, or `DGP 3-layer (D) W/o` according to layer and
posterior dimension.
Each run folder contains
`config_used.json`, `initial_values_and_priors.npz`, `results_summary.json`,
`posterior_parameter_summary.csv`, `time_complexity_summary.csv`, and optional
posterior samples and plots. Matching PDF versions of the two summary tables
are written unless `--no-plots` is used. Full model runs that sample `W`
include trace, density, and autocorrelation diagnostics for `W`, `W W^T`,
`Lambda`, `M`, and `V`. Layer 2 and 3 runs include `Q` diagnostics; layer 3
runs include `R` diagnostics. High-dimensional matrices are flattened and
limited to the first 12 entries for readability. Diagnostic plots requested as
`.png` are also written as matching `.pdf` files.
Metric summaries include RMSPE, NSME, CRPS, BIC, MLPPD, CP, ALCI, Score.
The top-level `metric_boxplots_by_layer/` folder contains per-layer boxplots
for each metric using compact model labels such as `1-BDR`, `2-BDR`, and
`1-W/o`. 

`posterior_parameter_summary.csv` contains:

```text
parameter | component | chain | mean | median | sd | 2.5% | 97.5% | ESS | Rhat | stest | pvalue
```

It includes all numeric sampled parameter arrays present in the run, plus the
derived `W W^T` projection when `W` is sampled. For matching parameters where
R/coda diagnostics are available, `stest` and `pvalue` use the coda
Heidelberger-Welch stationarity result; otherwise the table still writes a
Python stationarity fallback. The same table also includes complexity rows as
pseudo-parameters, such as `overall_time_complexity`, `time_complexity_W`,
`time_complexity_Q`, `time_complexity_R`, `time_complexity_Lambda`,
`time_complexity_M`, and `time_complexity_V` when those quantities are
selected/present in the run. For these rows, the complexity formula is written
in `component`, `chain=all`, and the posterior-statistic columns are blank.
`time_complexity_summary.csv` records the same complexity information in a
separate compact table.

The runner creates one summary row for every selected combination:

```text
number of selected application targets *
sum(number of selected D values for each selected variant) * len(layers)
```

Without `--variant-dimensions`, the selected-D count is
`len(posterior_dimensions) * len(variants)`.

Each successful row records its measured runtime in `total_seconds`; total
model time is the sum of `total_seconds` for rows with `status=ok`. Higher
`posterior_D`, deeper layers, more variants, more chains, and more MCMC
iterations all increase runtime.

The main sampler costs are the same as in the simulation runner: HMC for
sampled `W` is dominated by repeated GP likelihood/gradient evaluations,
latent `Q` and `R` elliptical-slice updates are dominated by GP covariance
factorization/solves, `Lambda` slice sampling scales with the number of slice
angle trials, and `M`/`V` use matrix-von-Mises-Fisher Gibbs updates. See
`Data Generation/README.md` for the detailed complexity table.
