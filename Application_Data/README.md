# Application Data Runner

Use `run_application.py` to run BDR posterior sampling on the real application
datasets in this folder.

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
python "Application_Data/run_application.py" \
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

`W_Known` is not included because the application datasets do not provide a
true known projection matrix. Use `full` to sample `W`, or use `No_W` and
`No_W_Selective` for W-not-sampled comparisons.

The top-level output directory contains `application_summary.csv`. Each run
folder contains `config_used.json`, `initial_values_and_priors.npz`,
`results_summary.json`, and optional posterior samples and plots.
