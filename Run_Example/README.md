# Run Examples

The script replicates the results in  Case 1: Synthetic Quadratic Response Surface with Known Structure(Section 3.1) and Case 2: Synthetic Data(Section 3.2) simulation setup for the different sample size, cases, input dimension, layers 1, 2, and 3, and the variants full (BDR), No_W_Selective (W/o), and W_Known(Oracle). To replicate Section 3.1.1 1D input subspace results, we specify these input parameters: SAMPLE_SIZE = 350,  DATA_CASES = ["case1"], DATA_DIMENSIONS = [1], LAYERS = [1, 2, 3], W_VARIANTS = ["full", "No_W_Selective", "W_Known"], VARIANT_DIMENSIONS = ["full=1,2,3",  "No_W_Selective=1,2,3", "W_known=1"], N_ITERATIONS = 2000, BURN_IN = 500, THIN = 3, KERNEL_TYPE = "isotropic_squared_exponential" then run "python3 -m py_compile Run_Example/run_one_case.py" on the terminal. After is done running, it automatically generates Table 1 and 2 performance metrics for 1D input subspace at n_{train}=280 and 480 when additionally set SAMPLE_SIZE=600. Additionally, it generates Figure 5, Figure 6, Figure SM37 and Figure SM38 with their corresponding diagnostics plots and Table SM4 (posterior summary table). Changing the input parameters SAMPLE_SIZE =350 or 600, DATA_CASES=["case1"], and DATA_DIMENSIONS=[2],  inside Run_Example/run_one_case.py runs Section 3.1.2 to replicate the same results.
Changing the input parameters SAMPLE_SIZE =300 or 500, DATA_CASES=["case2"], and DATA_DIMENSIONS=[1],  inside Run_Example/run_one_case.py runs Section 3.2.1 to replicate the same results.
Changing the input parameters SAMPLE_SIZE =300 or 500, DATA_CASES=["case2"], and DATA_DIMENSIONS=[2],  inside Run_Example/run_one_case.py runs Section 3.2.2 to replicate the same results.

## Section 3.1.1 1D Input Subspace (Case 1a Simulation)

`run_one_case.py` runs the synthetic Case 1a setup:

```bash
python3 Run_Example/run_one_case.py
```

The script calls:

```bash
python "Scripts/run_simulation.py" \
  --sample-size 350 \
  --data-cases case1 \
  --data-dimensions 1 \
  --layers 1 2 3 \
  --kernel-type isotropic_squared_exponential \
  --w-variants full No_W_Selective W_Known \
  --variant-dimensions full=1,2,3 No_W_Selective=1,2,3 W_known=1 \
  --n-iterations 2000 \
  --burn-in 500 \
  --thin 3 \
  --output-dir ./simulation_outputs
```

To check the file without running the full simulation:

```bash
python3 -m py_compile Run_Example/run_one_case.py
```

Expected outputs go under `simulation_outputs/`. The runner writes per-run
folders, `simulation_summary.csv`, `metrics_comparison_table.csv/.pdf`,
posterior summary tables, time-complexity tables, diagnostics, and
`metric_boxplots_by_layer/` plots.

## Switching Synthetic Cases

Edit the constants near the top of `run_one_case.py`:

| Case | DATA_CASES | DATA_DIMENSIONS | SAMPLE_SIZE values |
|---|---|---:|---|
| Case 1a | `["case1"]` | `[1]` | `350`, then `600` |
| Case 1b | `["case1"]` | `[2]` | `350`, then `600` |
| Case 2a | `["case2"]` | `[1]` | `300`, then `500` |
| Case 2b | `["case2"]` | `[2]` | `300`, then `500` |

For `W_Known`, keep `W_known` dimensions consistent with the true input
subspace. For example, use `W_known=1` for data dimension 1 and `W_known=2`
for data dimension 2.

## Running Application Data

Application data are run directly through `Scripts/run_application.py`:

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

Application outputs go under `application_outputs/` and include
`application_summary.csv`, comparison tables, run folders, diagnostic plots,
and metric boxplots. `W_Known` is not available for application data because
there is no true known projection matrix.


