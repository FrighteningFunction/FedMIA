# FedMIA on BINN

This repository contains a FedMIA membership inference evaluation for the BINN prostate-cancer model. The current main experiment is a target-client membership attack: given a candidate patient `(x, y)`, the attack estimates whether that patient was included in the target client's local training data during federated learning.

The implementation evaluates both FedMIA variants used in the paper:

- **FedMIA-I**: loss-based measurement, implemented as negative patient loss on each client's locally trained model.
- **FedMIA-II**: gradient-cosine measurement, comparing each client update with the candidate patient's gradient on the global model.

Both variants use non-target client measurements to estimate `Q_out`, apply the Gaussian CDF scoring step, average scores across communication rounds, and classify membership with a threshold.

## Main Run: Patient-Wise BINN Attack

Use this for the final patient-level BINN evaluation:

```bash
GPU=0 bash membership_attack_binn_patient_eval.sh
```

This launcher runs the paper-style BINN FedMIA experiment and additionally tracks audited patients across repeated runs. The audited patients are deliberately alternated between:

- **member**: present in target client 0
- **nonmember**: absent from target client 0, either in non-target clients or in holdout depending on the configured nonmember source

This makes patient-wise AUC, score gap, false-positive behavior, and most/least vulnerable patient reporting meaningful.

Useful override example:

```bash
RUNS=4 CLIENT_GRID=3,5,10 ROUND_GRID=50 LOCAL_EPOCH_GRID=2 BETA_GRID=1 AUDIT_PATIENT_COUNT=64 GPU=0 bash membership_attack_binn_patient_eval.sh
```

Quick plumbing check:

```bash
PLUMBING=1 bash membership_attack_binn_patient_eval.sh
```

## Shell Scripts

| Script | Purpose |
| --- | --- |
| `membership_attack_binn_patient_eval.sh` | Main recommended script. Runs BINN FedMIA with audited patient tracking and patient-wise vulnerability reports. |
| `membership_attack_binn_eval.sh` | Older compact evaluation script. Runs several fixed BINN configurations and aggregates config-level metrics, but is less focused on patient-wise reporting. |
| `membership_attack.sh` | Lower-level configurable launcher used by both BINN wrappers. Call this directly only if you want full control over the grid parameters. |
| `membership_attack_cifar100.sh` / `membership_attack_cifar100_eval.sh` | CIFAR-100/AlexNet reproduction scripts for the FedMIA paper-style image setting. |

In short: use `membership_attack_binn_patient_eval.sh` for the current thesis/report experiment; use `membership_attack.sh` only when constructing a custom grid manually.

## Important Parameters

Most parameters can be overridden as environment variables before the shell command.

| Parameter | Meaning |
| --- | --- |
| `RUNS` | Number of repeated federated trajectories per configuration. Needed for mean/std and patient-wise IN/OUT comparisons. |
| `CLIENT_GRID` | Number of FL clients, for example `3,5,10`. |
| `ROUND_GRID` | Number of communication rounds. |
| `LOCAL_EPOCH_GRID` | Local epochs per client per communication round. |
| `BETA_GRID` | Data heterogeneity setting. `iid` means approximately IID; numeric values use Dirichlet partitioning. Lower beta means stronger non-IID. |
| `SAMPLES_PER_CLIENT_GRID` | Number of patient records per client. `auto` uses the maximum disjoint amount available for the selected client count. |
| `AUDIT_PATIENT_COUNT` | Number of fixed patients tracked for patient-wise reporting. |
| `NONMEMBER_SOURCE` | Which samples count as target-client nonmembers. The patient-wise script defaults to `target_nonmembers`. |
| `THRESHOLD` | FedMIA decision threshold, usually `0.5`. |
| `GPU` | CUDA device id exposed through `CUDA_VISIBLE_DEVICES`. |

## Outputs

Experiment outputs are written automatically:

- `logs/`: text logs and JSONL progress logs
- `reports/`: main TXT/CSV experiment reports and patient-wise reports
- `aggregated_report/`: collected tables and HTML/CSV views for paper writing
- `charts/`: generated charts from collected reports

After new runs, regenerate aggregate tables with:

```bash
python3 scripts/collect_fedmia_report_results.py
```

Regenerate charts with:

```bash
python3 scripts/make_fedmia_charts.py
```

## Tests

Run the focused patient-reporting tests with:

```bash
python3 -m pytest tests/test_binn_paper_grid_patient_reporting.py
```

Run a syntax check for the main BINN experiment script with:

```bash
python3 -m py_compile experiments/fedmia_binn_paper_grid.py
```

## Data

The BINN experiment expects the prostate-cancer arrays under:

```text
data/datasets/ProstateCancer/pnet_x.npy
data/datasets/ProstateCancer/pnet_y.npy
```

The input `x` is tabular patient molecular data, and `y` is the binary class label used to compute supervised loss and gradients.

## Method Summary

For each communication round, FedMIA computes a scalar measurement for each candidate patient and each client. FedMIA-I uses negative loss on the client-local model. FedMIA-II computes the candidate patient's gradient on the global model and compares it to each client update by cosine similarity. The non-target client measurements estimate a Gaussian `Q_out`; the target client's measurement is scored with the Gaussian CDF. Scores are averaged over communication rounds, then thresholded to infer target-client membership.
