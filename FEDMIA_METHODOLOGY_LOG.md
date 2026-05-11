# FedMIA BINN Methodology Log

Date: 2026-05-09-

This note tracks the attack methodology changes made during the BINN/FedMIA evaluation work. It is intentionally kept at the repository root so it can serve as the human-readable source of truth for the experiment history.

## 1. Initial Portable FedMIA Implementation

File: `attacks/fedmia.py`

Purpose:

- Implement the paper's FedMIA scoring core in a model-agnostic way.
- Support high-dimensional client updates by reducing them to low-dimensional measurements.
- Estimate the non-member/null distribution `Qout` from non-target client measurements.

Core defaults:

- `threshold = 0.5`: neutral decision threshold for aggregate FedMIA confidence.
- `outlier_std_factor = 3.0`: paper-style 3-sigma filter for unusually high non-target measurements.
- `min_variance = 1e-8`: numerical floor so Gaussian CDF scoring never divides by zero.

Measurements used:

- Gradient cosine similarity, matching the FedMIA paper's main Eq. 7 style measurement.
- Later extended experimentally with negative loss, motivated by the FedMIA paper's note that loss can also be used as a measurement.

## 2. Synthetic/Small Smoke Tests

Files:

- `tests/test_tars_smoke.py`
- `tests/test_tars_real_dag_smoke.py`

Purpose:

- Verify that FedMIA scoring works end to end.
- Verify that the BINN data tensors, labels, gradients, updates, reports, and logs can all be wired together.

Important limitation:

- The first smoke test was not a real biological DAG experiment.
- The real-DAG smoke test used the Reactome-derived DAG and real prostate tensors, but remained a tiny execution check, not a privacy-risk estimate.

## 3. First Research Runner: Fixed-Cohort Federated FedMIA

File: `experiments/fedmia_binn_research.py`

Launcher before the LiRA-style protocol:

- `membership_attack.sh` called `experiments/fedmia_binn_research.py`.

Protocol:

- Build the real Reactome feature DAG.
- Load `pnet_x.npy` and `pnet_y.npy`.
- Create a fixed federated split:
  - client `0` is the target client,
  - clients `1..K-1` are non-target reference clients,
  - a fixed holdout set provides nonmember candidates.
- Train repeated FL trajectories.
- In every communication round:
  - train each client locally,
  - compute client update vectors,
  - compute gradient-cosine FedMIA measurements against candidate samples,
  - later also compute negative-loss measurements from local client models.
- Report cosine, loss, and combined FedMIA metrics.

Metrics:

- AUC
- log AUC
- TPR/TNR/FPR/FNR
- precision/recall/F1
- TPR at low FPR
- threshold sweep and best-F1 threshold

Observed issue:

- The run around `logs/fedmia_binn_research_2026-05-09-16-56_e3dfe934.log` reached usable model accuracy, but attack separation stayed weak:
  - `combined_auc` about `0.565`
  - `combined_f1` about `0.667`
  - `combined_best_f1` about `0.681`
- Since `F1 = 0.667` is close to the "predict almost everything as IN" baseline for balanced membership data, this was not strong evidence of a meaningful attack.

Conclusion:

- Useful as a FedMIA paper-style pilot.

## 4. New Runner: LiRA-Style Repeated-Patient FedMIA

This concept was inspired by the central BINN's evaluation method, but the concept itself does not pass the original method of FedMIA itself according to the paper. Therefore this concept was deemed ineffective dead end.

File: `experiments/fedmia_binn_lira_protocol.py`

Current launcher:

- `membership_attack.sh` calls `experiments/fedmia_binn_lira_protocol.py`.

Protocol change:

- Choose a fixed audited patient set.
- For every federated training trajectory, independently assign each audited patient:
  - `IN` with probability `0.5`,
  - `OUT` otherwise.
- If an audited patient is `IN`, place them in the target client.
- If an audited patient is `OUT`, exclude them from all clients.
- Non-target clients are sampled only from non-audited background patients.
- Train the federated BINN as before.
- Measure every audited patient in every run.
- Aggregate each patient's attack outcomes across trajectories.

This mirrors the central BINN report more closely:

- Central report: same patient is IN for some central models and OUT for others.
- New FedMIA protocol: same patient is IN for some FL trajectories and OUT for others.
- Central report: per-patient LiRA metrics are aggregated across patients.
- New FedMIA protocol: per-patient FedMIA metrics are aggregated across patients.

Measurements reported:

- `cosine`: gradient-cosine FedMIA.
- `loss`: negative-loss FedMIA.
- `combined`: average of cosine and loss aggregate FedMIA scores.

Report aggregation:

- Per-patient TPR, TNR, FPR, FNR, precision, recall, F1, AUC, log AUC, TPR at low FPR.
- Mean +/- sample standard deviation across audited patients.
- Pooled metrics across all patient-run decisions are also included for diagnostics.

Default launcher configuration:

- `RUNS=200`
- `ROUNDS=20`
- `LOCAL_EPOCHS=2`
- `NUM_CLIENTS=5`
- `SAMPLES_PER_CLIENT=64`
- `AUDIT_COUNT=64`
- `INCLUSION_PROB=0.5`
- `THRESHOLD=0.5`
- real Reactome feature DAG unless `FEATURE_LIMIT` is set.

Recommended pilot before a long run:

```bash
RUNS=4 ROUNDS=3 LOCAL_EPOCHS=1 AUDIT_COUNT=8 GPU=0 bash membership_attack.sh
```

Recommended central-comparable run shape:

```bash
RUNS=200 ROUNDS=20 LOCAL_EPOCHS=2 AUDIT_COUNT=64 GPU=0 bash membership_attack.sh
```

Open caveat:

- The central report used GO-annotated BINN, while this repository currently constructs a Reactome-derived DAG for the real-data experiment. That difference should be stated whenever comparing absolute metric values.

## 5. Completed 30-Run Baseline And Follow-Up Tweaks

Report:

- `reports/fedmia_binn_lira_protocol_2026-05-09-18-01_9f1a8677.txt`

Configuration:

- `RUNS=30`
- `ROUNDS=10`
- `LOCAL_EPOCHS=2`
- `NUM_CLIENTS=5`
- `SAMPLES_PER_CLIENT=64`
- `AUDIT_COUNT=32`
- full Reactome-mapped feature set

Result summary:

- Training reached a usable regime:
  - final training accuracy: `0.786771 +/- 0.027693`
- FedMIA showed moderate but not central-LiRA-strength leakage:
  - cosine patient mean AUC: `0.676267 +/- 0.120111`
  - cosine patient mean F1: `0.640154 +/- 0.084102`
  - loss patient mean AUC: `0.672130 +/- 0.114619`
  - loss patient mean F1: `0.633185 +/- 0.095442`
  - combined patient mean AUC: `0.675696 +/- 0.117979`
  - combined patient mean F1: `0.631075 +/- 0.087736`
- Low-FPR leakage exists but is not yet strong:
  - combined TPR at FPR 0.1: `0.258814 +/- 0.206680`
  - combined TPR at FPR 0.01: `0.151256 +/- 0.188138`

Interpretation:

- The current 5-client setup likely underestimates FedMIA because `Qout` is estimated from only 4 non-target clients per round.
- The current 10-round setup may also dilute temporal signal because early noisy rounds are averaged equally with later rounds.
- Increasing `AUDIT_COUNT` would make the audit more representative, but it should not be the first lever for attack strength.

Tweaks added after this run:

- `experiments/fedmia_binn_lira_protocol.py` now reports late-round diagnostics:
  - all rounds,
  - last half of communication rounds,
  - last quarter of communication rounds.
- `membership_attack.sh` defaults now target the next ablation:
  - `RUNS=10`
  - `ROUNDS=20`
  - `LOCAL_EPOCHS=2`
  - `NUM_CLIENTS=10`
  - `AUDIT_COUNT=32`

## 6. Round Grid Plan

Rationale:

- The 30-run baseline used only `ROUNDS=10`, which is light for FedMIA.
- FedMIA is explicitly designed to aggregate evidence across communication rounds.
- The paper's default setting uses far more communication rounds, so a round grid is a better next ablation than changing many variables at once.

Grid runner:

- `membership_attack_round_grid.sh`

Default grid:

```bash
RUNS=30 NUM_CLIENTS=10 AUDIT_COUNT=32 LOCAL_EPOCHS=2 ROUND_GRID="20 50 100" GPU=0 bash membership_attack_round_grid.sh
```

Grid cells:

- `ROUNDS=20`, `LOCAL_EPOCHS=2`
- `ROUNDS=50`, `LOCAL_EPOCHS=2`
- `ROUNDS=100`, `LOCAL_EPOCHS=2`

Fixed controls:

- `NUM_CLIENTS=10`, to improve FedMIA's non-target-client `Qout` estimate relative to the 5-client baseline.
- `AUDIT_COUNT=32`, to keep the audited population fixed while testing temporal evidence.
- `LOCAL_EPOCHS=2`, to isolate the effect of more communication rounds.
- full Reactome-mapped feature set unless a debug `FEATURE_LIMIT` is explicitly passed.

What to inspect:

- all-round metrics versus `last_half` and `last_quarter` diagnostics.
- whether patient mean AUC/F1 improves monotonically from 20 to 50 to 100 rounds.
- whether low-FPR TPR improves, especially `tpr_at_fpr_0.1` and `tpr_at_fpr_0.01`.

## 7. Switch Away From Round Grid

Reason:

- The completed 20-round / 10-client grid cell showed that the model trains
  properly, but attack strength stayed close to the earlier baseline:
  - final accuracy around `0.803`
  - patient mean AUC around `0.674`
  - patient mean F1 around `0.64`
  - FPR around `0.44-0.49` at the default `delta=0.5`
- The old `SAMPLES_PER_CLIENT=64` setting was inherited from smoke-test work.
  With `NUM_CLIENTS=10`, that trains on about `640` patient slots, while the
  prostate dataset has `1012` patients.
- For the current audited protocol, `AUDIT_COUNT=32`, leaving about `980`
  non-audited background patients. With `NUM_CLIENTS=10`, a near-full balanced
  allocation is therefore approximately `98` background patients per client.

Decision:

- Stop treating the round grid as the default path.
- Use one serious high-round experiment through `membership_attack.sh`.
- Keep `membership_attack_round_grid.sh` only for explicit future ablations.

Current primary command:

```bash
GPU=0 bash membership_attack.sh
```

Current primary defaults:

- `RUNS=30`
- `ROUNDS=100`
- `LOCAL_EPOCHS=2`
- `NUM_CLIENTS=10`
- `SAMPLES_PER_CLIENT=98`
- `AUDIT_COUNT=32`
- `INCLUSION_PROB=0.5`

## 8. Switch back to FedMIA Paper-Style Baseline

Reason:

- The "LiRA-style" repeated-patient protocol is not the FedMIA paper protocol.
- The baseline we need now is a direct FedMIA-style FL experiment:
  - horizontal FedAvg,
  - one target client,
  - non-target client updates estimate `Qout`,
  - FedMIA-I uses loss measurement,
  - FedMIA-II uses gradient-cosine measurement,
  - metrics prioritize AUC and TPR@low FPR.
- No LiRA/shadow-model membership matrix is used in this baseline.

Implementation:

- New script: `experiments/fedmia_binn_paper_grid.py`
- Main launcher: `membership_attack.sh`
- Logs:
  - `logs/fedmia_binn_paper_grid_<date>_<id>.log`
  - `logs/fedmia_binn_paper_grid_<date>_<id>.jsonl`
- Reports:
  - `reports/fedmia_binn_paper_grid_<date>_<id>.txt`
  - `reports/fedmia_binn_paper_grid_<date>_<id>.csv`

Paper settings represented by the grid:

- client count grid: paper evaluates `5-30` clients.
- communication rounds: paper defaults to `300` synchronous rounds.
- local epochs: paper varies `1-9`.
- Non-IID extent: paper uses Dirichlet `beta = 0.1, 1, 10, infinity`; `infinity` is `iid`.
- sample volume: paper varies samples per client, but BINN has only `1012`
  prostate patients. Therefore the script translates sample count to the
  maximum disjoint patient records per client after the paper's one-tenth
  holdout.

Exact BINN sample-count translation:

- total patients: `1012`
- one-tenth stratified holdout: `101`
- FL training pool: `911`
- max disjoint samples/client:
  - `5` clients: `182`
  - `10` clients: `91`
  - `20` clients: `45`
  - `30` clients: `30`

The default script setting uses `samples_per_client_grid=auto`, which means:

- compute `floor(train_pool / clients)` for each client count,
- use `sample_fraction_grid` to choose a fraction of that maximum,
- default `sample_fraction_grid=1.0`, i.e. use the full available BINN client
  sample volume for that client count.

Plumbing check:

```bash
PLUMBING=1 bash membership_attack.sh
```

Result:

- passed end-to-end on real pnet data and a capped real Reactome DAG.
- report:
  - `reports/fedmia_binn_paper_grid_2026-05-10-10-26_8536e955.txt`
- log:
  - `logs/fedmia_binn_paper_grid_2026-05-10-10-26_8536e955.log`
- CSV:
  - `reports/fedmia_binn_paper_grid_2026-05-10-10-26_8536e955.csv`

Recommended first serious baseline:

```bash
RUNS=30 CLIENT_GRID=10 ROUND_GRID=300 LOCAL_EPOCH_GRID=1 BETA_GRID=iid SAMPLE_FRACTION_GRID=1.0 GPU=0 bash membership_attack.sh
```

Recommended paper ablation grid after the baseline:

```bash
RUNS=5 CLIENT_GRID="5,10,20,30" ROUND_GRID="100,200,300" LOCAL_EPOCH_GRID="1,3,5,9" BETA_GRID="iid,10,1,0.1" SAMPLE_FRACTION_GRID="1.0" GPU=0 bash membership_attack.sh
```

This grid is intentionally expensive. It follows the paper's axes, but the
number of repetitions should be selected based on available GPU time.

## 9. Fix

The implementation above selected non-members incorrectly, and the 300 round runs were probably causing gradient collapse for the model. The next run will be run with the following configuration:

```bash
RUNS=1 CLIENT_GRID="10" ROUND_GRID="10" LOCAL_EPOCH_GRID="2" BETA_GRID="iid" SAMPLE_FRACTION_GRID="1.0" GPU=0 bash membership_attack.sh
```

this resulted with:
fedmia_binn_paper_grid_2026-05-11-07-34_8617e73d.txt

train_acc_final=0.753846 +/- 0.000000
    holdout_acc_final=0.792079 +/- 0.000000
    FedMIA-I loss auc=0.617679 +/- 0.000000 f1=0.546512 +/- 0.000000 tpr@fpr0.001=0.065934
    FedMIA-I pooled auc=0.617679 f1=0.546512 tpr=0.516484 tnr=0.626374 fpr=0.373626 tpr@fpr0.001=0.065934
    FedMIA-II cosine auc=0.608622 +/- 0.000000 f1=0.535714 +/- 0.000000 tpr@fpr0.001=0.000000
    FedMIA-II pooled auc=0.608622 f1=0.535714 tpr=0.494505 tnr=0.648352 fpr=0.351648 tpr@fpr0.001=0.000000

## 10. Increase rounds

```bash
RUNS=1 CLIENT_GRID="10" ROUND_GRID="100" LOCAL_EPOCH_GRID="2" BETA_GRID="iid" SAMPLE_FRACTION_GRID="1.0" GPU=0 bash membership_attack.sh
```



