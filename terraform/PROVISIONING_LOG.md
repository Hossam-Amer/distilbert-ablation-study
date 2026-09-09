# GCP provisioning attempts

Last checked: 2026-09-05

Project: `ablation-study-507710`

Requested machine:

- `n1-standard-16`
- 2 x `nvidia-tesla-t4`
- standard provisioning
- 100 GB balanced persistent disk

## Attempt results

| Zone | Region | Result |
|---|---|---|
| `europe-west1-b` | Belgium | Failed: insufficient resources for an N1 VM with two T4 GPUs. |
| `europe-west1-c` | Belgium | Failed: insufficient resources for an N1 VM with two T4 GPUs. |
| `europe-west1-d` | Belgium | Failed: insufficient resources for an N1 VM with two T4 GPUs. |
| `europe-west3-b` | Frankfurt | Failed: insufficient resources for an N1 VM with two T4 GPUs. |
| `europe-central2-b` | Warsaw | Failed: regional `NVIDIA_T4_GPUS` quota is 1; request requires 2. |

The capacity errors came from the Compute Engine API after Terraform attempted
creation. Terraform did not create a partial VM in any zone.

## Quota recheck

The following values were returned by `gcloud compute regions describe`:

| Region | T4 limit | T4 used | L4 limit | A100 limit |
|---|---:|---:|---:|---:|
| `europe-west1` | 1 | 0 | 1 | 0 |
| `europe-west3` | 1 | 0 | 1 | 0 |
| `europe-central2` | 1 | 0 | 1 | 0 |

`gcloud compute project-info describe` reported `GPUS_ALL_REGIONS` with a
limit of 0 and usage of 0. A regional T4 limit of 1 already prevents the
requested two-GPU VM even if zonal capacity becomes available.

## Current state

`gcloud compute instances list` returned no instances. Terraform state contains
only:

```text
data.google_compute_image.deep_learning
google_project_service.compute
```

No VM, GPU, or boot disk was created. The Compute Engine API remains enabled.
The local `terraform.tfvars` currently points to `europe-central2-b`, where the
quota error was explicit.

## Required resolution

Request both of the following for the same chosen region before retrying the
two-GPU configuration:

- regional `NVIDIA_T4_GPUS` limit of at least 2
- global `GPUS_ALL_REGIONS` limit of at least 2

Alternatively, changing the study to one GPU requires changing its execution
and effective batch-size design; that has not been done automatically.

## 2026-09-05 MLflow run attempt

Terraform retried `europe-central2-b` with the original two-T4 configuration.
The Compute Engine API rejected it with:

```text
Quota 'NVIDIA_T4_GPUS' exceeded. Limit: 1.0 in region europe-central2.
metric name = compute.googleapis.com/nvidia_t4_gpus
limit name = NVIDIA-T4-GPUS-per-project-region
```

No VM was created. To run the study within the available regional quota, the
Terraform accelerator count was then reduced to one. The project now requires
one CUDA GPU and runs one Python training process.

The one-T4 retry passed the regional quota check but failed at the project-wide
limit:

```text
Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
metric name = compute.googleapis.com/gpus_all_regions
limit name = GPUS-ALL-REGIONS-per-project
```

The Cloud Quotas API was enabled and a quota preference was submitted:

| Field | Value |
|---|---|
| Preference | `gpu-all-regions-1` |
| Requested value | 1 |
| Granted value at submission | 0 |
| State | Pending (`reconciling=true`) |
| Trace ID | `cfee3e29-1c7e-4c4a-ae1a-973825375314` |
| Submitted | `2026-09-05T11:21:53.457680120Z` |

No GPU VM can be created until Google grants this global quota preference.

## Quota approval and successful launch

Google approved the global quota preference automatically:

| Field | Value |
|---|---|
| Granted value | 1 |
| State detail | `Quota request approved to 1` |
| Updated | `2026-09-05T11:21:56.724550639Z` |

Terraform then created `ablation-study-vm` successfully in
`europe-central2-b` with one Tesla T4. The machine reported Ubuntu 22.04,
Python 3.10.12, and about 80 GB free after boot.

Environment setup required installing Ubuntu's `python3-venv` package. The
pinned Python dependencies then installed successfully, including PyTorch
2.5.1+cu124 and MLflow 3.15.2.

The first smoke attempt exposed that MLflow 3.15 no longer accepts a new file
store by default. Tracking was changed to the supported SQLite backend at
`results/mlflow.db`, with artifacts at `results/mlartifacts`. The repeated
two-step E3 smoke test completed successfully on the T4.

The complete eight-run suite was started in the background with PID `4855`.
Its combined output is stored at `results/suite.log`; each run also has a
separate `console.log` and MLflow record. The first active job was E1, seed 42,
and `nvidia-smi` showed its Python process using the T4.

## 2026-09-06 Spot L4 replacement

The interrupted T4 rerun was stopped before changing infrastructure. Terraform
was changed to the lower-cost Spot `g2-standard-4` shape with one NVIDIA L4,
4 vCPUs, 16 GB system memory, and a 50 GB balanced persistent boot disk.

The project had a `PREEMPTIBLE_NVIDIA_L4_GPUS` quota of 1 in each region
checked. Creation attempts in `europe-west4-a`, `europe-west4-b`, and
`europe-west4-c` failed because those zones had no available Spot L4 capacity.
Create-before-destroy kept the stopped T4 instance intact during those failures.

Creation then succeeded in `europe-west3-a` (Frankfurt):

```text
Instance: ablation-study-l4-spot
Machine:  g2-standard-4
GPU:      NVIDIA L4 (23034 MiB reported by nvidia-smi)
Model:    SPOT, termination action STOP
```

After successful L4 creation, Terraform destroyed the old stopped T4 VM. A
fresh virtual environment was installed successfully and the eight-run suite
started with PID `3290`. The first active job was E1, seed 42; the L4 reached
about 97% utilization while using about 2.25 GiB of GPU memory.

A one-time watcher was started on the VM to shut it down automatically when
the suite controller exits. This prevents compute and GPU billing from
continuing after either successful completion or a training failure. The boot
disk remains available so results can be downloaded after restarting the VM.

### Completion and download

The suite completed all eight experiment/seed runs and `analyze.py` logged one
finished `comparison-report` run to MLflow. The VM stopped automatically at
2026-09-06 18:36 Berlin time. It was restarted briefly to download the output
and stopped again at 20:54 Berlin time.

The verified local copy is `gcp-results-l4/` and contains 22 files totaling
about 529 MB: eight `metrics.json` files, eight checkpoints, `suite.log`, the
SQLite MLflow database, and the confusion-matrix artifact.
