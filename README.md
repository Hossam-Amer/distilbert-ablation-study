# Efficient Adaptation of DistilBERT

See [CODE_GUIDE.md](CODE_GUIDE.md) for variable definitions, function summaries,
PyTorch calls, and practical/theoretical explanations of the data and training flow.

This small study asks one question:

> How much trainable capacity is needed for five-class Amazon review sentiment,
> and can LoRA approach full fine-tuning with less memory?

The task is deliberately appropriate for the model: predict 1–5 stars from an
English review title and body with `distilbert-base-uncased`. It is multiclass
and ordinal, so confusing 4 with 5 stars is less serious than confusing 1 with
5 stars.

## Experiments

| ID | Method | Learning rate | What trains |
|---|---|---:|---|
| E1 | Full fine-tuning | 2e-5 | All parameters |
| E2 | Frozen backbone | 5e-4 | Classification head |
| E3 | LoRA rank 8, alpha 16 | 2e-4 | q/v adapters and head |
| E4 | LoRA rank 16, alpha 32 | 2e-4 | q/v adapters and head |

Everything else is fixed: up to four epochs, seeds 42 and 2026, batch size 32,
maximum length 128, AdamW, 6% warm-up, weight decay 0.01, gradient clipping 1.0,
FP16, and cross-entropy. Validation macro-F1 chooses the best checkpoint.

Hypotheses:

- E1 should give the best quality but use the most memory.
- E2 should struggle most with neutral and neighboring ratings.
- E3 should offer the best quality–efficiency trade-off.
- E4 tests whether a larger LoRA rank improves quality over E3.
- Most mistakes should be between adjacent star ratings.

QLoRA and more ranks are excluded because DistilBERT already fits comfortably
on the target GPUs; quantization would add complexity without answering the
capacity question more cleanly.

## Data

The code pins the English configuration of
[Amazon Reviews Multi](https://huggingface.co/datasets/goosmanlei/amazon_reviews_multi)
at revision `a7b1fa9703f1f930d5c8e1a65c5d2774aaab0c85`: 200,000 train,
5,000 validation, and 5,000 test reviews. It removes normalized title/body
duplicates from the earlier split while always preserving the test copy. It
records split sizes and duplicate counts once in the shared cache's `audit.json`. Reviews and model
weights are not committed.

## The whole implementation

```text
train.py       data, E1–E4 models, metrics and local seed results
analyze.py     four averaged MLflow runs and confusion matrices
run.sh         eight training runs and four averaged MLflow runs
requirements.txt
```

There are no YAML files, custom frameworks, trackers, or class hierarchies.
The experiment table is the `EXPERIMENTS` dictionary at the top of `train.py`.

## Run it

For a separate walkthrough of data quality, duplicate removal, token lengths,
and truncation, open `data_exploration.ipynb`. Install the project requirements
and JupyterLab (`pip install jupyterlab`), then run
`jupyter lab data_exploration.ipynb`. The notebook needs no GPU and does not modify
training caches.

Use Python 3.10 or 3.11 on a Linux host with one NVIDIA GPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

One real run is:

```bash
python train.py --experiment E3 --seed 42
```

The implementation intentionally runs on one machine.

To run the complete study:

```bash
bash run.sh
```

On a remote VM, start it independently of the SSH session with:

```bash
bash start_gcp.sh
tail -f results/suite.log
```

Start the VM's local-only MLflow interface with `bash start_mlflow_ui.sh`, then
reach it through an SSH port-forward rather than exposing port 5000 publicly.

For an automated Google Cloud GPU VM, follow
[terraform/README.md](terraform/README.md).

`run.sh` trains all eight experiment/seed combinations on the full training
set after deduplication. Individual seed results stay in local JSON files;
MLflow receives four runs, one per experiment, containing only two-seed averages.
If using a
rented machine, remember that stopping the script may not stop provider billing.

Start the local MLflow interface after a run:

```bash
mlflow ui --backend-store-uri sqlite:///results/mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`.

## What is measured

- Quality: macro-F1 (primary), exact accuracy, and within-one-star accuracy.
- Training efficiency: trainable parameters, total training time, and peak VRAM.
- Inference efficiency: milliseconds per review and peak VRAM.
- Diagnosis: confusion matrices.

Metrics are computed once over the full evaluation split on one GPU.
Training allows up to four epochs and checks validation macro-F1 once per epoch.
It stops after two consecutive epochs without an improvement of at least 0.001,
then restores the checkpoint with the highest validation score.
The test split is evaluated only after model selection.
Prepared data is cached under `work/prepared_full_v2_<revision>`; the first run
builds this cache, and subsequent runs reuse it.

Each MLflow run records its experiment's fixed hyperparameters and two-seed averages of per-epoch
losses, validation macro-F1, final quality, and training/inference efficiency.
It also stores hardware, software versions, and combined confusion matrices.
Individual checkpoints and seed-level JSON files stay local.

After the suite, `analyze.py` writes only the confusion matrices to
`results/report/`; the averaged comparison lives in MLflow. No standard
deviation is reported because two seeds are insufficient for a strong estimate.

## Five-day interview plan

1. Day 1: read the two Python files, explain the question, inspect data/audit.
2. Day 2: trace one batch through tokenization, loss,
   and macro-F1.
3. Day 3: rent GPUs, run `run.sh`, copy results, terminate the instance.
4. Day 4: compare the MLflow averages and inspect confusion matrices,
   and explain the quality and training-efficiency trade-offs.
5. Day 5: rehearse a five-minute story: question → controls → results →
   errors → limitations. Be ready to explain every line you kept.

The strongest presentation is not “I built a framework.” It is “I designed one
controlled comparison, implemented only what it needed, and can explain every
trade-off.”
