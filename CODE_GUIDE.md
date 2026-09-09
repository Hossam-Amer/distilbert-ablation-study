# Code and data guide

This guide explains the variables, data flow, PyTorch calls, and functions that
remain in `train.py`, `analyze.py`, `run.sh`, and `data_exploration.ipynb`.

## 1. End-to-end flow

1. Load the pinned English Amazon Reviews Multi dataset.
2. Convert star ratings 1–5 into class labels 0–4.
3. Remove normalized review duplicates across splits to prevent leakage.
4. Tokenize each title/body pair and truncate it to 128 tokens.
5. Train one experiment on one NVIDIA GPU using mixed precision.
6. Evaluate validation macro-F1 once per epoch.
7. Stop when validation improvement stalls and restore the best checkpoint.
8. Evaluate the test split once and save its local seed result.
9. Average both seeds, log one MLflow run per experiment, and create combined confusion matrices.

## 2. Main constants and arguments

| Variable | Practical role | Theoretical meaning |
|---|---|---|
| `DATASET` | Hugging Face dataset identifier. | Defines the source population. |
| `REVISION` | Pins one dataset commit. | Makes the input data reproducible. |
| `MODEL` | Selects DistilBERT and its tokenizer. | Supplies pretrained language representations. |
| `NUM_LABELS` | Creates five classifier outputs. | One class per star rating. |
| `MAX_LENGTH` | Truncates inputs to 128 tokens. | Bounds attention compute and memory. |
| `EPOCHS` | Allows at most four passes through training data. | Sets maximum training exposure. |
| `BATCH_SIZE` | Uses 32 examples per optimization step. | Affects gradient noise and GPU memory. |
| `WARMUP_RATIO` | Warms up for 6% of planned steps. | Reduces unstable early updates. |
| `MAX_GRAD_NORM` | Clips the gradient norm at 1.0. | Limits unusually large updates. |
| `EARLY_STOPPING_PATIENCE` | Allows two non-improving epochs. | Prevents unnecessary later epochs. |
| `EARLY_STOPPING_MIN_DELTA` | Requires a macro-F1 gain of 0.001. | Ignores tiny score fluctuations. |
| `MLFLOW_EXPERIMENT` | Names the local MLflow experiment. | Holds four averaged experiment runs. |
| `EXPERIMENTS` | Stores E1–E4 settings. | Defines the controlled ablation. |
| `args.experiment` | Chooses E1, E2, E3, or E4. | Selects the adaptation method. |
| `args.seed` | Chooses seed 42 or 2026. | Measures limited run variability. |

Results always go to `results/runs`; MLflow always uses
`results/mlflow.db` and `results/mlartifacts`.

## 3. Dataset and preprocessing

| Variable or call | Practical role | Theoretical meaning |
|---|---|---|
| `raw` | Holds train, validation, and test splits. | Separates fitting, selection, and testing. |
| `row` | One source review. | One supervised observation. |
| `labels` | Zero-based star classes. | Ground-truth targets for cross-entropy. |
| `text_key` | Normalized title/body text. | Detects exact normalized duplicates. |
| `test_keys`, `validation_keys` | Held-out text-key sets. | Support cross-split leakage removal. |
| `before`, `after_dedup` | Split sizes around cleaning. | Audit preprocessing effects. |
| `titles`, `bodies` | Text lists with missing values filled. | Preserve usable review content. |
| `encoded` | Token IDs and attention masks. | Numeric model input. |
| `audit` | Source version and cleaning counts. | Records data provenance. |
| `load_dataset(...)` | Downloads the pinned dataset. | Acquires observations. |
| `Dataset.map(...)` | Adds labels/keys or tokenizes batches. | Deterministic feature preparation. |
| `Dataset.filter(...)` | Removes cross-split duplicates. | Reduces evaluation leakage. |
| `tokenizer(..., truncation=True)` | Encodes at most 128 tokens. | Applies the vocabulary and input limit. |
| `save_to_disk` / `load_from_disk` | Writes or reuses prepared tokens. | Avoids repeating deterministic work. |

The cache is stored under `work/prepared_full_v2_<revision>`. Deleting it is
safe; the next run recreates it.

## 4. Dataset and DataLoader values

| Name | Shape or type | Meaning |
|---|---|---|
| `input_ids` | batch × sequence | Vocabulary indices. |
| `attention_mask` | batch × sequence | Marks real tokens instead of padding. |
| `labels` | batch | Correct class indices. |
| `batch` | dictionary of tensors | One minibatch produced by the loader. |
| `inputs` | GPU-resident dictionary | Values passed to `model(**inputs)`. |
| `logits` | batch × 5 | Unnormalized class scores. |
| `probabilities` | examples × 5 | Softmax-normalized scores. |
| `predictions` | examples | Index of the largest probability. |

`DataLoader` shuffles only training. `DataCollatorWithPadding` dynamically pads
each batch to a multiple of eight. `num_workers=2` overlaps input preparation
with compute, and `pin_memory=True` supports faster GPU transfers. Truncation
removes excess tokens; padding adds ignored positions to shorter examples.

## 5. Models and adaptation

| Setting or call | Practical role | Theoretical meaning |
|---|---|---|
| `AutoModelForSequenceClassification` | Loads DistilBERT with five outputs. | Transfers pretrained representations. |
| E1 full fine-tuning | Trains every parameter. | Maximum adaptation capacity. |
| E2 frozen backbone | Trains only the classifier. | Uses a fixed feature extractor. |
| E3 LoRA rank 8 | Trains rank-8 q/v adapters and head. | Learns low-rank attention updates. |
| E4 LoRA rank 16 | Trains rank-16 q/v adapters and head. | Tests additional adapter capacity. |
| `requires_grad=False` | Freezes backbone parameters. | Excludes weights from gradient updates. |
| `LoraConfig` | Defines adapter rank, scale, targets, and dropout. | Parameterizes a low-rank update. |
| `total_parameters` | Counts all model parameters. | Measures model size. |
| `trainable_parameters` | Counts gradient-enabled parameters. | Measures adaptation cost. |

The model minimizes multiclass cross-entropy. Ordinal tolerance is assessed
afterward with within-one-star accuracy.

## 6. Training variables and PyTorch calls

| Variable or call | Practical role | Theory |
|---|---|---|
| `optimizer` | AdamW over trainable parameters. | Adaptive updates with weight decay. |
| `scheduler` | Warm-up followed by linear decay. | Changes update size over training. |
| `scaler` | Mixed-precision gradient scaler. | Reduces FP16 underflow risk. |
| `loss` | Scalar cross-entropy objective. | Quantity differentiated during learning. |
| `history` | Per-epoch train loss, validation loss, and macro-F1. | Seed histories are averaged before MLflow logging. |
| `best_score` | Best validation result. | Selects without test leakage. |
| `checks_without_improvement` | Early-stopping counter. | Stops prolonged non-improvement. |
| `optimizer.zero_grad(set_to_none=True)` | Clears old gradients. | Prevents gradient accumulation. |
| `torch.autocast("cuda", dtype=torch.float16)` | Runs suitable operations in FP16. | Improves speed and memory efficiency. |
| `model(**inputs).loss` | Runs forward computation and loss. | Connects predictions with targets. |
| `scaler.scale(loss).backward()` | Computes scaled gradients. | Applies backpropagation. |
| `clip_grad_norm_` | Limits the combined gradient norm. | Controls unstable updates. |
| `scaler.step(optimizer)` | Applies a finite optimizer update. | Changes trainable parameters. |
| `scheduler.step()` | Advances the learning-rate schedule. | Sets the next update rate. |
| `torch.cuda.synchronize()` | Waits for queued GPU work. | Makes runtime measurement meaningful. |

There is no gradient accumulation. Validation selects the checkpoint; test data
is used only after training and selection are complete.

## 7. Evaluation and metrics

| Metric | Interpretation |
|---|---|
| Macro-F1 | Primary metric; averages class F1 equally. Higher is better. |
| Accuracy | Exact correct predictions divided by all predictions. Higher is better. |
| Within-one-star accuracy | Fraction no more than one star away. Higher is better. |
| Mean absolute star error | Average ordinal prediction distance. Lower is better. |
| Confusion matrix | Rows are true classes and columns are predicted classes. |

`model.eval()` changes dropout behavior, while `@torch.no_grad()` disables
gradient recording. `torch.softmax` converts logits to probabilities and
`argmax` selects the class. The confusion matrix remains because it shows which
ratings the model confuses.

## 8. Checkpoints, GPU measurements, and MLflow

`best.pt` stores only parameters with `requires_grad=True`. Reloading it requires
the same model and adaptation method. It is not an optimizer-resume checkpoint.

`torch.cuda.reset_peak_memory_stats` and `max_memory_allocated` measure separate
training and inference memory peaks. The timed test pass records milliseconds
per review at the configured batch size. Training
gradients are cleared before measuring inference memory. For each experiment,
MLflow logs the mean training loss, validation loss, and validation macro-F1
only for epochs completed by both seeds.

MLflow is local-only. Run its interface with:

```bash
mlflow ui --backend-store-uri sqlite:///results/mlflow.db --host 127.0.0.1 --port 5000
```

## 9. Function summary

| Function | Responsibility |
|---|---|
| `select_device` | Require and select CUDA. |
| `prepare_data` | Load, clean, tokenize, and audit data. |
| `make_model` | Build the selected adaptation. |
| `metrics` | Compute accuracy, macro-F1, within-one-star accuracy, and the confusion matrix. |
| `create_data_loader` | Create a training or evaluation loader. |
| `evaluate` | Compute full-split metrics and average loss. |
| `save_trainable` / `load_trainable` | Save and restore the best weights. |
| `parse_arguments` | Read the experiment and seed. |
| `set_random_seed` | Seed Python, NumPy, and PyTorch. |
| `analyze.configure_mlflow` | Configure local SQLite MLflow. |
| `load_prepared_data` | Build or reuse the token cache. |
| `create_loaders` | Build train, validation, and test loaders. |
| `prepare_model` | Move the model to CUDA and count parameters. |
| `train_model` | Optimize and select the best checkpoint. |
| `evaluate_best_checkpoint` | Test the checkpoint and measure VRAM. |
| `build_result` / `save_run_outputs` | Assemble and save the run record. |
| `run_experiment` / `main` | Coordinate one run. |
| `analyze.load_results` | Load eight current-format results. |
| `analyze.plot_confusion_matrices` | Combine both seeds and create the retained plot. |
| `analyze.log_experiments_to_mlflow` | Log one two-seed-average run per experiment. |

## 10. Output files

| Artifact | Contents |
|---|---|
| `work/prepared_full_v2_.../tokens` | Cached tokenized splits. |
| `work/prepared_full_v2_.../audit.json` | Dataset source and cleaning counts. |
| `best.pt` | Best trainable weights. |
| `metrics.json` | Quality, timing, memory, software, and selection data. |
| `results/mlflow.db` | Local MLflow database. |
| `results/mlartifacts` | Files stored with MLflow runs. |
| `confusion_matrices.png` | Four confusion matrices combined across both seeds. |

## 11. Exploration notebook

`data_exploration.ipynb` is separate from training. It investigates missing
text, rating validity, duplicates, leakage, class balance, token lengths, and
truncation. Its saved output found 33 missing training titles, no empty combined
reviews, and no invalid ratings. Cleaning removed 16 cross-split training rows
while keeping class shares close to 20%. At 128 tokens, about 4.75% of cleaned
training reviews are truncated.

Notebook assertions check rating validity, leakage removal, and token limits;
they do not prove model quality.
