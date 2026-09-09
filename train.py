"""Train one of four DistilBERT experiments on one machine."""

import argparse
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, disable_progress_bars, load_dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging as transformers_logging


DATASET = "goosmanlei/amazon_reviews_multi"
REVISION = "a7b1fa9703f1f930d5c8e1a65c5d2774aaab0c85"
MODEL = "distilbert-base-uncased"
NUM_LABELS = 5
MAX_LENGTH = 128
EPOCHS = 4
BATCH_SIZE = 32
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.001
EXPERIMENTS = {
    "E1": {"name": "full fine-tuning", "kind": "full", "lr": 2e-5},
    "E2": {"name": "frozen backbone", "kind": "frozen", "lr": 5e-4},
    "E3": {"name": "LoRA rank 8", "kind": "lora", "rank": 8, "alpha": 16, "lr": 2e-4},
    "E4": {"name": "LoRA rank 16", "kind": "lora", "rank": 16, "alpha": 32, "lr": 2e-4},
}

disable_progress_bars()
transformers_logging.set_verbosity_error()


def select_device():
    """Require the NVIDIA GPU used by this training project."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this project.")
    return torch.device("cuda")


def prepare_data(tokenizer):
    """Load, deduplicate, and tokenize the three dataset splits."""
    raw = load_dataset(DATASET, "en", revision=REVISION)

    def enrich(row):
        text = f"{row['review_title'] or ''} {row['review_body'] or ''}"
        return {
            "labels": int(row["stars"]) - 1,
            "text_key": " ".join(text.lower().split()),
        }

    raw = DatasetDict({name: split.map(enrich) for name, split in raw.items()})
    # Remove normalized text matches from earlier splits to prevent test leakage.
    test_keys = set(raw["test"]["text_key"])
    validation_keys = set(raw["validation"]["text_key"])
    before = {name: len(split) for name, split in raw.items()}
    raw["train"] = raw["train"].filter(lambda row: row["text_key"] not in test_keys | validation_keys)
    raw["validation"] = raw["validation"].filter(lambda row: row["text_key"] not in test_keys)
    after_dedup = {name: len(split) for name, split in raw.items()}

    def tokenize(batch):
        titles = [value or "" for value in batch["review_title"]]
        bodies = [value or "" for value in batch["review_body"]]
        encoded = tokenizer(titles, bodies, truncation=True, max_length=MAX_LENGTH)
        encoded["labels"] = batch["labels"]
        return encoded

    tokenized = DatasetDict({
        name: split.map(tokenize, batched=True, remove_columns=split.column_names)
        for name, split in raw.items()
    })
    audit = {
        "source": {"dataset": DATASET, "configuration": "en", "revision": REVISION},
        "original_sizes": before,
        "final_sizes": {name: len(split) for name, split in raw.items()},
        "removed_cross_split_duplicates": {
            "train": before["train"] - after_dedup["train"],
            "validation": before["validation"] - after_dedup["validation"],
        },
        "train_size_used": len(raw["train"]),
    }
    return tokenized, audit


def make_model(experiment):
    """Create the classifier and apply the selected adaptation method."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)

    spec = EXPERIMENTS[experiment]
    if spec["kind"] == "frozen":
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False
    elif spec["kind"] == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=spec["rank"],
            lora_alpha=spec["alpha"],
            lora_dropout=0.05,
            target_modules=["q_lin", "v_lin"],
            modules_to_save=["pre_classifier", "classifier"],
            bias="none",
        ))
    return model


def metrics(labels, logits):
    """Convert model scores into the study's evaluation metrics."""
    probabilities = torch.softmax(torch.tensor(logits), 1).numpy()
    predictions = probabilities.argmax(1)
    macro_f1 = precision_recall_fscore_support(
        labels, predictions, labels=range(NUM_LABELS), average="macro", zero_division=0
    )[2]
    distances = np.abs(predictions - labels)
    within_one_star_accuracy = float((distances <= 1).mean())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "within_one_star_accuracy": within_one_star_accuracy,
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=range(NUM_LABELS)
        ).tolist(),
    }


def create_data_loader(dataset, tokenizer, batch_size, training):
    """Build a shuffled training loader or an ordered evaluation loader."""
    columns = ["input_ids", "attention_mask", "labels"]
    dataset = dataset.select_columns(columns)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        num_workers=2, pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate one split and return metrics, loss, and model inference time."""
    model.eval()
    labels, logits = [], []
    local_loss_sum = 0.0
    local_examples = 0
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    for batch in data_loader:
        labels.extend(batch["labels"].tolist())
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(**inputs)
        batch_size = inputs["labels"].shape[0]
        local_loss_sum += float(output.loss) * batch_size
        local_examples += batch_size
        logits.extend(output.logits.float().cpu().tolist())

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - inference_started
    average_loss = local_loss_sum / local_examples
    result = metrics(np.asarray(labels), np.asarray(logits))
    model.train()
    return result, result["macro_f1"], average_loss, elapsed_seconds


def save_trainable(model, path):
    """Save only parameters updated by the selected experiment."""
    state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    torch.save(state, path)


def load_trainable(model, path, device):
    """Restore trainable parameters from the best validation checkpoint."""
    state = torch.load(path, map_location=device, weights_only=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])


def parse_arguments() -> argparse.Namespace:
    """Read experiment settings from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 2026), required=True)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prepared_data(tokenizer):
    """Create the token cache once, then reuse it in later runs."""
    cache = Path("work") / f"prepared_full_v2_{REVISION[:8]}"
    ready_file = cache / "READY"

    if not ready_file.exists():
        data, audit = prepare_data(tokenizer)
        cache.mkdir(parents=True, exist_ok=True)
        data.save_to_disk(cache / "tokens")
        (cache / "audit.json").write_text(json.dumps(audit, indent=2))
        ready_file.touch()

    data = load_from_disk(cache / "tokens")
    return data


def create_loaders(data, tokenizer):
    """Create loaders for training, validation, and testing."""
    train_loader = create_data_loader(
        data["train"], tokenizer, BATCH_SIZE, training=True
    )
    validation_loader = create_data_loader(
        data["validation"], tokenizer, BATCH_SIZE, training=False
    )
    test_loader = create_data_loader(
        data["test"], tokenizer, BATCH_SIZE, training=False
    )
    return train_loader, validation_loader, test_loader


def prepare_model(experiment, device):
    """Build the model on the GPU and count its parameters."""
    model = make_model(experiment).to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return model, total, trainable


def train_model(
    model,
    train_loader,
    validation_loader,
    experiment,
    run_dir,
    device,
):
    """Train the model and save the checkpoint with the best validation macro-F1."""
    parameters = (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=EXPERIMENTS[experiment]["lr"],
        weight_decay=0.01,
    )

    planned_steps = EPOCHS * len(train_loader)
    warmup_steps = round(WARMUP_RATIO * planned_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, planned_steps
    )
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best_score = -1.0
    early_stopping_score = -1.0
    checks_without_improvement = 0
    stopped_early = False
    training_started = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_loss_sum = 0.0
        epoch_examples = 0

        for batch in train_loader:
            inputs = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            # Mixed precision speeds up GPU training and reduces memory use.
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = inputs["labels"].shape[0]
            epoch_loss_sum += float(loss) * batch_size
            epoch_examples += batch_size

        # Validation selects the checkpoint; test data is not used here.
        _, score, validation_loss, _ = evaluate(
            model, validation_loader, device
        )
        history.append({
            "epoch": epoch + 1,
            "training_loss": epoch_loss_sum / epoch_examples,
            "validation_loss": validation_loss,
            "validation_macro_f1": score,
        })

        if score > best_score:
            best_score = score
            save_trainable(model, run_dir / "best.pt")

        if score >= early_stopping_score + EARLY_STOPPING_MIN_DELTA:
            early_stopping_score = score
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
            stopped_early = (
                checks_without_improvement >= EARLY_STOPPING_PATIENCE
            )
        if stopped_early:
            break

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started

    return {
        "history": history,
        "best_score": best_score,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "training_peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def evaluate_best_checkpoint(
    model, test_loader, run_dir, device, training
):
    """Evaluate the selected checkpoint and measure inference efficiency."""
    load_trainable(model, run_dir / "best.pt", device)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    test_metrics, _, _, inference_elapsed = evaluate(
        model, test_loader, device
    )
    training["test"] = test_metrics
    training["inference_ms_per_review"] = (
        1000 * inference_elapsed / len(test_loader.dataset)
    )
    training["inference_peak_memory_bytes"] = (
        torch.cuda.max_memory_allocated(device)
    )


def build_result(args, training, total_parameters, trainable_parameters):
    """Assemble the compact JSON summary needed by the report."""
    return {
        "experiment": args.experiment,
        "seed": args.seed,
        "epochs_completed": training["epochs_completed"],
        "training_seconds": training["training_seconds"],
        "training_peak_vram_gb": (
            training["training_peak_memory_bytes"] / 1e9
        ),
        "inference_ms_per_review": training["inference_ms_per_review"],
        "inference_peak_vram_gb": (
            training["inference_peak_memory_bytes"] / 1e9
        ),
        "best_validation_macro_f1": training["best_score"],
        "history": training["history"],
        "test": training["test"],
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "trainable_percent": 100 * trainable_parameters / total_parameters,
        },
    }


def save_run_outputs(result, run_dir, device):
    """Write compact final run metadata."""
    result["hardware"] = torch.cuda.get_device_name(device)
    result["software"] = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "peft": importlib.metadata.version("peft"),
        "mlflow": importlib.metadata.version("mlflow"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))

def run_experiment(args, device):
    """Run one seed and save the local inputs needed for aggregation."""
    set_random_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    run_dir = Path("results/runs") / f"{args.experiment}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = load_prepared_data(tokenizer)
    train_loader, validation_loader, test_loader = create_loaders(data, tokenizer)
    model, total_parameters, trainable_parameters = prepare_model(
        args.experiment, device
    )

    training = train_model(
        model,
        train_loader,
        validation_loader,
        args.experiment,
        run_dir,
        device,
    )
    evaluate_best_checkpoint(model, test_loader, run_dir, device, training)
    result = build_result(
        args,
        training,
        total_parameters,
        trainable_parameters,
    )

    save_run_outputs(result, run_dir, device)


def main():
    """Coordinate data preparation, training, evaluation, and output writing."""
    args = parse_arguments()
    device = select_device()
    run_experiment(args, device)


if __name__ == "__main__":
    main()
