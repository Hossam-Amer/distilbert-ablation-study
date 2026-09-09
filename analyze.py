"""Turn the eight run folders into the tables and plots used in the interview."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from train import (
    BATCH_SIZE,
    DATASET,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EXPERIMENTS,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    MODEL,
    REVISION,
    WARMUP_RATIO,
)


RESULTS = Path("results")
RUNS = RESULTS / "runs"
OUT = RESULTS / "report"
SEEDS = (42, 2026)
MLFLOW_EXPERIMENT = "distilbert-ablation"
CORE_METRICS = [
    "macro_f1", "accuracy", "within_one_star_accuracy",
]

def configure_mlflow():
    """Use the project's local SQLite MLflow store."""
    results_directory = RESULTS.resolve()
    results_directory.mkdir(parents=True, exist_ok=True)
    database_path = (results_directory / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=(results_directory / "mlartifacts").as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def load_results():
    """Load seed results for aggregation by experiment."""
    rows = []
    history_rows = []
    confusions = {}
    metadata = None

    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            run_directory = RUNS / f"{experiment}_seed{seed}"
            saved = json.loads((run_directory / "metrics.json").read_text())
            if metadata is None:
                metadata = saved

            row = {
                "experiment": experiment,
                "seed": seed,
                "epochs_completed": saved["epochs_completed"],
                **{
                    name: saved["test"][name]
                    for name in CORE_METRICS
                },
                "trainable_parameters": saved["parameters"]["trainable"],
                "trainable_percent": saved["parameters"]["trainable_percent"],
                "training_minutes": saved["training_seconds"] / 60,
                "training_peak_vram_gb": saved["training_peak_vram_gb"],
                "inference_ms_per_review": saved["inference_ms_per_review"],
                "inference_peak_vram_gb": saved["inference_peak_vram_gb"],
            }
            rows.append(row)
            for epoch_result in saved["history"]:
                history_rows.append({
                    "experiment": experiment,
                    "seed": seed,
                    **epoch_result,
                })

            confusion = np.asarray(saved["test"]["confusion_matrix"])
            confusions[experiment] = confusions.get(experiment, 0) + confusion

    return pd.DataFrame(rows), pd.DataFrame(history_rows), confusions, metadata


def plot_confusion_matrices(confusions):
    """Plot one confusion matrix per experiment, combined across both seeds."""
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))

    for axis, (experiment, confusion) in zip(axes.flat, confusions.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=range(1, 6),
        ).plot(ax=axis, cmap="Blues", colorbar=False)
        axis.set(
            title=f"{experiment}: {EXPERIMENTS[experiment]['name']}",
            xlabel="Predicted",
            ylabel="True",
        )

    figure.tight_layout()
    figure.savefig(OUT / "confusion_matrices.png", dpi=160)
    plt.close(figure)


def log_experiments_to_mlflow(results, histories, metadata):
    """Create one aggregate MLflow run for each experiment."""
    configure_mlflow()
    for experiment, specification in EXPERIMENTS.items():
        experiment_rows = results[results.experiment == experiment]
        averages = {
            "macro_f1": experiment_rows.macro_f1.mean(),
            "accuracy": experiment_rows.accuracy.mean(),
            "within_one_star_accuracy": (
                experiment_rows.within_one_star_accuracy.mean()
            ),
            "training_minutes": experiment_rows.training_minutes.mean(),
            "epochs_completed": experiment_rows.epochs_completed.mean(),
            "trainable_parameters": experiment_rows.trainable_parameters.mean(),
            "trainable_percent": experiment_rows.trainable_percent.mean(),
            "training_peak_vram_gb": (
                experiment_rows.training_peak_vram_gb.mean()
            ),
            "inference_ms_per_review": (
                experiment_rows.inference_ms_per_review.mean()
            ),
            "inference_peak_vram_gb": (
                experiment_rows.inference_peak_vram_gb.mean()
            ),
        }

        parameters = {
            "method": specification["name"],
            "seeds": ",".join(map(str, SEEDS)),
            "seeds_averaged": len(SEEDS),
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "model": MODEL,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": specification["lr"],
            "warmup_ratio": WARMUP_RATIO,
            "max_gradient_norm": MAX_GRAD_NORM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        }
        if specification["kind"] == "lora":
            parameters["lora_rank"] = specification["rank"]

        with mlflow.start_run(
            run_name=experiment,
            tags={"run_type": "seed-average"},
        ):
            mlflow.log_params(parameters)
            mlflow.log_metrics(averages)

            experiment_history = histories[
                histories.experiment == experiment
            ]
            for epoch, epoch_rows in experiment_history.groupby("epoch"):
                if epoch_rows.seed.nunique() != len(SEEDS):
                    continue
                mlflow.log_metrics(
                    {
                        "training_loss": epoch_rows.training_loss.mean(),
                        "validation_loss": epoch_rows.validation_loss.mean(),
                        "validation_macro_f1": (
                            epoch_rows.validation_macro_f1.mean()
                        ),
                    },
                    step=int(epoch),
                )

            mlflow.set_tag("hardware", metadata["hardware"])
            mlflow.log_params({
                f"{name}_version": version
                for name, version in metadata["software"].items()
            })
            mlflow.log_artifact(
                OUT / "confusion_matrices.png", artifact_path="comparison"
            )


def main():
    """Log four aggregate experiment runs and their confusion matrices."""
    OUT.mkdir(parents=True, exist_ok=True)
    results, histories, confusions, metadata = load_results()
    plot_confusion_matrices(confusions)
    log_experiments_to_mlflow(results, histories, metadata)
    print("Four experiment averages logged to MLflow.")


if __name__ == "__main__":
    main()
