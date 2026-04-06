import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment(exp_dir: Path):
    train_metrics = load_json(exp_dir / "train_metrics.json")
    best_metrics = load_json(exp_dir / "best_model_metrics.json")
    return train_metrics, best_metrics


def plot_metric_curves(experiments, output_path: Path, metric_key: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for bs_label, train_metrics, _ in experiments:
        epochs = [item["epoch"] for item in train_metrics]
        values = [item[metric_key] for item in train_metrics]
        ax.plot(epochs, values, linewidth=2, label=bs_label)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_metrics_bar(experiments, output_path: Path):
    bs_labels = [bs_label for bs_label, _, _ in experiments]
    acc = [best_metrics["accuracy"] for _, _, best_metrics in experiments]
    precision = [best_metrics["precision"] for _, _, best_metrics in experiments]
    recall = [best_metrics["recall"] for _, _, best_metrics in experiments]
    f1 = [best_metrics["f1"] for _, _, best_metrics in experiments]
    macro_f1 = [best_metrics["macro_f1"] for _, _, best_metrics in experiments]

    x = range(len(bs_labels))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - 2 * width for i in x], acc, width=width, label="Accuracy")
    ax.bar([i - width for i in x], precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar([i + width for i in x], f1, width=width, label="F1")
    ax.bar([i + 2 * width for i in x], macro_f1, width=width, label="Macro F1")

    ax.set_title("Best Metrics by Batch Size")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bs_labels)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary(experiments, output_path: Path):
    lines = []
    for bs_label, _, best_metrics in experiments:
        lines.append(
            f"{bs_label}: "
            f"epoch={best_metrics['epoch']}, "
            f"acc={best_metrics['accuracy']:.4f}, "
            f"precision={best_metrics['precision']:.4f}, "
            f"recall={best_metrics['recall']:.4f}, "
            f"f1={best_metrics['f1']:.4f}, "
            f"macro_f1={best_metrics['macro_f1']:.4f}, "
            f"val_loss={best_metrics['val_loss']:.4f}, "
            f"val_acc={best_metrics['val_acc']:.4f}, "
            f"inference_ms={best_metrics['inference_time_per_sample_ms']:.4f}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot comparison curves for multiple batch sizes.")
    parser.add_argument("--log-root", default="output/logs", help="Root directory containing experiment folders.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["lr_0.0005_bs_16", "lr_0.0005_bs_32", "lr_0.0005_bs_64"],
        help="Experiment folder names to compare.",
    )
    parser.add_argument("--outdir", default="output/figures/bs_comparison", help="Directory to save comparison figures.")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for exp_name in args.experiments:
        exp_dir = log_root / exp_name
        train_metrics, best_metrics = load_experiment(exp_dir)
        bs_label = f"bs={exp_name.split('_bs_')[1]}"
        experiments.append((bs_label, train_metrics, best_metrics))

    plot_metric_curves(experiments, outdir / "train_loss_comparison.png", "train_loss", "Train Loss Comparison", "Train Loss")
    plot_metric_curves(experiments, outdir / "val_loss_comparison.png", "val_loss", "Validation Loss Comparison", "Val Loss")
    plot_metric_curves(experiments, outdir / "train_acc_comparison.png", "train_acc", "Train Accuracy Comparison", "Train Accuracy")
    plot_metric_curves(experiments, outdir / "val_acc_comparison.png", "val_acc", "Validation Accuracy Comparison", "Val Accuracy")
    plot_metric_curves(experiments, outdir / "epoch_time_comparison.png", "epoch_time_seconds", "Epoch Time Comparison", "Seconds")
    plot_best_metrics_bar(experiments, outdir / "best_metrics_comparison.png")
    save_summary(experiments, outdir / "best_metrics_summary.txt")


if __name__ == "__main__":
    main()
