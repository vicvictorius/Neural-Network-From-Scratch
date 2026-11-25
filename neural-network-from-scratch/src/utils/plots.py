"""Gráficos padronizados para o README e para os notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # funciona sem interface gráfica (CI, servidores)
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: dict[str, list[float]], path: str | Path, title: str = ""
) -> None:
    """history precisa de train_loss, val_loss, train_acc, val_acc."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="treino")
    ax1.plot(epochs, history["val_loss"], label="validação")
    ax1.set(xlabel="época", ylabel="perda", title="Perda")
    ax2.plot(epochs, [100 * a for a in history["train_acc"]], label="treino")
    ax2.plot(epochs, [100 * a for a in history["val_acc"]], label="validação")
    ax2.set(xlabel="época", ylabel="acurácia (%)", title="Acurácia")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str | Path) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="previsto", ylabel="verdadeiro", title="Matriz de confusão (por linha)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_image_grid(
    images: np.ndarray, titles: list[str], path: str | Path, ncols: int = 8
) -> None:
    """images: (n, 784) ou (n, 28, 28)."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.9 * nrows))
    for ax, img, title in zip(np.ravel(axes), images, titles, strict=False):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(title, fontsize=7)
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_experiment_comparison(
    names: list[str], means: list[float], stds: list[float], path: str | Path
) -> None:
    """Barras horizontais com média e desvio padrão (acurácia de validação, %)."""
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
    ax.barh(
        [names[i] for i in order],
        [means[i] for i in order],
        xerr=[stds[i] for i in order],
        color="#4C72B0",
        capsize=4,
    )
    ax.set(xlabel="acurácia de validação (%)", title="Experimentos (média de 3 seeds)")
    ax.set_xlim(min(means) - 2, min(100, max(means) + 1.5))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
