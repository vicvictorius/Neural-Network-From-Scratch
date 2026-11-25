"""Análise e interpretabilidade do modelo final (Fase 3, semana 8).

Uso:
    python -m src.torch_nn.interpret --checkpoint checkpoints/E6_melhor.pt --out results/interpret

Gera: filtros da 1a camada, mapas de ativação, erros mais confiantes e Grad-CAM.
Usa o conjunto de VALIDAÇÃO, para não "gastar" o teste.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import CLASS_NAMES, load_fashion_mnist
from src.utils.plots import plot_image_grid

from .models import CNN, build_model


def load_model(checkpoint: str | Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["config"]


def plot_first_layer_filters(model: CNN, path: Path) -> None:
    w = model.features[0].weight.detach().cpu().numpy()[:, 0]  # (C, 3, 3)
    n = len(w)
    cols = 8
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols, n / cols + 0.5))
    for ax, f in zip(np.ravel(axes), w, strict=False):
        ax.imshow(f, cmap="RdBu", vmin=-np.abs(w).max(), vmax=np.abs(w).max())
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.suptitle("Filtros 3x3 da primeira camada (azul = positivo)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_maps(model: CNN, image: np.ndarray, path: Path, n_maps: int = 16) -> None:
    x = torch.from_numpy(image).reshape(1, 1, 28, 28)
    with torch.no_grad():
        maps = model.features[: 2 if isinstance(model.features[1], torch.nn.ReLU) else 3](x)[0]
    fig, axes = plt.subplots(2, n_maps // 2 + 1, figsize=(n_maps // 2 + 1, 2.4))
    axes = np.ravel(axes)
    axes[0].imshow(image.reshape(28, 28), cmap="gray")
    axes[0].set_title("entrada", fontsize=7)
    for ax, m in zip(axes[1:], maps[:n_maps], strict=False):
        ax.imshow(m.numpy(), cmap="viridis")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Mapas de ativação da primeira camada")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grad_cam(model: CNN, x: torch.Tensor, target: int | None = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017): pondera os mapas da última camada
    convolucional pela média dos gradientes da classe alvo."""
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        store["act"] = out

    handle = model.last_conv_activation.register_forward_hook(hook)
    logits = model(x.reshape(1, 1, 28, 28))
    handle.remove()
    target = int(logits.argmax()) if target is None else target
    model.zero_grad()
    logits[0, target].backward()
    act, grad = store["act"][0], store["act"].grad[0]  # (C, h, w)
    weights = grad.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = F.interpolate(cam[None, None], size=(28, 28), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    return cam / (cam.max() + 1e-8)


def plot_grad_cam(model: CNN, images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 4.2))
    for i in range(n):
        x = torch.from_numpy(images[i])
        cam = grad_cam(model, x)
        pred = int(model(x.reshape(1, 1, 28, 28)).argmax())
        axes[0, i].imshow(images[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title(f"real: {CLASS_NAMES[labels[i]]}", fontsize=6)
        axes[1, i].imshow(images[i].reshape(28, 28), cmap="gray")
        axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, i].set_title(f"prev.: {CLASS_NAMES[pred]}", fontsize=6)
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.suptitle("Grad-CAM: onde o modelo olha para decidir")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def most_confident_errors(model, X, y, k: int = 16):
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()
    pred = probs.argmax(1)
    wrong = np.where(pred != y)[0]
    order = wrong[np.argsort(-probs[wrong, pred[wrong]])][:k]
    titles = [f"{CLASS_NAMES[y[i]]} -> {CLASS_NAMES[pred[i]]}\n{probs[i, pred[i]]:.0%}"
              for i in order]
    return X[order], titles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/interpret")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(args.checkpoint)
    ds = load_fashion_mnist()
    X_err, titles = most_confident_errors(model, ds.X_val, ds.y_val)
    plot_image_grid(X_err, titles, out / "erros_confiantes.png")
    if isinstance(model, CNN):
        plot_first_layer_filters(model, out / "filtros.png")
        plot_feature_maps(model, ds.X_val[0], out / "mapas_ativacao.png")
        idx = [int(np.where(ds.y_val == c)[0][0]) for c in range(8)]
        plot_grad_cam(model, ds.X_val[idx], ds.y_val[idx], out / "grad_cam.png")
    print(f"Figuras salvas em {out}/")
