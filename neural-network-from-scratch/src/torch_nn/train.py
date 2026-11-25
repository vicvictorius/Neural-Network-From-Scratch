"""Treino de MLP ou CNN em PyTorch, guiado por um YAML.

Uso:
    python -m src.torch_nn.train --config configs/torch_mlp.yaml
    python -m src.torch_nn.train --config configs/cnn_base.yaml --seed 1
    python -m src.torch_nn.train --config configs/cnn_base.yaml --evaluate-test  # só no final
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import CLASS_NAMES, load_fashion_mnist
from src.utils.config import load_config, save_config
from src.utils.metrics import MetricsLogger, confusion_matrix
from src.utils.plots import plot_confusion_matrix, plot_training_curves
from src.utils.seed import set_seed

from .augment import augment
from .models import build_model, count_params


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Macs com Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    ds = TensorDataset(torch.from_numpy(X).reshape(-1, 1, 28, 28), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    if cfg["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 0.0))
    if cfg["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                               momentum=cfg.get("momentum", 0.0),
                               weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"otimizador desconhecido: {cfg['name']}")


def build_scheduler(cfg: dict | None, optimizer, epochs: int):
    name = (cfg or {}).get("name", "none")
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 10),
                                               gamma=cfg.get("gamma", 0.1))
    raise ValueError(f"scheduler desconhecido: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Devolve (perda média, acurácia, previsões)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds = 0.0, []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item()
        preds.append(logits.argmax(1).cpu())
    preds_t = torch.cat(preds).numpy()
    y_all = loader.dataset.tensors[1].numpy()
    return total_loss / len(y_all), float((preds_t == y_all).mean()), preds_t


def train(config: dict, evaluate_test: bool = False, verbose: bool = True) -> dict:
    set_seed(config["seed"])
    device = get_device()
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")
    t = config["training"]

    ds = load_fashion_mnist()
    train_loader = make_loader(ds.X_train, ds.y_train, t["batch_size"], True, config["seed"])
    val_loader = make_loader(ds.X_val, ds.y_val, 1024, False, config["seed"])

    model = build_model(config["model"]).to(device)
    optimizer = build_optimizer(config["optimizer"], model)
    scheduler = build_scheduler(config.get("scheduler"), optimizer, t["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    if verbose:
        print(f"Dispositivo: {device} | parâmetros: {count_params(model):,}")

    logger = MetricsLogger(out_dir / "metrics.csv")
    ckpt = Path("checkpoints") / f"{out_dir.name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    best_val, best_epoch, wait = -1.0, 0, 0

    for epoch in range(1, t["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if t.get("augmentation"):
                xb = augment(xb)
            optimizer.zero_grad()           # 1. zera gradientes do passo anterior
            logits = model(xb)              # 2. forward
            loss = loss_fn(logits, yb)      # 3. perda
            loss.backward()                 # 4. backward (autograd)
            optimizer.step()                # 5. atualiza pesos
            run_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(xb)
        if scheduler:
            scheduler.step()

        val_loss, val_acc, _ = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - start
        logger.log(epoch=epoch, train_loss=run_loss / seen, train_acc=correct / seen,
                   val_loss=val_loss, val_acc=val_acc, lr=optimizer.param_groups[0]["lr"],
                   seconds=elapsed)
        if verbose:
            print(f"época {epoch:3d} | perda {run_loss / seen:.4f} | acc treino "
                  f"{100 * correct / seen:.2f}% | acc val {100 * val_acc:.2f}% | {elapsed:.1f}s")

        if val_acc > best_val:
            best_val, best_epoch, wait = val_acc, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": config}, ckpt)
        else:
            wait += 1
            if wait >= t["patience"]:
                if verbose:
                    print(f"Early stopping: sem melhora há {wait} épocas.")
                break

    history = {k: logger.column(k) for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    plot_training_curves(history, out_dir / "curves.png", title=out_dir.name)
    result = {"best_val_acc": best_val, "best_epoch": best_epoch,
              "seconds_per_epoch": float(np.mean(logger.column("seconds"))),
              "checkpoint": str(ckpt)}

    if evaluate_test:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
        test_loader = make_loader(ds.X_test, ds.y_test, 1024, False, config["seed"])
        _, test_acc, preds = evaluate(model, test_loader, device)
        result["test_acc"] = test_acc
        plot_confusion_matrix(confusion_matrix(ds.y_test, preds), CLASS_NAMES,
                              out_dir / "confusion_matrix.png")
        print(f"Acurácia de TESTE: {100 * test_acc:.2f}%")

    if verbose:
        print(f"Melhor validação: {100 * best_val:.2f}% (época {best_epoch})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="sobrescreve a seed do YAML")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="use apenas no modelo final da fase")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["output_dir"] = f"{cfg['output_dir']}_seed{args.seed}"
    train(cfg, evaluate_test=args.evaluate_test)
