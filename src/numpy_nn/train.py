"""Treino da MLP em NumPy.

Uso:
    python -m src.numpy_nn.train --config configs/numpy_mlp.yaml
    python -m src.numpy_nn.train --config configs/numpy_mlp.yaml --evaluate-test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.data import CLASS_NAMES, iterate_minibatches, load_fashion_mnist
from src.utils.config import load_config, save_config
from src.utils.metrics import MetricsLogger, accuracy, confusion_matrix
from src.utils.plots import plot_confusion_matrix, plot_training_curves
from src.utils.seed import set_seed

from .losses import SoftmaxCrossEntropy
from .model import Sequential, build_mlp
from .optimizers import build_optimizer


def evaluate(model: Sequential, loss_fn: SoftmaxCrossEntropy, X, y, batch_size=1024):
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model.forward(X[i : i + batch_size])
        total_loss += loss_fn.forward(logits, y[i : i + batch_size]) * len(logits)
        preds.append(logits.argmax(axis=1))
    return total_loss / len(X), accuracy(y, np.concatenate(preds))


def train(config: dict, evaluate_test: bool = False) -> dict:
    rng = set_seed(config["seed"])
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    ds = load_fashion_mnist()
    m, o, t = config["model"], config["optimizer"], config["training"]
    model = build_mlp(784, m["hidden"], 10, m["activation"], m["dropout"], m["init"], rng)
    loss_fn = SoftmaxCrossEntropy()
    opt = build_optimizer(o["name"], model.params(), model.grads(), lr=o["lr"],
                          weight_decay=o.get("weight_decay", 0.0),
                          momentum=o.get("momentum", 0.9))
    print(f"Modelo com {model.num_params():,} parâmetros")

    logger = MetricsLogger(out_dir / "metrics.csv")
    best_val, best_epoch, wait = -1.0, 0, 0
    ckpt = Path("checkpoints") / f"{out_dir.name}.npz"

    for epoch in range(1, t["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        epoch_loss, correct, seen = 0.0, 0, 0
        for xb, yb in iterate_minibatches(ds.X_train, ds.y_train, t["batch_size"], rng=rng):
            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            opt.step()
            epoch_loss += loss * len(xb)
            correct += int((logits.argmax(axis=1) == yb).sum())
            seen += len(xb)

        val_loss, val_acc = evaluate(model, loss_fn, ds.X_val, ds.y_val)
        elapsed = time.perf_counter() - start
        logger.log(epoch=epoch, train_loss=epoch_loss / seen, train_acc=correct / seen,
                   val_loss=val_loss, val_acc=val_acc, seconds=elapsed)
        print(f"época {epoch:3d} | perda treino {epoch_loss / seen:.4f} | "
              f"acc treino {100 * correct / seen:.2f}% | acc val {100 * val_acc:.2f}% | "
              f"{elapsed:.1f}s")

        if val_acc > best_val:
            best_val, best_epoch, wait = val_acc, epoch, 0
            model.save(ckpt)
        else:
            wait += 1
            if wait >= t["patience"]:
                print(f"Early stopping: sem melhora há {wait} épocas.")
                break

    history = {k: logger.column(k) for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    plot_training_curves(history, out_dir / "curves.png", title="MLP NumPy")
    result = {"best_val_acc": best_val, "best_epoch": best_epoch}

    if evaluate_test:
        model.load(ckpt)
        y_pred = model.predict(ds.X_test)
        result["test_acc"] = accuracy(ds.y_test, y_pred)
        plot_confusion_matrix(confusion_matrix(ds.y_test, y_pred), CLASS_NAMES,
                              out_dir / "confusion_matrix.png")
        print(f"Acurácia de TESTE: {100 * result['test_acc']:.2f}%")

    print(f"Melhor validação: {100 * best_val:.2f}% (época {best_epoch})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/numpy_mlp.yaml")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="use apenas no modelo final da fase")
    args = parser.parse_args()
    train(load_config(args.config), evaluate_test=args.evaluate_test)
