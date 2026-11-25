"""Métricas e registro de treino em CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Linha = classe verdadeira, coluna = classe prevista."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


class MetricsLogger:
    """Grava uma linha por época em um CSV (fácil de abrir no pandas ou Excel)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, float]] = []

    def log(self, **values: float) -> None:
        self.rows.append(values)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

    def column(self, name: str) -> list[float]:
        return [row[name] for row in self.rows]
