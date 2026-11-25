"""Download, leitura e preparação do Fashion-MNIST.

Uso:
    python -m src.data          # baixa (se preciso) e imprime os formatos

Os arquivos originais ficam em data/fashion-mnist/ (fora do Git).
"""

from __future__ import annotations

import gzip
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = [
    "Camiseta/top",
    "Calça",
    "Pulôver",
    "Vestido",
    "Casaco",
    "Sandália",
    "Camisa",
    "Tênis",
    "Bolsa",
    "Bota",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "fashion-mnist"


@dataclass
class Dataset:
    """Os três conjuntos já normalizados. Imagens achatadas: (n, 784), float32 em [0, 1]."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def download(data_dir: Path = DATA_DIR) -> None:
    """Baixa os 4 arquivos .gz se ainda não existirem."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        target = data_dir / filename
        if target.exists():
            continue
        print(f"Baixando {filename}...")
        urllib.request.urlretrieve(BASE_URL + filename, target)


def _read_images(path: Path) -> np.ndarray:
    # Formato IDX: 16 bytes de cabeçalho, depois 1 byte (0-255) por pixel.
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28)


def _read_labels(path: Path) -> np.ndarray:
    # Formato IDX: 8 bytes de cabeçalho, depois 1 byte por rótulo.
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)


def load_fashion_mnist(
    data_dir: Path = DATA_DIR, val_size: int = 10_000, seed: int = 42
) -> Dataset:
    """Carrega o Fashion-MNIST e separa treino/validação de forma reproduzível.

    A validação sai do conjunto de treino original. O teste (10k) só deve ser
    usado UMA vez, no modelo final de cada fase.
    """
    download(data_dir)
    X_full = _read_images(data_dir / FILES["train_images"]).astype(np.float32) / 255.0
    y_full = _read_labels(data_dir / FILES["train_labels"])
    X_test = _read_images(data_dir / FILES["test_images"]).astype(np.float32) / 255.0
    y_test = _read_labels(data_dir / FILES["test_labels"])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_full))
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    return Dataset(
        X_train=X_full[train_idx],
        y_train=y_full[train_idx],
        X_val=X_full[val_idx],
        y_val=y_full[val_idx],
        X_test=X_test,
        y_test=y_test,
    )


def iterate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Gera (X_batch, y_batch). O último batch pode ser menor que batch_size."""
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        (rng or np.random.default_rng()).shuffle(idx)
    for start in range(0, n, batch_size):
        batch = idx[start : start + batch_size]
        yield X[batch], y[batch]


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """(n,) inteiros -> (n, num_classes) com 1.0 na classe correta."""
    out = np.zeros((len(y), num_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


if __name__ == "__main__":
    ds = load_fashion_mnist()
    print("treino   :", ds.X_train.shape, ds.y_train.shape)
    print("validação:", ds.X_val.shape, ds.y_val.shape)
    print("teste    :", ds.X_test.shape, ds.y_test.shape)
    print(f"pixels em [{ds.X_train.min():.1f}, {ds.X_train.max():.1f}]")
    print("exemplos por classe (treino):", np.bincount(ds.y_train))
