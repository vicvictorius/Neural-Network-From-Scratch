"""Sequential: empilha camadas e coordena forward/backward.

Testes: pytest tests/test_numpy_training.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .activations import ACTIVATIONS
from .layers import Dense, Dropout, Layer


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        # A saída de cada camada é a entrada da próxima: composição de funções,
        # f3(f2(f1(x))). Cada camada guarda internamente o que o backward precisa.
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Regra da cadeia aplicada de trás para frente: cada camada recebe
        # dL/d(sua saída), calcula os gradientes dos próprios parâmetros e
        # devolve dL/d(sua entrada), que é exatamente o dL/d(saída) da camada
        # anterior. Por isso a ordem é invertida. Isto é o backpropagation.
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[np.ndarray]:
        # Lista "achatada" com os arrays de pesos de todas as camadas, na ordem
        # das camadas. Camadas sem parâmetros (ativações) contribuem com [].
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        # MESMA ordem de params(): o otimizador casa params()[i] com grads()[i].
        # São referências aos arrays dW e db das camadas, não cópias, por isso
        # o otimizador sempre enxerga os gradientes do último backward.
        return [g for layer in self.layers for g in layer.grads()]

    def train(self) -> None:
        for layer in self.layers:
            layer.training = True

    def eval(self) -> None:
        for layer in self.layers:
            layer.training = False

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        preds = [self.forward(X[i : i + batch_size]).argmax(axis=1)
                 for i in range(0, len(X), batch_size)]
        return np.concatenate(preds)

    def num_params(self) -> int:
        return sum(p.size for p in self.params())

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, *self.params())

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        for i, p in enumerate(self.params()):
            p[...] = data[f"arr_{i}"]


def build_mlp(
    n_in: int,
    hidden: list[int],
    n_out: int,
    activation: str = "relu",
    dropout: float = 0.0,
    init: str = "he",
    rng: np.random.Generator | None = None,
) -> Sequential:
    """Ex.: build_mlp(784, [128, 64], 10) -> 784 -> 128 -> 64 -> 10"""
    rng = rng or np.random.default_rng()
    layers: list[Layer] = []
    sizes = [n_in, *hidden]
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [Dense(a, b, init=init, rng=rng), ACTIVATIONS[activation]()]
        if dropout > 0:
            layers.append(Dropout(dropout, rng=rng))
    layers.append(Dense(sizes[-1], n_out, init=init, rng=rng))
    return Sequential(layers)
