"""Gradient check: compara o gradiente analítico (backward) com uma
aproximação numérica por diferenças finitas centrais.

    df/dx ≈ (f(x + eps) - f(x - eps)) / (2 * eps)

Se o erro relativo for < 1e-6 (em float64), o backward está correto.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Estima df/dx mexendo em um elemento de x por vez.

    f é uma função SEM argumentos: ela lê x "por referência". Como alteramos x
    in-place, cada chamada de f() enxerga o valor perturbado. É por isso que os
    testes passam algo como `lambda: float(np.sum(layer.forward(x) * R))`.

    Custo: 2 chamadas de f por elemento de x. Para uma matriz 784x128 seriam
    ~200 mil forwards, por isso o gradiente numérico só serve para VERIFICAR o
    backpropagation (que calcula tudo em um único backward), nunca para treinar.
    """
    grad = np.zeros_like(x, dtype=np.float64)

    # np.ndindex gera todas as tuplas de índice: (0, 0), (0, 1), ... para
    # qualquer número de dimensões. Funciona igual para vetores e matrizes.
    for idx in np.ndindex(x.shape):
        original = x[idx]

        x[idx] = original + eps
        f_plus = f()

        x[idx] = original - eps
        f_minus = f()

        # Restaurar é essencial: sem isso, cada iteração herda as perturbações
        # anteriores e o gradiente sai errado de um jeito difícil de depurar.
        x[idx] = original

        # Diferença CENTRAL: o erro é proporcional a eps², contra eps da
        # diferença simples (f(x+eps) - f(x)) / eps. Com eps = 1e-5, isso é a
        # diferença entre um erro de ~1e-10 e um de ~1e-5.
        grad[idx] = (f_plus - f_minus) / (2 * eps)

    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))
