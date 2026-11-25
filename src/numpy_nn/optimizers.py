"""Otimizadores. Atualizam os parâmetros IN-PLACE (p -= ..., p += ...), para
que as camadas vejam os novos valores sem precisar reatribuir nada.

Testes: pytest tests/test_numpy_training.py -k otimizadores -v
"""

from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params: list[np.ndarray], grads: list[np.ndarray], lr: float,
                 weight_decay: float = 0.0) -> None:
        if len(params) != len(grads):
            raise ValueError("params e grads precisam ter o mesmo tamanho")
        self.params = params
        self.grads = grads
        self.lr = lr
        self.weight_decay = weight_decay

    def _grad(self, i: int) -> np.ndarray:
        """Gradiente do parâmetro i, já com regularização L2 (pronto)."""
        g = self.grads[i]
        if self.weight_decay and self.params[i].ndim > 1:  # não regulariza bias
            g = g + self.weight_decay * self.params[i]
        return g

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """w <- w - lr * g"""

    def step(self) -> None:
        # Um passo na direção oposta ao gradiente (a direção de maior descida).
        # `p -= ...` modifica o array original, que é o mesmo objeto guardado
        # na camada. `p = p - ...` só mudaria a variável local p.
        for i, p in enumerate(self.params):
            p -= self.lr * self._grad(i)


class Momentum(Optimizer):
    """v <- mu * v - lr * g ;  w <- w + v"""

    def __init__(self, params, grads, lr, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params, grads, lr, weight_decay)
        self.momentum = momentum
        self.v = [np.zeros_like(p) for p in params]

    def step(self) -> None:
        # v acumula uma média dos gradientes recentes, como uma bola rolando
        # morro abaixo que ganha velocidade em direções consistentes e amortece
        # oscilações em direções que trocam de sinal a cada passo.
        for i, p in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] - self.lr * self._grad(i)
            p += self.v[i]


class Adam(Optimizer):
    """m <- b1*m + (1-b1)*g          v <- b2*v + (1-b2)*g^2
    m_hat = m / (1 - b1^t)          v_hat = v / (1 - b2^t)
    w <- w - lr * m_hat / (raiz(v_hat) + eps)
    """

    def __init__(self, params, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, grads, lr, weight_decay)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self) -> None:
        # t conta os passos (uma vez por step, não por parâmetro).
        self.t += 1
        for i, p in enumerate(self.params):
            g = self._grad(i)
            # m: média móvel do gradiente (direção, como no momentum).
            # v: média móvel do gradiente ao quadrado (escala típica de cada peso).
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            # m e v começam em zero, então nos primeiros passos ficam
            # subestimados (no passo 1, m = 0,1 * g). Dividir por (1 - beta^t)
            # corrige isso; o efeito some quando t cresce.
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            # Dividir por raiz(v_hat) dá a cada peso seu próprio tamanho de
            # passo: pesos com gradientes grandes andam com mais cautela.
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def build_optimizer(name: str, params, grads, **kwargs) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        kwargs.pop("momentum", None)
        return SGD(params, grads, **kwargs)
    if name == "momentum":
        return Momentum(params, grads, **kwargs)
    if name == "adam":
        kwargs.pop("momentum", None)
        return Adam(params, grads, **kwargs)
    raise ValueError(f"otimizador desconhecido: {name}")
