"""Funções de ativação como camadas sem parâmetros.

Sem ativações não lineares, empilhar camadas Dense não adiantaria nada:
W2 @ (W1 @ x) = (W2 @ W1) @ x, que é uma única camada linear. A não
linearidade é o que permite à rede aprender fronteiras de decisão curvas.

Todas seguem o mesmo padrão de backward (regra da cadeia elemento a elemento):
    dL/dx = dL/dy * f'(x)
onde * é o produto ELEMENTO A ELEMENTO, não o produto matricial.
"""

from __future__ import annotations

import numpy as np

from .layers import Layer


class ReLU(Layer):
    """f(x) = max(0, x)   f'(x) = 1 se x > 0, senão 0"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # A máscara booleana guarda onde a ReLU "deixou passar". Ela é ao mesmo
        # tempo a forma de calcular a saída e a própria derivada.
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Onde x <= 0 a saída era constante (zero), então o gradiente morre ali.
        # Se um neurônio fica com x <= 0 para TODOS os exemplos, ele nunca mais
        # recebe gradiente: é a "ReLU morta". A inicialização He ajuda a evitar.
        return grad_out * self._mask


class Sigmoid(Layer):
    """f(x) = 1 / (1 + e^-x)   f'(x) = f(x) * (1 - f(x))"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # A fórmula direta 1 / (1 + exp(-x)) estoura para x muito negativo:
        # exp(1000) é infinito em float64. As duas formas abaixo são
        # matematicamente iguais (multiplique a primeira por e^x / e^x), mas
        # cada uma só chama exp() com argumento <= 0, que fica sempre em (0, 1].
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        # Guardamos a SAÍDA, não a entrada: a derivada se escreve em função dela.
        self._out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # O valor máximo de f(1 - f) é 0,25 (em x = 0). Em uma rede profunda de
        # sigmoids, o gradiente é multiplicado por no máximo 0,25 a cada camada
        # e some rapidamente: o "vanishing gradient", um dos motivos de a ReLU
        # ter se tornado o padrão.
        return grad_out * self._out * (1.0 - self._out)


class Tanh(Layer):
    """f(x) = tanh(x)   f'(x) = 1 - tanh(x)^2"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # np.tanh já é numericamente estável. Saída em (-1, 1), centrada em
        # zero, o que costuma treinar melhor que a sigmoid em camadas ocultas.
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1.0 - self._out**2)


ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
