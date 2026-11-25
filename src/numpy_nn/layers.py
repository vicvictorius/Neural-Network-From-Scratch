"""Camadas com forward e backward escritos à mão.

Testes: pytest tests/test_numpy_layers.py -v

Convenção de formatos: x tem formato (batch, features).
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Interface comum de todas as camadas.

    - forward(x): calcula a saída e guarda o que o backward vai precisar.
    - backward(grad_out): recebe dL/d(saída), guarda dL/d(parâmetros) e
      devolve dL/d(entrada) para a camada anterior.
    - O backward NÃO atualiza pesos: isso é trabalho do otimizador.
    """

    training: bool = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class Dense(Layer):
    """y = x @ W + b

    x: (batch, n_in)   W: (n_in, n_out)   b: (n_out,)   y: (batch, n_out)
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        init: str = "he",
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        # A escala da inicialização controla a variância do sinal ao atravessar
        # as camadas. Cada saída soma n_in termos x*w; se Var(w) = s², a
        # variância da saída fica perto de n_in * s² * Var(x). Para ela não
        # explodir nem sumir camada após camada, queremos n_in * s² ≈ 1.
        if init == "he":
            # A ReLU zera ~metade das entradas, o que corta a variância pela
            # metade. O fator 2 compensa essa perda.
            scale = np.sqrt(2.0 / n_in)
        elif init == "xavier":
            # tanh e sigmoid não cortam metade do sinal: basta n_in * s² = 1.
            scale = np.sqrt(1.0 / n_in)
        else:
            raise ValueError(f"init desconhecido: {init!r} (use 'he' ou 'xavier')")
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        # Os gradientes têm o mesmo formato dos parâmetros.
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # O backward vai precisar da entrada para calcular dL/dW = x^T @ grad.
        # Guardar o que o backward precisa é o padrão de TODA camada.
        self._x = x
        # x @ W: (batch, n_in) @ (n_in, n_out) -> (batch, n_out)
        # + b: broadcasting soma o mesmo vetor (n_out,) em todas as linhas.
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Regra prática: o gradiente de algo tem o MESMO formato desse algo.
        # Ela praticamente dita a ordem das multiplicações:
        #
        #   dL/dW precisa ser (n_in, n_out). Temos x (batch, n_in) e
        #   grad_out (batch, n_out). Única combinação: x^T @ grad_out.
        #   A multiplicação também soma as contribuições de todos os
        #   exemplos do batch, porque W é compartilhado por todos eles.
        #
        #   dL/db precisa ser (n_out,). b foi somado em todas as linhas,
        #   então recebe a soma dos gradientes de todas as linhas.
        #
        #   dL/dx precisa ser (batch, n_in): grad_out @ W^T.
        #
        # Escrevemos com [...] (in-place) porque o otimizador guarda
        # referências a self.dW e self.db. `self.dW = ...` criaria um array
        # novo e o otimizador continuaria olhando para o antigo, zerado.
        self.dW[...] = self._x.T @ grad_out
        self.db[...] = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class Dropout(Layer):
    """Inverted dropout: em treino zera neurônios com probabilidade p e
    multiplica o resto por 1/(1-p); em avaliação não faz nada."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("p deve estar em [0, 1)")
        self.p = p
        self.rng = rng or np.random.default_rng()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Em avaliação, a rede inteira deve participar: sem sorteio.
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        # Cada neurônio sobrevive com probabilidade (1 - p). Dividir os
        # sobreviventes por (1 - p) mantém o VALOR ESPERADO da ativação igual
        # ao de avaliação: E[saída] = (1 - p) * x / (1 - p) = x. Sem isso, a
        # camada seguinte veria sinais de magnitudes diferentes em treino e em
        # avaliação. Esta é a versão "invertida" usada pelo PyTorch.
        self._mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # A saída é x * máscara, então dL/dx = dL/dsaída * máscara: o gradiente
        # passa exatamente pelos neurônios que sobreviveram no forward, com o
        # mesmo fator 1/(1 - p).
        if self._mask is None:
            return grad_out
        return grad_out * self._mask
