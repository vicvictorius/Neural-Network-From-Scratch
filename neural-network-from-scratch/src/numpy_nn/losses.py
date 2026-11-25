"""Softmax + Cross-Entropy combinadas (estável e com gradiente simples)."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """logits: (N, C) -> probabilidades (N, C), cada linha soma 1.

    Subtrair o máximo de cada linha não muda o resultado:
        e^(z - m) / soma(e^(z - m)) = e^-m * e^z / (e^-m * soma(e^z))
    O fator e^-m cancela. Mas agora o maior expoente é 0, então exp() nunca
    passa de 1 e não há overflow, mesmo com logits na casa de 1000.
    """
    # keepdims=True mantém o formato (N, 1), para o broadcasting subtrair o
    # máximo de cada linha da linha inteira. Sem ele, o formato seria (N,) e
    # o NumPy alinharia pelo eixo errado.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """L = -(1/N) * soma log(p da classe correta)

    O gradiente em relação aos logits é (p - y_onehot) / N.
    Dedução completa em docs/derivacoes.md, seção 3.

    Por que juntar softmax e cross-entropy numa classe só? Separadas, o
    backward do softmax exigiria uma matriz Jacobiana C x C por exemplo.
    Juntas, quase tudo se cancela e o gradiente vira uma subtração.
    """

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """logits: (N, C)   y: (N,) com inteiros de 0 a C-1"""
        self._probs = softmax(logits)
        self._y = y
        n = logits.shape[0]

        # Indexação avançada: para cada linha i, pega a coluna y[i].
        # Equivale a [probs[0, y[0]], probs[1, y[1]], ...] sem laço em Python.
        correct = self._probs[np.arange(n), y]

        # O 1e-12 evita log(0) = -infinito se a rede der probabilidade zero à
        # classe certa (possível com logits extremos por arredondamento).
        # Perda alta quando p_correta é baixa: log(0.01) ≈ -4.6, log(0.99) ≈ 0.
        return float(-np.mean(np.log(correct + 1e-12)))

    def backward(self) -> np.ndarray:
        # dL/dz = (p - y_onehot) / N, sem construir a matriz one-hot:
        # copiamos p e subtraímos 1 só na coluna da classe correta.
        # Intuição: o gradiente empurra para cima o logit da classe certa
        # (p - 1 < 0) e para baixo todos os outros (p > 0), na proporção do
        # quanto a rede está errada.
        n = self._probs.shape[0]
        grad = self._probs.copy()  # .copy(): não queremos alterar self._probs
        grad[np.arange(n), self._y] -= 1.0
        # Dividir por N porque a perda é uma MÉDIA no batch. Sem isso, o
        # tamanho do passo do otimizador dependeria do tamanho do batch.
        return grad / n
