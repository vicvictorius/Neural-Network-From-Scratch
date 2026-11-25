"""Testes da Fase 1. Eles funcionam como especificação: implemente cada
peça até o respectivo teste passar. Rodar: pytest tests/test_numpy_layers.py -v
"""

import numpy as np
import pytest

from src.numpy_nn import Dense, Dropout, ReLU, Sigmoid, SoftmaxCrossEntropy, Tanh
from src.numpy_nn.gradcheck import numerical_gradient, relative_error
from src.numpy_nn.losses import softmax

TOL = 1e-6


def check_layer_input_grad(layer, x, rng):
    """Compara dL/dx do backward com a aproximação numérica.
    L = soma(saída * R), com R aleatório, logo dL/d(saída) = R."""
    out = layer.forward(x)
    R = rng.normal(size=out.shape)
    dx = layer.backward(R)
    num = numerical_gradient(lambda: float(np.sum(layer.forward(x) * R)), x)
    return relative_error(dx, num)


# ---------- Dense ----------

def test_dense_formato_da_saida(rng):
    layer = Dense(5, 3, rng=rng)
    assert layer.forward(rng.normal(size=(4, 5))).shape == (4, 3)


def test_dense_inicializacao_he(rng):
    layer = Dense(1000, 500, init="he", rng=rng)
    assert abs(layer.W.std() - np.sqrt(2 / 1000)) < 0.005
    assert np.all(layer.b == 0)


def test_dense_gradiente_da_entrada(rng):
    layer = Dense(5, 3, rng=rng)
    assert check_layer_input_grad(layer, rng.normal(size=(4, 5)), rng) < TOL


@pytest.mark.parametrize("which", ["W", "b"])
def test_dense_gradiente_dos_parametros(rng, which):
    layer = Dense(5, 3, rng=rng)
    x = rng.normal(size=(4, 5))
    R = rng.normal(size=(4, 3))
    layer.forward(x)
    layer.backward(R)
    analytic = layer.dW if which == "W" else layer.db
    param = layer.W if which == "W" else layer.b
    num = numerical_gradient(lambda: float(np.sum(layer.forward(x) * R)), param)
    assert relative_error(analytic, num) < TOL


# ---------- Ativações ----------

@pytest.mark.parametrize("act_cls", [ReLU, Sigmoid, Tanh])
def test_ativacoes_gradiente(rng, act_cls):
    x = rng.normal(size=(4, 6))
    x[np.abs(x) < 1e-3] = 0.5  # evita o ponto não diferenciável da ReLU
    assert check_layer_input_grad(act_cls(), x, rng) < TOL


def test_sigmoid_estavel_para_valores_extremos():
    out = Sigmoid().forward(np.array([[-1000.0, 0.0, 1000.0]]))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, [[0.0, 0.5, 1.0]])


# ---------- Softmax + Cross-Entropy ----------

def test_softmax_soma_um_e_estavel():
    p = softmax(np.array([[1000.0, 1000.0, 1000.0], [1.0, 2.0, 3.0]]))
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[0], 1 / 3)


def test_cross_entropy_valor_conhecido():
    # Logits iguais para 4 classes -> p = 0.25 -> perda = log(4)
    loss = SoftmaxCrossEntropy().forward(np.zeros((2, 4)), np.array([0, 3]))
    assert np.isclose(loss, np.log(4))


def test_cross_entropy_gradiente(rng):
    loss_fn = SoftmaxCrossEntropy()
    logits = rng.normal(size=(5, 4))
    y = rng.integers(0, 4, size=5)
    loss_fn.forward(logits, y)
    analytic = loss_fn.backward()
    num = numerical_gradient(lambda: loss_fn.forward(logits, y), logits)
    assert relative_error(analytic, num) < TOL


# ---------- Dropout ----------

def test_dropout_em_avaliacao_nao_faz_nada(rng):
    layer = Dropout(0.5, rng=rng)
    layer.training = False
    x = rng.normal(size=(10, 10))
    assert np.array_equal(layer.forward(x), x)


def test_dropout_preserva_media_em_treino(rng):
    layer = Dropout(0.5, rng=rng)
    out = layer.forward(np.ones((1000, 100)))
    assert abs(out.mean() - 1.0) < 0.02
    assert 0.45 < np.mean(out == 0) < 0.55


def test_dropout_backward_usa_a_mesma_mascara(rng):
    layer = Dropout(0.5, rng=rng)
    out = layer.forward(np.ones((20, 20)))
    grad = layer.backward(np.ones((20, 20)))
    assert np.array_equal(out == 0, grad == 0)
