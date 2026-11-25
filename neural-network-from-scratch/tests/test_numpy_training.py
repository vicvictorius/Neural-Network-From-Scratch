"""Testes de integração da Fase 1: otimizadores, Sequential e o teste de overfit."""

import numpy as np
import pytest

from src.numpy_nn import SGD, Adam, Momentum, SoftmaxCrossEntropy
from src.numpy_nn.model import build_mlp
from src.numpy_nn.optimizers import build_optimizer


def test_sgd_um_passo():
    w = np.array([1.0, 2.0])
    g = np.array([0.5, -1.0])
    SGD([w], [g], lr=0.1).step()
    assert np.allclose(w, [0.95, 2.1])


@pytest.mark.parametrize("opt_name", ["sgd", "momentum", "adam"])
def test_otimizadores_minimizam_quadratica(opt_name):
    # f(w) = soma(w^2), mínimo em w = 0
    w = np.array([3.0, -2.0])
    g = np.zeros_like(w)
    lr = 0.05 if opt_name == "adam" else 0.1
    opt = build_optimizer(opt_name, [w], [g], lr=lr, momentum=0.9)
    for _ in range(300):
        g[...] = 2 * w
        opt.step()
    assert np.allclose(w, 0, atol=1e-2)


def test_otimizadores_atualizam_in_place():
    for cls in (SGD, Momentum, Adam):
        w = np.ones(3)
        ref = w
        cls([w], [np.ones(3)], lr=0.1).step()
        assert ref is w and not np.allclose(w, 1.0)


def test_sequential_formatos(rng):
    model = build_mlp(20, [16, 8], 3, rng=rng)
    assert model.forward(rng.normal(size=(5, 20))).shape == (5, 3)
    assert model.num_params() == 20 * 16 + 16 + 16 * 8 + 8 + 8 * 3 + 3


def test_sequential_salvar_e_carregar(rng, tmp_path):
    model = build_mlp(10, [8], 3, rng=rng)
    x = rng.normal(size=(4, 10))
    expected = model.forward(x)
    model.save(tmp_path / "w.npz")
    other = build_mlp(10, [8], 3, rng=np.random.default_rng(99))
    other.load(tmp_path / "w.npz")
    assert np.allclose(other.forward(x), expected)


def test_overfit_em_poucos_exemplos(rng):
    """Teste de sanidade mais importante: uma rede correta consegue DECORAR
    um conjunto pequeno. Se este teste falhar, há bug no forward/backward."""
    X = rng.normal(size=(64, 30))
    y = rng.integers(0, 4, size=64)
    model = build_mlp(30, [64], 4, rng=rng)
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(model.params(), model.grads(), lr=0.01)
    for _ in range(300):
        loss = loss_fn.forward(model.forward(X), y)
        model.backward(loss_fn.backward())
        opt.step()
    assert loss < 0.05
    assert np.mean(model.predict(X) == y) == 1.0
