"""Fase 2: prova que a sua rede NumPy calcula EXATAMENTE o mesmo que o PyTorch.

Copia os mesmos pesos para as duas redes e compara a perda e todos os gradientes.
Este é um dos testes mais fortes do portfólio.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.numpy_nn import SoftmaxCrossEntropy  # noqa: E402
from src.numpy_nn.model import build_mlp  # noqa: E402
from src.torch_nn.models import MLP  # noqa: E402

pytestmark = pytest.mark.torch


def test_numpy_e_pytorch_dao_o_mesmo_resultado():
    rng = np.random.default_rng(0)
    np_model = build_mlp(784, [32, 16], 10, rng=rng)
    th_model = MLP([32, 16]).double()

    # PyTorch guarda Linear.weight como (out, in); a nossa Dense usa (in, out).
    np_dense = [layer for layer in np_model.layers if hasattr(layer, "W")]
    th_linear = [m for m in th_model.modules() if isinstance(m, torch.nn.Linear)]
    for d, lin in zip(np_dense, th_linear, strict=True):
        lin.weight.data = torch.from_numpy(d.W.T.copy())
        lin.bias.data = torch.from_numpy(d.b.copy())

    x = rng.normal(size=(8, 784))
    y = rng.integers(0, 10, size=8)

    loss_fn = SoftmaxCrossEntropy()
    np_loss = loss_fn.forward(np_model.forward(x), y)
    np_model.backward(loss_fn.backward())

    th_loss = torch.nn.functional.cross_entropy(th_model(torch.from_numpy(x)), torch.from_numpy(y))
    th_loss.backward()

    assert np.isclose(np_loss, th_loss.item(), atol=1e-8)
    for d, lin in zip(np_dense, th_linear, strict=True):
        assert np.allclose(d.dW, lin.weight.grad.numpy().T, atol=1e-8)
        assert np.allclose(d.db, lin.bias.grad.numpy(), atol=1e-8)
