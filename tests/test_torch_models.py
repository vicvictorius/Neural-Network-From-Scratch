"""Testes da Fase 2/3 (PyTorch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.torch_nn.augment import augment, random_horizontal_flip, random_shift  # noqa: E402
from src.torch_nn.models import CNN, MLP, build_model, count_params  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.mark.parametrize("shape", [(4, 784), (4, 1, 28, 28)])
def test_mlp_aceita_os_dois_formatos(shape):
    assert MLP()(torch.randn(*shape)).shape == (4, 10)


def test_mlp_numero_de_parametros_igual_ao_numpy():
    assert count_params(MLP([128, 64])) == 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10


@pytest.mark.parametrize("bn", [False, True])
def test_cnn_formato_da_saida(bn):
    model = CNN([32, 64], 128, batchnorm=bn, dropout=0.25)
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 10)


def test_cnn_formato_intermediario():
    feats = CNN([32, 64]).features(torch.randn(2, 1, 28, 28))
    assert feats.shape == (2, 64, 7, 7)


def test_build_model_le_config():
    assert isinstance(build_model({"type": "cnn", "channels": [16, 32]}), CNN)
    with pytest.raises(ValueError):
        build_model({"type": "transformer"})


def test_augmentation_preserva_formato_e_valores():
    x = torch.rand(8, 1, 28, 28)
    out = augment(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1


def test_flip_com_p_1_espelha_tudo():
    x = torch.rand(2, 1, 28, 28)
    assert torch.equal(random_horizontal_flip(x, p=1.0), x.flip(-1))


def test_shift_zero_nao_muda_nada():
    x = torch.rand(2, 1, 28, 28)
    assert torch.equal(random_shift(x, max_shift=0), x)


def test_cnn_decora_um_batch_pequeno():
    torch.manual_seed(0)
    x, y = torch.rand(32, 1, 28, 28), torch.randint(0, 10, (32,))
    model = CNN([8, 16], 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert np.mean((model(x).argmax(1) == y).numpy()) == 1.0
