"""Testes de fumaça: rodam os pipelines completos (treino, avaliação, gráficos,
checkpoints, interpretabilidade) com um dataset sintético minúsculo.

Não verificam acurácia; verificam que tudo executa de ponta a ponta e gera os
arquivos esperados. Pegam erros de integração que os testes unitários não pegam.
"""

import gzip
import struct

import numpy as np
import pytest

from src import data as data_module
from src.data import Dataset


def fake_dataset(n_train=96, n_eval=40, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)

    def split(n):
        return rng.random((n, 784), dtype=np.float32), rng.integers(0, 10, n)

    return Dataset(*split(n_train), *split(n_eval), *split(n_eval))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Roda cada teste numa pasta temporária (checkpoints/ e results/ vão para lá)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------- Leitura do formato IDX ----------

def write_idx(path, array, magic, header_dims):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">I", magic))
        for d in header_dims:
            f.write(struct.pack(">I", d))
        f.write(array.astype(np.uint8).tobytes())


def test_load_fashion_mnist_com_arquivos_idx_falsos(tmp_path):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (30, 28, 28))
    labels = rng.integers(0, 10, 30)
    for key in ("train", "test"):
        prefix = "train" if key == "train" else "t10k"
        write_idx(tmp_path / f"{prefix}-images-idx3-ubyte.gz", imgs, 2051, (30, 28, 28))
        write_idx(tmp_path / f"{prefix}-labels-idx1-ubyte.gz", labels, 2049, (30,))
    ds = data_module.load_fashion_mnist(tmp_path, val_size=10, seed=1)
    assert ds.X_train.shape == (20, 784) and ds.X_val.shape == (10, 784)
    assert ds.X_test.shape == (30, 784)
    assert ds.X_train.dtype == np.float32 and 0.0 <= ds.X_train.min() <= ds.X_train.max() <= 1.0
    # A divisão treino/validação é uma permutação: nenhum exemplo se perde.
    assert sorted(np.concatenate([ds.y_train, ds.y_val]).tolist()) == sorted(labels.tolist())
    assert np.allclose(ds.X_test[0], imgs[0].ravel() / 255.0)


# ---------- Utilitários ----------

def test_config_deep_update_e_save(tmp_path):
    from src.utils.config import deep_update, load_config, save_config

    base = {"model": {"dropout": 0.0, "hidden": [8]}, "seed": 1}
    out = deep_update(base, {"model": {"dropout": 0.5}})
    assert out == {"model": {"dropout": 0.5, "hidden": [8]}, "seed": 1}
    assert base["model"]["dropout"] == 0.0  # o original não é alterado
    save_config(out, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == out


def test_metrics(tmp_path):
    from src.utils.metrics import MetricsLogger, accuracy, confusion_matrix

    y, p = np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2])
    assert accuracy(y, p) == 0.75
    cm = confusion_matrix(y, p, num_classes=3)
    assert cm.sum() == 4 and cm[1, 2] == 1 and np.trace(cm) == 3
    log = MetricsLogger(tmp_path / "m.csv")
    log.log(epoch=1, val_acc=0.5)
    log.log(epoch=2, val_acc=0.7)
    assert log.column("val_acc") == [0.5, 0.7]
    assert (tmp_path / "m.csv").read_text().count("\n") == 3


def test_seed_reproduzivel():
    from src.utils.seed import set_seed

    a = set_seed(7).random(3)
    b = set_seed(7).random(3)
    assert np.array_equal(a, b)


def test_graficos_sao_gerados(tmp_path):
    from src.utils.plots import (
        plot_confusion_matrix,
        plot_experiment_comparison,
        plot_image_grid,
        plot_training_curves,
    )

    h = {"train_loss": [1, 0.5], "val_loss": [1.1, 0.7],
         "train_acc": [0.5, 0.8], "val_acc": [0.4, 0.7]}
    plot_training_curves(h, tmp_path / "c.png", title="t")
    plot_confusion_matrix(np.eye(3, dtype=int) * 5 + 1, ["a", "b", "c"], tmp_path / "cm.png")
    plot_image_grid(np.zeros((3, 784)), ["x", "y", "z"], tmp_path / "g.png", ncols=2)
    plot_experiment_comparison(["E0", "E1"], [90.0, 91.0], [0.1, 0.2], tmp_path / "e.png")
    assert all((tmp_path / f).stat().st_size > 0 for f in ("c.png", "cm.png", "g.png", "e.png"))


# ---------- Pipeline NumPy ----------

def test_treino_numpy_de_ponta_a_ponta(workdir, monkeypatch):
    from src.numpy_nn import train as np_train

    monkeypatch.setattr(np_train, "load_fashion_mnist", lambda: fake_dataset())
    cfg = {
        "seed": 0, "output_dir": "results/np",
        "model": {"hidden": [16], "activation": "relu", "dropout": 0.1, "init": "he"},
        "optimizer": {"name": "momentum", "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
        "training": {"epochs": 3, "batch_size": 32, "patience": 1},
    }
    result = np_train.train(cfg, evaluate_test=True)
    assert 0.0 <= result["test_acc"] <= 1.0 and result["best_epoch"] >= 1
    for f in ("config.yaml", "metrics.csv", "curves.png", "confusion_matrix.png"):
        assert (workdir / "results/np" / f).exists()
    assert (workdir / "checkpoints/np.npz").exists()


# ---------- Pipeline PyTorch ----------

torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parametrize("model_cfg, extras", [
    ({"type": "mlp", "hidden": [16], "dropout": 0.0},
     {"optimizer": {"name": "sgd", "lr": 0.05, "momentum": 0.9},
      "scheduler": {"name": "step", "step_size": 1}}),
    ({"type": "cnn", "channels": [4, 8], "hidden": 16, "batchnorm": True, "dropout": 0.25},
     {"optimizer": {"name": "adam", "lr": 1e-3}, "scheduler": {"name": "cosine"}}),
])
def test_treino_torch_de_ponta_a_ponta(workdir, monkeypatch, model_cfg, extras):
    from src.torch_nn import train as th_train

    monkeypatch.setattr(th_train, "load_fashion_mnist", lambda: fake_dataset())
    cfg = {"seed": 0, "output_dir": f"results/{model_cfg['type']}", "model": model_cfg,
           "training": {"epochs": 2, "batch_size": 32, "patience": 1,
                        "augmentation": model_cfg["type"] == "cnn"}, **extras}
    result = th_train.train(cfg, evaluate_test=True)
    assert 0.0 <= result["test_acc"] <= 1.0
    ckpt = torch.load(result["checkpoint"], weights_only=True)
    assert ckpt["config"]["model"] == model_cfg
    assert (workdir / f"results/{model_cfg['type']}/confusion_matrix.png").exists()


@pytest.mark.torch
def test_builders_rejeitam_opcoes_invalidas():
    from src.torch_nn.models import MLP
    from src.torch_nn.train import build_optimizer, build_scheduler

    model = MLP([4])
    with pytest.raises(ValueError):
        build_optimizer({"name": "rmsprop", "lr": 0.1}, model)
    opt = build_optimizer({"name": "adam", "lr": 0.1}, model)
    assert build_scheduler(None, opt, 5) is None
    with pytest.raises(ValueError):
        build_scheduler({"name": "plateau"}, opt, 5)


@pytest.mark.torch
def test_interpretabilidade(workdir, monkeypatch):
    from src.torch_nn import interpret
    from src.torch_nn import train as th_train

    monkeypatch.setattr(th_train, "load_fashion_mnist", lambda: fake_dataset())
    cfg = {"seed": 0, "output_dir": "results/cnn_i",
           "model": {"type": "cnn", "channels": [4, 8], "hidden": 16},
           "optimizer": {"name": "adam", "lr": 1e-3},
           "training": {"epochs": 1, "batch_size": 32, "patience": 1}}
    result = th_train.train(cfg, verbose=False)

    model, loaded_cfg = interpret.load_model(result["checkpoint"])
    assert loaded_cfg["model"]["type"] == "cnn"
    ds = fake_dataset()
    cam = interpret.grad_cam(model, torch.from_numpy(ds.X_val[0]))
    assert cam.shape == (28, 28) and cam.min() >= 0 and cam.max() <= 1.0 + 1e-6
    X_err, titles = interpret.most_confident_errors(model, ds.X_val, ds.y_val, k=4)
    assert len(X_err) == len(titles) <= 4
    interpret.plot_first_layer_filters(model, workdir / "f.png")
    interpret.plot_feature_maps(model, ds.X_val[0], workdir / "m.png", n_maps=4)
    interpret.plot_grad_cam(model, ds.X_val[:3], ds.y_val[:3], workdir / "g.png")
    assert all((workdir / f).exists() for f in ("f.png", "m.png", "g.png"))
