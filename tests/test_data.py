import numpy as np

from src.data import iterate_minibatches, one_hot


def test_minibatches_cobrem_todos_os_exemplos(rng):
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    seen = np.concatenate([yb for _, yb in iterate_minibatches(X, y, 16, rng=rng)])
    assert sorted(seen.tolist()) == list(range(50))


def test_minibatches_ultimo_batch_menor(rng):
    X, y = np.zeros((50, 2)), np.zeros(50)
    sizes = [len(xb) for xb, _ in iterate_minibatches(X, y, 16, shuffle=False)]
    assert sizes == [16, 16, 16, 2]


def test_minibatches_mantem_pares_x_y(rng):
    X = np.arange(30).reshape(30, 1)
    y = np.arange(30)
    for xb, yb in iterate_minibatches(X, y, 7, rng=rng):
        assert np.array_equal(xb[:, 0], yb)


def test_one_hot():
    out = one_hot(np.array([0, 2]), num_classes=3)
    assert np.array_equal(out, [[1, 0, 0], [0, 0, 1]])
