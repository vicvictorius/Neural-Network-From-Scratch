"""Seeds fixas para resultados reproduzíveis."""

import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Fixa as seeds de Python, NumPy e (se instalado) PyTorch.

    Retorna um Generator do NumPy para ser passado adiante explicitamente,
    o que é preferível a depender do estado global.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return np.random.default_rng(seed)
