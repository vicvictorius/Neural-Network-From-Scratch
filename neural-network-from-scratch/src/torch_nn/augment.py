"""Data augmentation em batch, direto em tensores (rápido e sem dependências).

Só é aplicado no TREINO. Validação e teste sempre usam as imagens originais.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_horizontal_flip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """x: (N, 1, 28, 28). Espelha cada imagem com probabilidade p.
    Faz sentido para roupas (uma camisa espelhada continua camisa); não faria
    sentido para dígitos (um 3 espelhado não é um 3)."""
    flip = torch.rand(x.shape[0], device=x.device) < p
    out = x.clone()
    out[flip] = out[flip].flip(-1)
    return out


def random_shift(x: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    """Translada cada imagem até max_shift pixels em x e y (preenche com zero)."""
    n = x.shape[0]
    padded = F.pad(x, (max_shift,) * 4)
    dx = torch.randint(0, 2 * max_shift + 1, (n,), device=x.device)
    dy = torch.randint(0, 2 * max_shift + 1, (n,), device=x.device)
    rows = (dy[:, None] + torch.arange(28, device=x.device)[None, :])  # (N, 28)
    cols = (dx[:, None] + torch.arange(28, device=x.device)[None, :])
    batch = torch.arange(n, device=x.device)[:, None, None]
    return padded[batch, 0, rows[:, :, None], cols[:, None, :]].unsqueeze(1)


def augment(x: torch.Tensor) -> torch.Tensor:
    return random_shift(random_horizontal_flip(x))
