"""MLP e CNN em PyTorch.

Os dois modelos aceitam entrada (N, 784) ou (N, 1, 28, 28) e devolvem logits (N, 10).
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Mesma arquitetura da Fase 1: 784 -> 128 -> 64 -> 10 (por padrão)."""

    def __init__(self, hidden: list[int] | None = None, dropout: float = 0.0,
                 n_in: int = 784, n_out: int = 10) -> None:
        super().__init__()
        hidden = hidden or [128, 64]
        layers: list[nn.Module] = []
        sizes = [n_in, *hidden]
        for a, b in zip(sizes[:-1], sizes[1:], strict=True):
            layers += [nn.Linear(a, b), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(sizes[-1], n_out))
        self.net = nn.Sequential(*layers)
        self._init_he()

    def _init_he(self) -> None:
        # Mesma inicialização da versão NumPy, para comparação justa.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class CNN(nn.Module):
    """[Conv 3x3 -> (BatchNorm) -> ReLU -> MaxPool 2x2] x len(channels)
    -> Flatten -> Linear(hidden) -> ReLU -> (Dropout) -> Linear(10)

    Com channels=[32, 64]: 1x28x28 -> 32x14x14 -> 64x7x7 -> 3136 -> 128 -> 10
    """

    def __init__(self, channels: list[int] | None = None, hidden: int = 128,
                 batchnorm: bool = False, dropout: float = 0.0, n_out: int = 10) -> None:
        super().__init__()
        channels = channels or [32, 64]
        blocks: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not batchnorm))
            if batchnorm:
                blocks.append(nn.BatchNorm2d(out_ch))
            act = nn.ReLU()
            blocks.append(act)
            if i == len(channels) - 1:
                self.last_conv_activation = act  # usado pelo Grad-CAM
            blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        side = 28 // (2 ** len(channels))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * side * side, hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.reshape(x.shape[0], 1, 28, 28)))


def build_model(model_cfg: dict) -> nn.Module:
    if model_cfg["type"] == "mlp":
        return MLP(model_cfg.get("hidden", [128, 64]), model_cfg.get("dropout", 0.0))
    if model_cfg["type"] == "cnn":
        return CNN(model_cfg.get("channels", [32, 64]), model_cfg.get("hidden", 128),
                   model_cfg.get("batchnorm", False), model_cfg.get("dropout", 0.0))
    raise ValueError(f"tipo de modelo desconhecido: {model_cfg['type']}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
