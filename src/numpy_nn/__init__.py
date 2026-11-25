"""Mini-biblioteca de redes neurais em NumPy (Fase 1)."""

from .activations import ReLU, Sigmoid, Tanh
from .layers import Dense, Dropout, Layer
from .losses import SoftmaxCrossEntropy
from .model import Sequential
from .optimizers import SGD, Adam, Momentum

__all__ = [
    "Layer", "Dense", "Dropout", "ReLU", "Sigmoid", "Tanh",
    "SoftmaxCrossEntropy", "Sequential", "SGD", "Momentum", "Adam",
]
