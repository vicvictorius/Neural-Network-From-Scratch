"""Exporta 10 imagens do conjunto de TESTE (uma por classe) como PNG para a demo.

Uso:
    python -m scripts.export_examples
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.data import CLASS_NAMES, load_fashion_mnist


def main() -> None:
    ds = load_fashion_mnist()
    out = Path("app/examples")
    out.mkdir(parents=True, exist_ok=True)
    for c, name in enumerate(CLASS_NAMES):
        i = int(np.where(ds.y_test == c)[0][0])
        img = (ds.X_test[i].reshape(28, 28) * 255).astype(np.uint8)
        # Amplia para 280x280 para ficar visível na interface.
        big = Image.fromarray(img).resize((280, 280), Image.NEAREST)
        big.save(out / f"{c}_{name.split('/')[0]}.png")
    print(f"{len(CLASS_NAMES)} exemplos salvos em {out}/")


if __name__ == "__main__":
    main()
