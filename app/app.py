"""Demo interativa (Fase 4).

Rodar localmente:
    python -m scripts.export_examples      # uma vez, para gerar os exemplos
    python app/app.py

No Hugging Face Spaces, este arquivo é o ponto de entrada. Copie para o Space:
app.py, model.pt (o checkpoint final), a pasta examples/ e o requirements.txt
desta pasta, e o arquivo models.py de src/torch_nn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # permite importar src/ ao rodar no repositório

try:
    from src.torch_nn.models import build_model
except ImportError:  # no Space, models.py fica ao lado de app.py
    from models import build_model

CLASS_NAMES = ["Camiseta/top", "Calça", "Pulôver", "Vestido", "Casaco",
               "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]
CHECKPOINT = ROOT / "model.pt" if (ROOT / "model.pt").exists() else \
    ROOT.parent / "checkpoints" / "E6_final.pt"


def load() -> torch.nn.Module:
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


MODEL = load()


def preprocess(image: Image.Image, invert: bool) -> torch.Tensor:
    """Converte qualquer imagem para o formato do Fashion-MNIST:
    28x28, tons de cinza, objeto claro sobre fundo escuro, valores em [0, 1]."""
    img = ImageOps.grayscale(image)
    if invert:  # fotos reais costumam ter fundo claro
        img = ImageOps.invert(img)
    img = ImageOps.pad(img, (28, 28), color=0)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).reshape(1, 1, 28, 28)


def predict(image: Image.Image | None, invert: bool):
    if image is None:
        return {}, None
    x = preprocess(image, invert)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1)[0].numpy()
    preview = Image.fromarray((x[0, 0].numpy() * 255).astype(np.uint8)).resize((140, 140))
    return {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}, preview


examples_dir = ROOT / "examples"
examples = (
    [[str(p), False] for p in sorted(examples_dir.glob("*.png"))]
    if examples_dir.exists()
    else None
)

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Imagem de uma peça de roupa"),
            gr.Checkbox(value=True, label="Inverter cores (use para fotos com fundo claro)")],
    outputs=[gr.Label(num_top_classes=3, label="Previsão"),
             gr.Image(label="O que o modelo vê (28x28)")],
    examples=examples,
    title="Classificador Fashion-MNIST: rede neural do zero ao PyTorch",
    description=(
        "CNN treinada no Fashion-MNIST. Limitação conhecida: o modelo aprendeu com imagens "
        "28x28 centralizadas e sem fundo, então fotos reais funcionam pior. "
        "Código: https://github.com/SEU_USUARIO/neural-network-from-scratch"
    ),
)

if __name__ == "__main__":
    demo.launch()
