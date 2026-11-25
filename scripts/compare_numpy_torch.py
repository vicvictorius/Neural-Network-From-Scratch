"""Fase 2: compara a MLP NumPy com a MLP PyTorch e gera a tabela do README.

Uso:
    python -m scripts.compare_numpy_torch

Lê results/numpy_mlp/metrics.csv e results/torch_mlp/metrics.csv (rode os
dois treinos antes) e conta as linhas de código de cada implementação.
"""

from __future__ import annotations

import csv
from pathlib import Path


def read_metrics(path: Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["val_acc"]))
    return {
        "best_val_acc": 100 * float(best["val_acc"]),
        "epochs": len(rows),
        "sec_per_epoch": sum(float(r["seconds"]) for r in rows) / len(rows),
    }


def count_code_lines(paths: list[Path]) -> int:
    """Linhas que não são vazias, comentários nem docstrings de uma linha."""
    total = 0
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and not (s.startswith('"""') and s.endswith('"""')):
                total += 1
    return total


def main() -> None:
    np_m = read_metrics(Path("results/numpy_mlp/metrics.csv"))
    th_m = read_metrics(Path("results/torch_mlp/metrics.csv"))
    np_files = [p for p in Path("src/numpy_nn").glob("*.py") if p.name != "__init__.py"]
    th_files = [Path("src/torch_nn/models.py"), Path("src/torch_nn/train.py")]
    table = [
        "| Critério | NumPy | PyTorch |",
        "| --- | --- | --- |",
        f"| Melhor acurácia de validação (%) | {np_m['best_val_acc']:.2f} | "
        f"{th_m['best_val_acc']:.2f} |",
        f"| Épocas até o early stopping | {np_m['epochs']} | {th_m['epochs']} |",
        f"| Segundos por época | {np_m['sec_per_epoch']:.1f} | {th_m['sec_per_epoch']:.1f} |",
        f"| Linhas de código | {count_code_lines(np_files)} | {count_code_lines(th_files)} |",
        "| Cálculo do gradiente | manual (backward à mão) | automático (autograd) |",
    ]
    out = Path("results/comparacao_numpy_torch.md")
    out.write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))
    print(f"\nTabela salva em {out}")


if __name__ == "__main__":
    main()
