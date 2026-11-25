"""Roda os experimentos da Fase 3 com várias seeds e gera a tabela final.

Uso:
    python -m scripts.run_experiments --seeds 0 1 2
    python -m scripts.run_experiments --only E0_base E2_batchnorm --seeds 0

Saídas:
    results/cnn/<experimento>_seed<n>/   (métricas, curvas, config de cada execução)
    results/cnn/experimentos.csv         (média e desvio por experimento)
    results/cnn/experimentos.md          (tabela pronta para colar no README)
    results/cnn/experimentos.png         (gráfico comparativo)

Retomável: cada execução concluída grava summary.json na sua pasta e é pulada
nas próximas chamadas. Use --force para refazer tudo.

Escolha o modelo final pela VALIDAÇÃO. Depois rode o teste uma única vez:
    python -m src.torch_nn.train --config configs/cnn_final.yaml --evaluate-test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.torch_nn.train import train
from src.utils.config import deep_update, load_config
from src.utils.plots import plot_experiment_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="configs/experiments.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--only", nargs="*", help="nomes dos experimentos a rodar")
    parser.add_argument("--force", action="store_true", help="refaz execuções já concluídas")
    args = parser.parse_args()

    plan = load_config(args.plan)
    base = load_config(plan["base"])
    out_root = Path(base["output_dir"]).parent
    rows = []

    for name, overrides in plan["experiments"].items():
        if args.only and name not in args.only:
            continue
        accs, secs = [], []
        for seed in args.seeds:
            cfg = deep_update(base, overrides or {})
            cfg["seed"] = seed
            cfg["output_dir"] = str(out_root / f"{name}_seed{seed}")
            summary = Path(cfg["output_dir"]) / "summary.json"
            if summary.exists() and not args.force:
                result = json.loads(summary.read_text(encoding="utf-8"))
                print(f"\n=== {name} | seed {seed} === (já concluído, reaproveitado)")
            else:
                print(f"\n=== {name} | seed {seed} ===", flush=True)
                result = train(cfg, verbose=False)
                summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
            accs.append(100 * result["best_val_acc"])
            secs.append(result["seconds_per_epoch"])
            print(f"validação: {accs[-1]:.2f}%", flush=True)
        rows.append({"experimento": name, "media_val_acc": float(np.mean(accs)),
                     "desvio": float(np.std(accs)), "seg_por_epoca": float(np.mean(secs)),
                     "seeds": len(accs)})

    rows.sort(key=lambda r: -r["media_val_acc"])
    with open(out_root / "experimentos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| Experimento | Acurácia de validação (%) | Desvio | s/época |",
             "| --- | --- | --- | --- |"]
    # Com 1 seed não há desvio padrão: mostrar "± 0,00" sugeriria precisão falsa.
    def spread(r: dict) -> str:
        return f"± {r['desvio']:.2f}" if r["seeds"] > 1 else "n/d (1 seed)"

    lines += [f"| {r['experimento']} | {r['media_val_acc']:.2f} | {spread(r)} | "
              f"{r['seg_por_epoca']:.1f} |" for r in rows]
    (out_root / "experimentos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_experiment_comparison([r["experimento"] for r in rows],
                               [r["media_val_acc"] for r in rows],
                               [r["desvio"] for r in rows], out_root / "experimentos.png")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
