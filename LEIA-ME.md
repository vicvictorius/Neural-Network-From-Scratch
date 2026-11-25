# Kit do projeto: rede neural do zero ao PyTorch

Conteúdo deste pacote:

- `neural-network-from-scratch/`: o repositório completo. É esta pasta que vira o seu projeto Git. Comece por `docs/GUIA.md` (seção "Modo local") e pelo `TAREFAS.md`.
- `extras/prompt-mentor-neural-network.md`: prompt para continuar o projeto em um chat novo, já atualizado com a situação atual (metade da Fase 1 concluída). Fica fora do repositório de propósito.

Estado atual: código completo, resultados gerados e README preenchido.

| Modelo | Teste | Meta |
| --- | --- | --- |
| MLP NumPy | 88,86 % | ≥ 85 % |
| MLP PyTorch | 89,08 % | ≥ 87 % |
| CNN final | 92,39 % | ≥ 91 % |

49 testes, 93% de cobertura. Rode `bash scripts/check.sh` (ou `.\scripts\check.ps1` no Windows) dentro do repositório para conferir.

Para ver os resultados, comece pelo `README.md` do repositório e pelos notebooks (já executados). Os pesos treinados estão em `checkpoints/`, fora do Git: para publicar a demo, copie `checkpoints/E6_final.pt` como `app/model.pt`, conforme `docs/GUIA.md`.

O que falta, e é seu: as seções "Por que construir do zero" e "O que aprendi" do README (marcadas com comentários), as respostas às perguntas dos notebooks e a publicação no GitHub e no Hugging Face.

Lembrete: a pasta `neural-network-from-scratch/gabarito/` está no `.gitignore` e não deve ser publicada.
