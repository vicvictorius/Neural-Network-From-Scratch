# Guia do kit: do zero à v1.0

Este guia diz exatamente o que fazer em cada fase, com os comandos. Ele complementa o plano do projeto (o documento com objetivos, cronograma e critérios de aceite).

## O que já vem pronto e o que você escreve

| Parte | Situação | Por quê |
| --- | --- | --- |
| Infraestrutura (CI, ruff, pre-commit, templates, .gitignore) | Pronta | Não é o foco do aprendizado; é o padrão profissional |
| `src/data.py` e `src/utils/` | Prontos | Leia e entenda, mas o foco está nas redes |
| `src/numpy_nn/` | **Exercícios com 22 TODOs** | É aqui que você aprende deep learning de verdade |
| `src/numpy_nn/train.py` | Pronto | Leia linha a linha depois que os testes passarem |
| `src/torch_nn/` | Pronto e comentado | Na Fase 2 o objetivo é entender o PyTorch e comparar |
| Testes (`tests/`) | Prontos | Funcionam como especificação: implemente até passarem |
| `gabarito/` | Solução da Fase 1, fora do Git | Só para destravar, nunca para copiar |
| README, notebooks, `docs/aprendizados.md` | Modelos para você preencher | É a parte que mostra o SEU raciocínio |

Para o projeto ser honestamente seu: reescreva os arquivos prontos à sua maneira quando quiser, e explique tudo com suas palavras no README. Em entrevistas, vão perguntar sobre qualquer linha.

## Modo local (enquanto o projeto não vai para o GitHub)

O projeto funciona inteiro sem GitHub: o Git local guarda commits, branches e tags, e o histórico vai junto quando você publicar.

| No GitHub | No modo local |
| --- | --- |
| Issues e milestones | `TAREFAS.md` (marque `[x]` e faça commit junto com a tarefa) |
| GitHub Actions (CI) | `bash scripts/check.sh` ou `.\scripts\check.ps1` antes de cada commit; use o argumento `numpy` durante a Fase 1 |
| Pull request | Branch local + `git merge` no `main` |
| `gh auth login` e `setup_github_backlog.sh` | Pule por enquanto |

Se a rede bloquear o download dos dados, baixe pelo navegador os 4 arquivos `.gz` do repositório público `zalandoresearch/fashion-mnist` (pasta `data/fashion`) e coloque-os em `data/fashion-mnist/`.

Para publicar depois: leve a pasta do projeto inteira, incluindo a pasta oculta `.git`, para a máquina pessoal, crie o repositório vazio no GitHub e rode `git remote add origin ...` e `git push -u origin main --tags`. Em máquina corporativa, siga a política interna e use o meio de transferência aprovado.

## Situação atual do projeto

O código está completo e os resultados foram gerados nesta versão do kit: MLP NumPy 88,86%, MLP PyTorch 89,08% e CNN 92,39% de teste, com 49 testes e 93% de cobertura. Os experimentos da CNN usaram o protocolo rápido (`configs/experiments_rapido.yaml`: 1 seed, 8 épocas). Os resultados ficam em `results/` (versionado) e os pesos em `checkpoints/` (fora do Git).

Ficam com você: escrever as seções "Por que construir do zero" e "O que aprendi" do README, responder as perguntas dos notebooks, os itens abertos do `TAREFAS.md` e a publicação. Se tiver uma máquina mais rápida, rode o protocolo completo com `python -m scripts.run_experiments --seeds 0 1 2` (é retomável: se parar no meio, rode de novo).

As seções das fases abaixo continuam valendo como referência de como cada etapa foi feita e de como refazê-la.

## Fase 0: primeiros passos

### 1. Criar o repositório

1. No GitHub, crie um repositório **vazio** chamado `neural-network-from-scratch` (sem README, sem .gitignore).
2. No terminal, dentro da pasta descompactada do kit:

```bash
git init
git add .
git commit -m "chore: estrutura inicial do projeto"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/neural-network-from-scratch.git
git push -u origin main
```

Confira com `git status` que nada de `gabarito/`, `data/` ou `checkpoints/` foi enviado.

### 2. Ambiente

```bash
python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows:   .venv\Scripts\activate
python -m pip install --upgrade pip
# Só CPU (mais leve):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pre-commit install
```

### 3. Dados e backlog

```bash
python -m src.data                       # baixa e mostra os formatos
gh auth login                            # uma vez; o token fica com o gh
bash scripts/setup_github_backlog.sh     # cria 36 issues, 5 milestones e labels
```

No GitHub, crie um Project (aba Projects > New project > Board) e adicione as issues. Mova cada uma para "In Progress" e "Done" conforme avança.

### 4. Estudo

Antes da Fase 1, derive no papel as seções 1 e 3 de `docs/derivacoes.md`.

## Fase 1: NumPy do zero

### Ordem dos TODOs (cada um destrava os testes do seguinte)

| Ordem | TODO | Arquivo | Teste para rodar |
| --- | --- | --- | --- |
| 1 | 0 | gradcheck.py | (base de todos) |
| 2 | 1, 2, 3 | layers.py (Dense) | `pytest tests/test_numpy_layers.py -k dense -v` |
| 3 | 6 a 11 | activations.py | `pytest tests/test_numpy_layers.py -k "ativacoes or sigmoid" -v` |
| 4 | 12, 13, 14 | losses.py | `pytest tests/test_numpy_layers.py -k "softmax or cross" -v` |
| 5 | 18 a 21 | model.py | `pytest tests/test_numpy_training.py -k sequential -v` |
| 6 | 15, 16, 17 | optimizers.py | `pytest tests/test_numpy_training.py -k otimizadores -v` |
| 7 | (nenhum) | teste de overfit | `pytest tests/test_numpy_training.py -k overfit -v` |
| 8 | 4, 5 | layers.py (Dropout) | `pytest tests/test_numpy_layers.py -k dropout -v` |

### Fluxo de trabalho por TODO

```bash
git checkout -b feat/dense-layer
# implementa, roda o teste até passar
ruff check . && pytest tests/test_numpy_layers.py -k dense
git add src/numpy_nn/layers.py
git commit -m "feat: camada Dense com forward e backward"
git push -u origin feat/dense-layer
# abre o PR no GitHub, revisa, faz merge, e:
git checkout main && git pull
```

### Treino

```bash
python -m src.numpy_nn.train --config configs/numpy_mlp.yaml
```

Na versão de referência, 4 épocas já chegam a cerca de 87% de validação, em ~2 s por época numa CPU comum. Se o seu ficar muito abaixo, há bug: volte aos gradient checks.

Para comparar otimizadores, copie o YAML para `configs/numpy_mlp_sgd.yaml` e `configs/numpy_mlp_momentum.yaml`, troque `optimizer.name`, `lr` (sugestão: SGD 0.1, Momentum 0.05) e `output_dir`, e plote as três curvas de `metrics.csv` no notebook.

Só no fim da fase:

```bash
python -m src.numpy_nn.train --config configs/numpy_mlp.yaml --evaluate-test
git tag v0.1 && git push --tags
```

## Fase 2: PyTorch

1. Leia `src/torch_nn/models.py` e `train.py` comparando com a sua versão NumPy. Anote em `docs/aprendizados.md` o que corresponde a quê (ex.: `loss.backward()` substitui o seu `model.backward(loss_fn.backward())`).
2. Rode `pytest tests/test_equivalence.py -v`. Se falhar, a diferença está na sua implementação NumPy (ou na transposição dos pesos).
3. Treine e compare:

```bash
python -m src.torch_nn.train --config configs/torch_mlp.yaml
python -m scripts.compare_numpy_torch
```

4. Escreva no README o parágrafo de análise. Pergunta-guia: o que o autograd faz por você, e quando entender o backprop manual ainda é útil?

## Fase 3: CNN e experimentos

```bash
python -m scripts.run_experiments --only E0_base --seeds 0     # teste rápido
python -m scripts.run_experiments --seeds 0 1 2                # tudo (demora)
```

Sem GPU, cada época da CNN leva de 20 s a 1 min. Para rodar tudo no Google Colab:

```python
!git clone https://github.com/SEU_USUARIO/neural-network-from-scratch.git
%cd neural-network-from-scratch
!pip install -q pyyaml
!python -m scripts.run_experiments --seeds 0 1 2
# Baixe results/ e checkpoints/ antes de fechar a sessão.
```

Com base nos resultados de E1 a E5, monte o modelo final em `configs/cnn_final.yaml` (no protocolo completo, o equivalente é o `E6_melhor` de `configs/experiments.yaml`). Depois:

```bash
python -m src.torch_nn.train --config configs/cnn_final.yaml --evaluate-test
python -m src.torch_nn.interpret --checkpoint checkpoints/E6_final.pt
```

Regra de ouro: escolha tudo pela validação. O teste é usado uma única vez, no modelo final.

## Fase 4: lançamento

### Cobertura e CI

```bash
pytest --cov=src --cov-report=term-missing
```

O CI roda sozinho a cada push. Até a Fase 1 estar completa, os testes de NumPy falham: isso é esperado. Faça o primeiro push com a estrutura e mantenha o `main` verde a partir do fim da Fase 1.

### Demo no Hugging Face Spaces

1. Copie o melhor checkpoint: `cp checkpoints/E6_final.pt app/model.pt` e rode `python -m scripts.export_examples`.
2. Teste local: `python app/app.py`.
3. Em huggingface.co, crie um Space (SDK Gradio, gratuito).
4. Envie para o Space: `app.py`, `model.pt`, a pasta `examples/`, `app/requirements.txt` (como `requirements.txt`), `app/README.md` (como `README.md`) e uma cópia de `src/torch_nn/models.py` ao lado do `app.py`.
5. Se for automatizar o envio pelo GitHub Actions, guarde o token do Hugging Face em Settings > Secrets do GitHub. Nunca no código, no YAML ou no notebook.

### Release

```bash
git tag v1.0 && git push --tags
```

No GitHub: Releases > Draft a new release > tag v1.0, com um resumo dos resultados.

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
| --- | --- | --- |
| `ModuleNotFoundError: src` | Rodou de dentro de uma subpasta | Rode da raiz e com `python -m` |
| Perda vira `nan` | Softmax sem subtrair o máximo, ou lr alto | Revise TODO 12; reduza lr 10x |
| Perda não cai | Backward errado ou otimizador não altera in-place | Gradient check; `p -= ...` em vez de `p = p - ...` |
| Gradient check falha só na ReLU | x muito perto de 0 | O teste já evita; confira se a máscara é `x > 0` |
| Acurácia de treino 100%, validação estagnada | Overfitting | Dropout, L2, augmentation, early stopping |
| Validação muito acima do esperado | Vazamento de dados | Confirme que o teste não entra no treino nem na escolha de hiperparâmetros |
| CI falha com erro de instalação do torch | Versão incompatível | Mantenha o `--index-url` de CPU no workflow |

## Depois da v1.0

- Escreva o post: o que construiu, o resultado principal, um bug que ensinou algo, link da demo e do repositório.
- Fixe o repositório no seu perfil e adicione topics: `deep-learning`, `neural-network`, `numpy`, `pytorch`, `from-scratch`, `fashion-mnist`.
- Escolha uma extensão do plano (mini-autograd, transfer learning, ONNX + FastAPI).
