#!/usr/bin/env bash
# Cria labels, milestones e as issues do plano no seu repositório GitHub.
#
# Pré-requisitos:
#   1. GitHub CLI instalado: https://cli.github.com
#   2. Login feito com `gh auth login` (o token fica guardado pelo gh, nunca neste arquivo)
#   3. Rodar dentro da pasta do repositório, já com o remoto configurado
#
# Uso: bash scripts/setup_github_backlog.sh
# Rodar duas vezes cria issues duplicadas: rode uma vez só.

set -euo pipefail

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Repositório: $REPO"

# ---------- Labels ----------
for spec in "fase-0:BFD4F2" "fase-1:C5DEF5" "fase-2:BFDADC" "fase-3:D4C5F9" "fase-4:FEF2C0" \
            "estudo:E99695" "codigo:0E8A16" "testes:1D76DB" "docs:FBCA04" "experimento:5319E7"; do
  gh label create "${spec%%:*}" --color "${spec##*:}" --force >/dev/null
done

# ---------- Milestones ----------
make_milestone () {
  gh api "repos/$REPO/milestones" -f title="$1" -f description="$2" >/dev/null 2>&1 || true
}
make_milestone "Fase 0 - Setup"          "Repositório, ambiente, dados e fundamentos"
make_milestone "Fase 1 - NumPy (v0.1)"   "Rede neural do zero, acurácia de teste >= 85%"
make_milestone "Fase 2 - PyTorch (v0.2)" "Mesma MLP em PyTorch + equivalência, >= 87%"
make_milestone "Fase 3 - CNN (v0.3)"     "CNN + experimentos E0-E6, >= 91%"
make_milestone "Fase 4 - Lançamento (v1.0)" "CI, demo, README final"

issue () { # título, milestone, labels, corpo
  gh issue create --title "$1" --milestone "$2" --label "$3" --body "$(printf '%b' "$4")" >/dev/null
  echo "  + $1"
}

M0="Fase 0 - Setup"; M1="Fase 1 - NumPy (v0.1)"; M2="Fase 2 - PyTorch (v0.2)"
M3="Fase 3 - CNN (v0.3)"; M4="Fase 4 - Lançamento (v1.0)"

echo "Fase 0"
issue "Configurar ambiente virtual e dependências" "$M0" "fase-0,codigo" "- [ ] venv criado\n- [ ] pip install -r requirements.txt sem erros\n- [ ] pre-commit install"
issue "Validar carregamento do Fashion-MNIST" "$M0" "fase-0,codigo" "python -m src.data imprime (50000, 784), (10000, 784), (10000, 784)"
issue "Notebook 00: exploração dos dados" "$M0" "fase-0,docs" "25 imagens com rótulos, contagem por classe, imagem média por classe"
issue "Estudo: regra da cadeia e backprop" "$M0" "fase-0,estudo" "3Blue1Brown cap. 3 e 4; derivar Dense e Softmax+CE no papel; registrar em docs/derivacoes.md"

echo "Fase 1"
issue "TODO 0: gradiente numérico (gradcheck)" "$M1" "fase-1,codigo" "Implementar numerical_gradient. É a base de todos os outros testes."
issue "TODOs 1-3: camada Dense" "$M1" "fase-1,codigo" "pytest tests/test_numpy_layers.py -k dense -v"
issue "TODOs 6-11: ativações ReLU, Sigmoid e Tanh" "$M1" "fase-1,codigo" "pytest tests/test_numpy_layers.py -k ativacoes -v"
issue "TODOs 12-14: Softmax + Cross-Entropy" "$M1" "fase-1,codigo" "pytest tests/test_numpy_layers.py -k 'softmax or cross' -v"
issue "TODOs 18-21: Sequential" "$M1" "fase-1,codigo" "pytest tests/test_numpy_training.py -k sequential -v"
issue "TODOs 15-17: otimizadores SGD, Momentum e Adam" "$M1" "fase-1,codigo" "pytest tests/test_numpy_training.py -k otimizadores -v"
issue "Teste de overfit passando" "$M1" "fase-1,testes" "pytest tests/test_numpy_training.py -k overfit -v"
issue "TODOs 4-5: Dropout" "$M1" "fase-1,codigo" "pytest tests/test_numpy_layers.py -k dropout -v"
issue "Treino completo da MLP NumPy" "$M1" "fase-1,experimento" "python -m src.numpy_nn.train --config configs/numpy_mlp.yaml; meta: validação >= 85%"
issue "Comparar SGD, Momentum e Adam" "$M1" "fase-1,experimento" "Três configs, um gráfico com as três curvas de validação no notebook 01"
issue "Notebook 01: análise da MLP NumPy" "$M1" "fase-1,docs" "Curvas, matriz de confusão, erros confiantes, histogramas de ativação"
issue "Avaliar no teste e criar tag v0.1" "$M1" "fase-1" "python -m src.numpy_nn.train --config configs/numpy_mlp.yaml --evaluate-test; git tag v0.1"

echo "Fase 2"
issue "Estudo: tensores, autograd, nn.Module e DataLoader" "$M2" "fase-2,estudo" "Tutoriais oficiais 'Learn the Basics' do PyTorch"
issue "Ler e entender src/torch_nn/models.py e train.py" "$M2" "fase-2,estudo" "Comentar cada um dos 5 passos do loop de treino com suas palavras"
issue "Teste de equivalência NumPy x PyTorch passando" "$M2" "fase-2,testes" "pytest tests/test_equivalence.py -v"
issue "Treino da MLP PyTorch" "$M2" "fase-2,experimento" "python -m src.torch_nn.train --config configs/torch_mlp.yaml; meta >= 87%"
issue "Tabela comparativa NumPy x PyTorch" "$M2" "fase-2,docs" "python -m scripts.compare_numpy_torch; colar no README com um parágrafo de análise"
issue "Avaliar no teste e criar tag v0.2" "$M2" "fase-2" "--evaluate-test; git tag v0.2"

echo "Fase 3"
issue "Estudo: convolução, padding, stride e pooling" "$M3" "fase-3,estudo" "CS231n, módulo ConvNets; calcular formatos à mão"
issue "Opcional: Conv2D em NumPy com gradient check" "$M3" "fase-3,codigo" "Implementação lenta com laços, só para aprender"
issue "E0: CNN base" "$M3" "fase-3,experimento" "python -m scripts.run_experiments --only E0_base --seeds 0 1 2"
issue "E1 a E5: experimentos controlados" "$M3" "fase-3,experimento" "Um PR por experimento, com o resultado na descrição"
issue "E6: melhor combinação" "$M3" "fase-3,experimento" "Ajustar configs/experiments.yaml a partir de E1-E5"
issue "Rastrear experimentos (TensorBoard ou W&B)" "$M3" "fase-3,experimento" "Prints dos painéis no README. Chave do W&B em variável de ambiente, nunca no código."
issue "Interpretabilidade: filtros, mapas, Grad-CAM" "$M3" "fase-3,docs" "python -m src.torch_nn.interpret --checkpoint checkpoints/E6_final.pt"
issue "Avaliar no teste e criar tag v0.3" "$M3" "fase-3" "Uma única avaliação de teste; meta >= 91%; git tag v0.3"

echo "Fase 4"
issue "Cobertura de testes >= 80%" "$M4" "fase-4,testes" "pytest --cov=src --cov-report=term-missing"
issue "CI verde no GitHub Actions" "$M4" "fase-4,testes" "Badge no README"
issue "Demo Gradio local" "$M4" "fase-4,codigo" "python -m scripts.export_examples && python app/app.py"
issue "Publicar demo no Hugging Face Spaces" "$M4" "fase-4,codigo" "Ver docs no GUIA; token só em Secrets"
issue "README final" "$M4" "fase-4,docs" "Seguir a estrutura do README modelo; GIF da demo; seção O que aprendi"
issue "Release v1.0 e divulgação" "$M4" "fase-4,docs" "Release com notas; fixar no perfil; topics; post no LinkedIn"

echo "Pronto! Veja em: https://github.com/$REPO/issues"
