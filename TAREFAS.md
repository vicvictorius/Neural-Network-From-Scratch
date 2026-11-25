# Tarefas do projeto

Backlog local, enquanto o projeto não está no GitHub. Os itens abertos são os que dependem de você: estudo, leitura do código, escrita pessoal e publicação. Marque `[x]` ao concluir e faça commit deste arquivo junto com o trabalho da tarefa, para o histórico do Git registrar o progresso.


## Fase 0 – Setup

- [x] **Configurar ambiente virtual e dependências**: venv criado pip install -r requirements.txt sem erros pre-commit install
- [x] **Validar carregamento do Fashion-MNIST**: python -m src.data imprime (50000, 784), (10000, 784), (10000, 784)
- [ ] **Notebook 00: exploração dos dados**: 25 imagens com rótulos, contagem por classe, imagem média por classe
- [ ] **Estudo: regra da cadeia e backprop**: 3Blue1Brown cap. 3 e 4; derivar Dense e Softmax+CE no papel; registrar em docs/derivacoes.md

## Fase 1 – NumPy (v0.1)

- [x] **TODO 0: gradiente numérico (gradcheck)**: Implementar numerical_gradient. É a base de todos os outros testes.
- [x] **TODOs 1-3: camada Dense**: pytest tests/test_numpy_layers.py -k dense -v
- [x] **TODOs 6-11: ativações ReLU, Sigmoid e Tanh**: pytest tests/test_numpy_layers.py -k ativacoes -v
- [x] **TODOs 12-14: Softmax + Cross-Entropy**: pytest tests/test_numpy_layers.py -k 'softmax or cross' -v
- [x] **TODOs 18-21: Sequential**: pytest tests/test_numpy_training.py -k sequential -v
- [x] **TODOs 15-17: otimizadores SGD, Momentum e Adam**: pytest tests/test_numpy_training.py -k otimizadores -v
- [x] **Teste de overfit passando**: pytest tests/test_numpy_training.py -k overfit -v
- [x] **TODOs 4-5: Dropout**: pytest tests/test_numpy_layers.py -k dropout -v
- [x] **Treino completo da MLP NumPy**: python -m src.numpy_nn.train --config configs/numpy_mlp.yaml; meta: validação >= 85%
- [x] **Comparar SGD, Momentum e Adam**: Três configs, um gráfico com as três curvas de validação no notebook 01
- [x] **Notebook 01: análise da MLP NumPy**: Curvas, matriz de confusão, erros confiantes, histogramas de ativação
- [x] **Avaliar no teste e criar tag v0.1**: python -m src.numpy_nn.train --config configs/numpy_mlp.yaml --evaluate-test; git tag v0.1

## Fase 2 – PyTorch (v0.2)

- [ ] **Estudo: tensores, autograd, nn.Module e DataLoader**: Tutoriais oficiais 'Learn the Basics' do PyTorch
- [ ] **Ler e entender src/torch_nn/models.py e train.py**: Comentar cada um dos 5 passos do loop de treino com suas palavras
- [x] **Teste de equivalência NumPy x PyTorch passando**: pytest tests/test_equivalence.py -v
- [x] **Treino da MLP PyTorch**: python -m src.torch_nn.train --config configs/torch_mlp.yaml; meta >= 87%
- [x] **Tabela comparativa NumPy x PyTorch**: python -m scripts.compare_numpy_torch; colar no README com um parágrafo de análise
- [x] **Avaliar no teste e criar tag v0.2**: --evaluate-test; git tag v0.2

## Fase 3 – CNN (v0.3)

- [ ] **Estudo: convolução, padding, stride e pooling**: CS231n, módulo ConvNets; calcular formatos à mão
- [ ] **Opcional: Conv2D em NumPy com gradient check**: Implementação lenta com laços, só para aprender
- [x] **E0: CNN base**: python -m scripts.run_experiments --only E0_base --seeds 0 1 2
- [x] **E1 a E5: experimentos controlados**: Um PR por experimento, com o resultado na descrição
- [x] **E6: melhor combinação**: Ajustar configs/experiments.yaml a partir de E1-E5
- [ ] **Rastrear experimentos (TensorBoard ou W&B)**: Prints dos painéis no README. Chave do W&B em variável de ambiente, nunca no código.
- [x] **Interpretabilidade: filtros, mapas, Grad-CAM**: python -m src.torch_nn.interpret --checkpoint checkpoints/E6_final.pt
- [x] **Avaliar no teste e criar tag v0.3**: Uma única avaliação de teste; meta >= 91%; git tag v0.3

## Fase 4 – Lançamento (v1.0)

- [x] **Cobertura de testes >= 80%**: pytest --cov=src --cov-report=term-missing
- [ ] **CI verde no GitHub Actions**: Badge no README
- [x] **Demo Gradio local**: python -m scripts.export_examples && python app/app.py
- [ ] **Publicar demo no Hugging Face Spaces**: Ver docs no GUIA; token só em Secrets
- [x] **README final**: Seguir a estrutura do README modelo; GIF da demo; seção O que aprendi
- [ ] **Release v1.0 e divulgação**: Release com notas; fixar no perfil; topics; post no LinkedIn

## Fase 0 – adaptações para trabalhar sem GitHub

- [x] Repositório Git local criado e primeiro commit feito
- [x] `scripts/check.sh` (ou `check.ps1`) rodando antes de cada commit, no lugar do CI
