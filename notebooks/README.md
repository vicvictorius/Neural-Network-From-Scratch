# Notebooks

Notebooks servem para explorar e visualizar. Toda lógica reutilizável vive em `src/` e é importada aqui:

```python
import sys; sys.path.append("..")
from src.data import load_fashion_mnist, CLASS_NAMES
```

Antes de cada commit: reinicie o kernel, rode tudo do início (Restart & Run All) e salve, para que os resultados exibidos sejam reproduzíveis.

| Notebook | Fase | Conteúdo |
| --- | --- | --- |
| `00_exploracao.ipynb` | 0 | 25 imagens com rótulos, contagem por classe, imagem média por classe |
| `01_numpy_mlp.ipynb` | 1 | curvas de treino, comparação SGD x Momentum x Adam, matriz de confusão, erros confiantes, histogramas das ativações por camada |
| `02_numpy_vs_pytorch.ipynb` | 2 | tabela comparativa, tempo por época, discussão sobre autograd |
| `03_cnn_experimentos.ipynb` | 3 | tabela e gráfico dos experimentos, curvas sobrepostas de E0 x E6 |
| `04_interpretabilidade.ipynb` | 3 | filtros, mapas de ativação, Grad-CAM, análise das classes confundidas |
