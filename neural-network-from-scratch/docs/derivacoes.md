# Derivações

Faça cada derivação no papel ANTES de implementar. Depois, confira com este arquivo e reescreva-o com suas palavras: esta pasta é parte do portfólio.

## Notação

| Símbolo | Significado | Formato |
| --- | --- | --- |
| N | tamanho do batch | escalar |
| X | entrada da camada | (N, n_in) |
| W, b | pesos e bias | (n_in, n_out), (n_out,) |
| Z | saída da camada densa, Z = XW + b | (N, n_out) |
| L | perda (um número) | escalar |
| G | gradiente que chega de cima, dL/dZ | (N, n_out) |

Regra prática que evita a maioria dos bugs: **o gradiente de qualquer coisa tem o mesmo formato dessa coisa.** dL/dW tem formato de W, dL/dX tem formato de X.

## 1. Camada densa

Z = XW + b, ou elemento a elemento: Z[i, j] = soma_k X[i, k] W[k, j] + b[j]

**dL/dW.** W[k, j] influencia Z[i, j] para todo exemplo i, com derivada X[i, k]:

dL/dW[k, j] = soma_i X[i, k] G[i, j]   =>   **dL/dW = Xᵀ G**   (n_in, N) @ (N, n_out)

**dL/db.** b[j] soma 1 em Z[i, j] para todo i:

**dL/db = soma de G nas linhas** (eixo 0)

**dL/dX.** X[i, k] influencia Z[i, j] para todo j, com derivada W[k, j]:

dL/dX[i, k] = soma_j G[i, j] W[k, j]   =>   **dL/dX = G Wᵀ**   (N, n_out) @ (n_out, n_in)

Confira: os três resultados têm os formatos de W, b e X.

## 2. Ativações (aplicadas elemento a elemento)

Se A = f(Z), então dL/dZ = dL/dA * f'(Z) (produto elemento a elemento, não matricial).

| Ativação | f(z) | f'(z) |
| --- | --- | --- |
| ReLU | max(0, z) | 1 se z > 0, senão 0 |
| Sigmoid | 1 / (1 + e^-z) | f(z)(1 - f(z)) |
| Tanh | tanh(z) | 1 - tanh(z)² |

Note que sigmoid e tanh escrevem a derivada em função da própria saída: por isso guardamos a saída no forward.

## 3. Softmax + Cross-Entropy

Para um exemplo com logits z (C classes) e classe correta c:

p_k = e^(z_k) / soma_j e^(z_j)          L = -log(p_c)

Derivada do softmax:

dp_k/dz_j = p_k (1 - p_j) se k = j;   -p_k p_j se k ≠ j

Aplicando a regra da cadeia em L = -log p_c:

dL/dz_j = -(1/p_c) dp_c/dz_j

- se j = c: -(1/p_c) p_c (1 - p_c) = p_c - 1
- se j ≠ c: -(1/p_c)(-p_c p_j) = p_j

Juntando: **dL/dz = p - y**, onde y é o one-hot da classe correta. Com a média no batch, divide-se por N:

**dL/dZ = (P - Y) / N**

Por isso implementamos as duas juntas: separadas, teríamos uma matriz Jacobiana C×C por exemplo; juntas, o resultado é uma subtração.

### Estabilidade numérica

softmax(z) = softmax(z - max(z)), porque e^(z - m) / soma e^(z - m) = e^(-m) e^z / (e^(-m) soma e^z). Subtrair o máximo faz o maior expoente ser 0, e e^0 = 1 nunca estoura.

## 4. Inicialização

Com pesos de variância σ², a variância da saída de uma camada densa é n_in · σ² · Var(x). Para que o sinal não cresça nem suma camada a camada:

- Xavier (tanh, sigmoid): σ² = 1 / n_in
- He (ReLU): σ² = 2 / n_in (o fator 2 compensa a ReLU zerar metade das ativações)

## 5. Otimizadores

- SGD: w ← w - η g
- Momentum: v ← μ v - η g;  w ← w + v (acumula "velocidade" em direções consistentes)
- Adam: m ← β₁m + (1-β₁)g;  v ← β₂v + (1-β₂)g²;  m̂ = m/(1-β₁ᵗ);  v̂ = v/(1-β₂ᵗ);  w ← w - η m̂ / (√v̂ + ε)

Sem a correção de viés, m e v começam em zero e ficam subestimados nos primeiros passos.

## 6. Regularização

- L2: adiciona (λ/2)·||W||² à perda, logo soma λW ao gradiente.
- Dropout invertido: em treino, mantém cada neurônio com probabilidade 1-p e divide por 1-p, para que o valor esperado da ativação seja o mesmo em treino e em avaliação.

## 7. Gradient check

df/dx ≈ [f(x + ε) - f(x - ε)] / (2ε), com erro O(ε²). Use float64 e ε ≈ 1e-5. Erro relativo:

|a - n| / max(1e-8, |a| + |n|)

- < 1e-7: perfeito
- 1e-7 a 1e-5: aceitável para ativações não suaves
- > 1e-4: há bug
