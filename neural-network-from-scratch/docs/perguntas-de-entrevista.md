# Perguntas para testar seu entendimento

Responda sem consultar, em voz alta ou por escrito. Se não conseguir explicar, ainda não domina: volte ao material. São perguntas comuns em entrevistas sobre projetos assim.

## Fundamentos (Fase 1)

1. O que é backpropagation, em uma frase? E por que ele é eficiente comparado ao gradiente numérico?
2. Por que dL/dW = Xᵀ G? Justifique pelos formatos das matrizes.
3. Por que o gradiente de Softmax + Cross-Entropy é tão simples (P - Y)?
4. Por que subtrair o máximo dos logits não altera o softmax?
5. O que acontece se todos os pesos começarem iguais? E se começarem grandes demais?
6. Por que He para ReLU e Xavier para tanh?
7. O que é uma "ReLU morta" e como detectá-la?
8. Qual a diferença entre SGD, Momentum e Adam? Quando o SGD com momentum pode generalizar melhor que o Adam?
9. Por que o Adam precisa de correção de viés?
10. O que o dropout faz em treino e em inferência? Por que dividir por (1 - p)?
11. Para que serve o gradient check e por que usar diferenças centrais?
12. Por que um teste de overfit em poucos exemplos é útil?

## PyTorch (Fase 2)

13. O que faz `optimizer.zero_grad()` e o que acontece se você esquecer?
14. Qual a diferença entre `model.train()` e `model.eval()`? Quais camadas mudam de comportamento?
15. Para que serve `torch.no_grad()`?
16. Por que o teste de equivalência precisou transpor os pesos?
17. Por que `nn.CrossEntropyLoss` recebe logits e não probabilidades?

## CNNs e experimentos (Fase 3)

18. Por que uma CNN funciona melhor que uma MLP em imagens? Fale de compartilhamento de pesos e invariância.
19. Calcule o número de parâmetros de uma Conv2d(32, 64, 3x3).
20. Como o formato muda de 1x28x28 até a camada densa na sua CNN?
21. O que a batch normalization faz e por que acelera o treino?
22. Por que flip horizontal é uma augmentation válida para roupas e não para dígitos?
23. Por que rodar cada experimento com 3 seeds?
24. Por que escolher hiperparâmetros pela validação e não pelo teste?
25. O que o Grad-CAM mostra e quais são suas limitações?
26. Quais classes o seu modelo mais confunde e por quê?

## Engenharia (Fase 4)

27. O que o seu CI verifica e por que isso importa?
28. Como você garante que seus resultados são reproduzíveis?
29. Por que a demo funciona pior com fotos reais? Como você melhoraria isso?
30. Se tivesse mais 2 semanas, o que faria a seguir e por quê?
