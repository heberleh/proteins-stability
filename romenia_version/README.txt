
Essa pasta representa os códigos que foram feitos
anterioremente para um suposto caso mais geral e
que eu parei de codidificar.

Aqui, estamos considerando sempre 3 classes,
do conjunto de dados da Romênia, formado por
28 amostras. No treino e teste independentes
separamos 22 e 6 amostras, respectivamente.

Aqui a ideia é:
1. Verificar a crossvalidação de vários classificadores
usando o conjunto com 28 amostras e principalmente o
conjunto de 22 amostras;

2. Verificar o comportamento desses modelos de 22 amostras
testando-os com 6 amostras independentes. Porém,
sem usar o resultado nas decisões dos processos seguintes.

3. Criar métodos de seleção de proteínas, armazenando
a frequência delas em sub-selecionadores. Isto é,
um Leave-One-Out será o selecionador geral e os 
modelos internos, que realizam combinações ou cálculos
de ranques serão os sub-selecionadores. Entre eles estão:
Kruskal (3 classes), SVM-RFE, NSC, e geração de combinações.

4. Os sub-selecionadores utilizam modelos de avaliação.
Em princípio foi escolhido a Decision Tree por conta dos
resultados da crossvalidação a qual o treino (22) foi submetido.
Além de também haver razões biológicas, como as apresentadas
no artigo do Ideker.