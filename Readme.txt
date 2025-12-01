A biblioteca sklearn espera matrizes para a variavel X: [[1],[2],[3],[4]], para a biblioteca cada linha é uma amostra e cada coluna
é uma feature (caracteristica). No caso de arrays 1D, nao é possivel saber se sao 1 amostra e varias features ou se sao varias amostras com 
somente uma feature

X = [
    [1,1.5,1.8],
    [2,2.5,2.8],
    [3,3.5,3.8],
    [4,4.5,4.8],
    [5,5.5,5.8],
]

    RESUMO CAPITULO 2

🔹 Métodos

    •fit() → Aprende com os dados (ajusta o modelo).
    Exemplo: calcula média, desvio padrão, máximos e mínimos no caso de normalização, ou ajusta coeficientes no caso de regressão.

    •transform() → Aplica uma transformação nos dados já “aprendida” pelo fit() (ex: normalização, codificação, etc.).
    Exemplo: StandardScaler, OneHotEncoder.

    •fit_transform() → Combina fit e transform (ajusta e transforma ao mesmo tempo, útil no pré-processamento).

    •predict() → Realiza previsões com base no modelo treinado.

    •Pipelines (sklearn.pipeline.Pipeline) →
    Permitem encadear várias etapas (pré-processamento + modelo) em uma única estrutura, garantindo que tudo ocorra na ordem
    correta durante o treino e teste.
    Eexemplo:
            model = make_pipeline(poly, std_scaler, reg_linear)
            model.fit(X,y)
            O pipeline funciona da seguinte forma, ao chamar o .fit() em cada etapa dos pipelines é chamado o metodo fit_transform exceto no
            ultimo metodo no qual é chamado apenas fit()

Apenas os transformadores possuem os metodos transform. Metodos preditivos possuem predict, score e fit

⚙️ Pré-processamento (sklearn.preprocessing)

    •OneHotEncoder → Lida com variáveis categóricas, convertendo-as em vetores binários (0 e 1).
    Exemplo: "vermelho", "verde", "azul" → [1,0,0], [0,1,0], [0,0,1].

    •StandardScaler → Normaliza os dados com base na média e desvio padrão.
    ⚠️ Sensível a outliers.

    •MinMaxScaler → Redimensiona os dados para um intervalo definido (geralmente entre 0 e 1).
    Menos sensível a outliers que o StandardScaler.

    SimpleImputer: vem do sklearn.impute, metodo utilizado para tratar valores NAN

Métricas (sklearn.metrics)

Usadas para avaliar o desempenho de modelos.

    •mean_squared_error (MSE) → mede o erro quadrático médio.

    •root_mean_squared_error (RMSE) → raiz quadrada do MSE (mesma unidade da variável de saída).

    •r2_score → mede o quão bem o modelo explica a variabilidade dos dados.

    •mean_absolute_error (MAE) → média dos erros absolutos (menos sensível a outliers que o MSE).

    •precision_score, recall_score, f1_score, precision_recall_curve

🤖 Modelos

    •from sklearn.linear_model import LinearRegression
    Modelo de regressão linear — útil quando há uma relação linear entre as variáveis.
    Suporta regressão múltipla (vários parâmetros).

    •from sklearn.tree import DecisionTreeRegressor, plot_tree
    Modelo de árvore de decisão — divide os dados em ramos baseados em regras simples.
    plot_tree() → visualiza a estrutura da árvore.

    •from sklearn.model_selection import cross_val_score
    Realiza validação cruzada, testando o modelo em diferentes partições dos dados para medir desempenho de forma mais confiável.

    •from sklearn.ensemble import RandomForestRegressor
    Modelo de floresta aleatória, combina várias árvores de decisão com diferentes amostras e parâmetros.
    Mais robusto, porém mais pesado computacionalmente.


### Capitulo 3

• Analise de imagens de numeros com 700 pixels, e foram dividos em matrizes de 28x28

• O estudo se iniciou tentando detectar numeros 5, porem devido ao desbalanceamento gerou-se um modelo pouco confiavel, que aprendeu a detectar valores diferentes de 5. 

• sklearn.datasets é um utilitario para buscar datasets

• SGDClassifier: metodo de classificação aleatoria em lotes, util para analises de altas dimensoes. Mais impreciso que metodos 
diretos, como Regressao Linear, porem é mais rapido.

• Metricas de classificação:
    - Acuracia: Acertos em relação ao total de amostras (Verdadeiros Positivos), ou seja, acertos sao falar que 1 não é 5 e que 5 é 5
    - Precisão: Entre os Verdadeiros Positivos e Falsos Positivos, qual a taxa de Verdadeiros Positivos. Entre oque passou, o quanto esta correto ?
    - Recall: Entre os Verdadeiros Positivos e Falsos Negativos, qual a taxa de filtragem de Verdadeiros Positivos. Entre todos os positivos, acertou quanto ?

    Essas metricas, Acuracia, Precisão e Recall, podem ser obtidas atraves da matriz de confusão: from sklearn.metrics import confusion_matrix
    Ou mais diretamente: from sklearn.metrics import precision_score, recall_score

    - F1 Score: media harmonica entre precisao e recall, para ter um valor alto ambos devem ser altos.
        Pode ser obtido por from sklearn.metrics import f1_score
    
    - Curva Precisao VS Recall, utilizada para verificar os thresholds, limiares. É possivel ver graficamente como eles se influenciam.

• RandomForestRegressor: regressor usado para melhor aproximação dos dados. Utiliza um conjunto de arvores com parametros aleatorios para fazer as previsoes, 
o resultado final é a media de cada arvore. Porem é mais lento, mas evita overfitting.

• Ao final notou-se uma grande taxa de erros entre alguns numeros, por exemplo alta taxa de erro na decisao de 5 igual a 8. Isso pode se dar
devido a mal escrita dos valores, possivelmente um humano teria dificuldade em entender o numero escrito.

• Uma tatica para aumentar o dataset é gerar novas imagens deslocadas alguns pixels, assim temos mais dados para treino e isso pode diminuir os erros.