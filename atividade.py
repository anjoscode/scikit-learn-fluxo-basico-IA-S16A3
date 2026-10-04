from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Etapa 1 — Preparação dos dados
# X representa as características dos alunos: [horas de estudo, número de faltas]
# y representa os resultados: 0 para reprovado e 1 para aprovado
X = [
    [2, 1],
    [4, 0],
    [1, 3],
    [3, 1],
    [5, 0],
    [2, 2]
]

y = [0, 1, 0, 1, 1, 0]


# Etapa 2 — Separação em treino e teste
# Não devemos treinar e testar com os mesmos dados para garantir que o modelo
# realmente aprendeu os padrões, em vez de apenas decorá-los.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Etapa 3 — Criação do modelo
# Escolhi a Regressão Logística por ser um modelo simples e eficiente para classificação binária (0 ou 1).
# Treinar um modelo significa ajustar o algoritmo para que ele entenda a relação entre X e y.
model = LogisticRegression()


# Etapa 4 — Treinamento
# Os dados de treino servem como a base de aprendizado para o modelo descobrir as regras do problema.
model.fit(X_train, y_train)


# Etapa 5 — Avaliação
# Gerando previsões nos dados de teste e calculando a acurácia
y_pred = model.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

# A acurácia indica a porcentagem de acertos que o modelo teve no conjunto de teste.
print(f"Previsões: {y_pred}")
print(f"Rótulos reais: {y_test}")
print(f"Acurácia: {acuracia * 100:.2f}%")