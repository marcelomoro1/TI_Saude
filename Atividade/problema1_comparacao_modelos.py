import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Modelos de classificação
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1. Carregar os dados
df = pd.read_csv('dados_predicao_modelos.csv')

# 2. Features e variável alvo
features = ['Idade', 'Renda_Anual_K', 'Score_Credito', 'Pontuacao_Engajamento']
target = 'Compro_Produto'

X = df[features]
y = df[target]

# 3. Divisão treino/teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Padronizar (necessário para SVM e KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Modelos de classificação
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# 6. Avaliar cada modelo (treino e teste, para analisar overfitting)
resultados = []

print("Avaliação dos Modelos:\n")

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train, average='macro', zero_division=0)

    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

    resultados.append((nome, acc_train, f1_train, acc_test, f1_test))

    print(f"Modelo: {nome}")
    print(f"Acurácia (Treino): {acc_train:.4f} | Acurácia (Teste): {acc_test:.4f}")
    print(f"F1-Score Macro (Treino): {f1_train:.4f} | F1-Score Macro (Teste): {f1_test:.4f}")
    print("Matriz de Confusão (Treino):")
    print(confusion_matrix(y_train, y_pred_train))
    print("Relatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print("Matriz de Confusão (Teste):")
    print(confusion_matrix(y_test, y_pred_test))
    print("-" * 60)

# 7. Mostrar ranking final por F1-Score Macro (Teste)
resultados.sort(key=lambda x: x[4], reverse=True)

print("\nRanking Final dos Modelos (por F1-Score Macro no Teste):")
print(f"{'Modelo':<22} {'Acc Treino':<12} {'F1 Treino':<12} {'Acc Teste':<12} {'F1 Teste':<12}")
print("-" * 70)
for nome, acc_train, f1_train, acc_test, f1_test in resultados:
    print(f"{nome:<22} {acc_train:<12.4f} {f1_train:<12.4f} {acc_test:<12.4f} {f1_test:<12.4f}")

# 8. Criar DataFrame com os resultados
df_resultados = pd.DataFrame(
    resultados,
    columns=['Modelo', 'Acuracia_Treino', 'F1_Score_Macro_Treino', 'Acuracia_Teste', 'F1_Score_Macro_Teste']
)

# 9. Salvar em CSV
df_resultados.to_csv('resultados_modelos_classificacao.csv', index=False)
print("\nResultados exportados para 'resultados_modelos_classificacao.csv' com sucesso!")

# 10. Plotar gráfico comparativo (Teste)
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(df_resultados))

# Barras de acurácia (teste)
plt.bar(index, df_resultados['Acuracia_Teste'], bar_width, label='Acurácia (Teste)', color='skyblue')

# Barras de F1-score macro (teste) ao lado
plt.bar(index + bar_width, df_resultados['F1_Score_Macro_Teste'], bar_width, label='F1-Score Macro (Teste)', color='orange')

plt.xlabel('Modelo')
plt.ylabel('Pontuação')
plt.title('Comparação de Modelos - Acurácia vs F1-Score (Macro) no Teste')
plt.xticks(index + bar_width / 2, df_resultados['Modelo'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Salvar e mostrar gráfico
plt.savefig('grafico_comparacao_modelos.png')
plt.show()
