import pandas as pd
from pycaret.classification import setup, compare_models, pull

# 1. Carregar os dados
df = pd.read_csv('dados_saude_predicao.csv')

# 2. Features e variável alvo
features = ['Idade', 'Pressao_Arterial', 'Colesterol_Total', 'Frequencia_Cardiaca_Max']
target = 'Risco_Internacao'

df = df[features + [target]]

# 3. Configurar o ambiente do PyCaret (faz o split treino/teste e a normalização automaticamente)
clf = setup(data=df, target=target, session_id=42, train_size=0.7, verbose=False)

# 4. Comparar todos os modelos automaticamente
melhores_modelos = compare_models()

# 5. Pegar a tabela de comparação gerada pelo PyCaret
resultados_pycaret = pull()
print("\nTabela comparativa do PyCaret (Problema 2):")
print(resultados_pycaret)

# 6. Salvar em CSV
resultados_pycaret.to_csv('resultados_pycaret_problema2.csv', index=False)
print("\nResultados exportados para 'resultados_pycaret_problema2.csv' com sucesso!")
