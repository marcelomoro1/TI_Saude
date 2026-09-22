import pandas as pd
from pycaret.classification import setup, compare_models, pull

# 1. Carregar os dados
df = pd.read_csv('dados_predicao_modelos.csv')

# 2. Features e variável alvo
features = ['Idade', 'Renda_Anual_K', 'Score_Credito', 'Pontuacao_Engajamento']
target = 'Compro_Produto'

df = df[features + [target]]

# 3. Configurar o ambiente do PyCaret (faz o split treino/teste e a normalização automaticamente)
clf = setup(data=df, target=target, session_id=42, train_size=0.7, verbose=False)

# 4. Comparar todos os modelos automaticamente
melhores_modelos = compare_models()

# 5. Pegar a tabela de comparação gerada pelo PyCaret
resultados_pycaret = pull()
print("\nTabela comparativa do PyCaret (Problema 1):")
print(resultados_pycaret)

# 6. Salvar em CSV
resultados_pycaret.to_csv('resultados_pycaret_problema1.csv', index=False)
print("\nResultados exportados para 'resultados_pycaret_problema1.csv' com sucesso!")
