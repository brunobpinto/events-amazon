import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carrega os dados
df_datasus = pd.read_csv("resultados/datasus_amazonia.csv")
df_eventos = pd.read_csv("resultados/eventos_amazonia.csv")

# Juntar pelas colunas 'uf' e 'ano'
df_final = df_eventos.merge(df_datasus, on=['uf', 'ano'], how='left')

# Verificar resultado
print(df_final.shape)
print(df_final.columns)
print(df_final.head())

# Salvar em CSV
df_final.to_csv("resultados/eventos-doencas_amazonia.csv", index=False)