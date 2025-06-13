import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import os

# Criar pasta para salvar os gráficos
os.makedirs("graficos", exist_ok=True)

# Redirecionar prints para arquivo
sys.stdout = open("analise/pre-processamento/analise_exploratoria.txt", "w", encoding="utf-8")

# ========================
# CARREGAMENTO E ESTRUTURA
# ========================
df = pd.read_csv("resultados/eventos-doencas_amazonia.csv")

print("\n=== Estrutura dos Dados ===")
print(df.info())

print("\n=== Primeiras Linhas ===")
print(df.head())

# ========================
# NORMALIZAÇÃO DAS VARIÁVEIS AMBIENTAIS
# ========================
scaler = MinMaxScaler()
variaveis_ambiente = ['deforestation_soma', 'mining_soma', 'focos_ativos']
df[variaveis_ambiente] = scaler.fit_transform(df[variaveis_ambiente])

print("\n=== Colunas Normalizadas ===")
print(df[variaveis_ambiente].head())

# ========================
# VALORES IGUAIS A ZERO
# ========================
print("\n=== Valores Iguais a Zero ===")
valores_zero = df[df == 0].sum()
print(valores_zero[valores_zero > 0])

# ========================
# ESTATÍSTICAS DESCRITIVAS
# ========================
colunas_numericas = df.select_dtypes(include='number').drop(columns=['cod_municipio', 'ano'], errors='ignore')
print("\n=== Resumo Estatístico ===")
print(colunas_numericas.describe())

# ========================
# CORRELAÇÃO: variáveis ambientais x doenças
# ========================
causas = [col for col in df.columns if col.startswith('t_')]
df_corr = df[causas + variaveis_ambiente].corr()

# ========================
# MATRIZ DE CORRELAÇÃO
# ========================
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.loc[causas, variaveis_ambiente], annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlação entre Variáveis Ambientais e Causas de Morte")
plt.tight_layout()
plt.savefig("analise/pre-processamento/heatmap_correlacao_geral.png")
plt.show()

# ========================
# TABELA COMPLETA DE CORRELAÇÃO
# ========================
print("\n=== Tabela de Correlação Completa ===")
print(df_corr[variaveis_ambiente].loc[causas].sort_values(by='deforestation_soma', ascending=False))

# ========================
# HISTOGRAMAS com limite no eixo X
# ========================
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
axs = axs.flatten()

causas_destaque = [
    't_osteomuscular', 't_malformacoes', 't_respiratorio',
    't_infecciosas', 't_comunicaveis', 't_causas_externas', 't_neoplasias'
]

for idx, causa in enumerate(causas_destaque):
    data = df[causa].dropna()
    upper = data.quantile(0.99)
    axs[idx].hist(data[data <= upper], bins=30, alpha=0.7)
    axs[idx].set_title(f'Histograma – {causa}')
    axs[idx].set_xlabel(causa)
    axs[idx].set_ylabel('Frequência')
    axs[idx].grid(True)

for j in range(len(causas_destaque), len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig("analise/pre-processamento/histogramas_causas.png")
plt.show()

# ========================
# DISPERSÕES: causas vs. eventos
# ========================
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(x='deforestation_soma', y='t_osteomuscular', data=df)
plt.title('Desmatamento vs. Osteomuscular')
plt.grid(True)

plt.subplot(2, 2, 2)
sns.scatterplot(x='mining_soma', y='t_respiratorio', data=df)
plt.title('Mineração vs. Respiratório')
plt.grid(True)

plt.subplot(2, 2, 3)
sns.scatterplot(x='focos_ativos', y='t_infecciosas', data=df)
plt.title('Focos Ativos vs. Infecciosas')
plt.grid(True)

plt.subplot(2, 2, 4)
sns.scatterplot(x='focos_ativos', y='t_comunicaveis', data=df)
plt.title('Focos Ativos vs. Comunicáveis')
plt.grid(True)

plt.tight_layout()
plt.savefig("analise/pre-processamento/dispersion_causas_eventos.png")
plt.show()

# ========================
# OUTLIERS (IQR)
# ========================
def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((series < lower) | (series > upper)).sum()

outlier_counts = {
    causa: count_outliers(df[causa].dropna()) for causa in causas_destaque
}
outliers = pd.Series(outlier_counts, name='outlier_count')

print("\n=== Outliers Detectados ===")
print(outliers)

plt.figure(figsize=(12, 6))
df[causas_destaque].boxplot(vert=False)
plt.title("Boxplots – Causas de Morte com Alta Correlação")
plt.tight_layout()
plt.savefig("analise/pre-processamento/boxplots_outliers.png")
plt.show()

# ========================
# VISUALIZAÇÕES CLARAS DE CORRELAÇÃO
# ========================
from numpy import abs as np_abs

# Seleciona causas com |correlação| > 0.75
causas_top = df_corr[variaveis_ambiente].loc[lambda d: np_abs(d).max(axis=1) > 0.75].index.tolist()
causas_top = [c for c in causas_top if c not in variaveis_ambiente]

# GRÁFICO DE BARRAS – Top correlações por variável ambiental
for var in variaveis_ambiente:
    cor = df_corr[var].drop(variaveis_ambiente).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cor.values, y=cor.index, hue=cor.index, palette='Reds_r', legend=False)
    plt.title(f"Top Correlações com {var}")
    plt.xlabel("Coeficiente de Correlação")
    plt.tight_layout()
    plt.savefig(f"analise/pre-processamento/top_correlacoes_{var}.png")
    plt.show()

# REGPLOTS – Visualização com linha de tendência
plt.figure(figsize=(14, 10))
graficos = [
    ('deforestation_soma', 't_osteomuscular'),
    ('mining_soma', 't_respiratorio'),
    ('focos_ativos', 't_infecciosas'),
    ('focos_ativos', 't_comunicaveis'),
]

for i, (x, y) in enumerate(graficos, start=1):
    plt.subplot(2, 2, i)
    sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha': 0.4, 's': 20}, line_kws={"color": "red"})
    plt.title(f'{x} vs. {y}')
    plt.grid(True)

plt.tight_layout()
plt.savefig("analise/pre-processamento/regplot_causas_vs_ambiente_simplificado.png")
plt.show()

# Finaliza log
sys.stdout.close()
sys.stdout = sys.__stdout__