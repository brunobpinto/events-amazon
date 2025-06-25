import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Carregar os dados
df = pd.read_csv("resultados/eventos-doencas_amazonia.csv")

# Visualizando os dados
df.head()
df.describe()
print(df.columns)

colunas = [
    'deforestation_soma', 'mining_soma', 't_infecciosas', 't_neoplasias',
    't_sangue', 't_endocrinas', 't_nervoso', 't_olho', 't_ouvido',
    't_cardiovascular', 't_respiratorio', 't_digestivo', 't_pele',
    't_osteomuscular', 't_genitourinario', 't_malformacoes', 't_causas_externas',
    't_influencia', 't_comunicaveis', 't_nao_comunicaveis', 't_malaria',
    't_cancer_mama', 't_cancer_colo_do_utero', 't_srag', 't_total'
]

# Pré-processamento para o Clustering: normalização dos dados
scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(df[colunas])

# Codificação OneHot para colunas categóricas
categorical_features = ['uf']
numerical_features = [ 'ano', 'deforestation_soma', 'mining_soma', 'focos_ativos',
       't_infecciosas', 't_neoplasias', 't_sangue', 't_endocrinas',
       't_nervoso', 't_olho', 't_ouvido', 't_cardiovascular', 't_respiratorio',
       't_digestivo', 't_pele', 't_osteomuscular', 't_genitourinario',
       't_malformacoes', 't_causas_externas', 't_influencia', 't_comunicaveis',
       't_nao_comunicaveis', 't_malaria', 't_cancer_mama',
       't_cancer_colo_do_utero', 't_srag', 't_total']

# Pipeline de pré-processamento
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', MinMaxScaler(), numerical_features)
])

X = preprocessor.fit_transform(df)

# Detectando outliers com Isolation Forest
iso = IsolationForest(contamination=0.05)
yhat = iso.fit_predict(X)

# Remover outliers
mask = yhat != -1
X_clean = X[mask]

## Algoritmo K-MEANS
silhouette_scores_kmeans = []
range_n_clusters = range(2, 10)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_clean)
    score = silhouette_score(X_clean, labels)
    silhouette_scores_kmeans.append(score)

# Gráfico do Silhouette Score
plt.plot(range_n_clusters, silhouette_scores_kmeans, marker='o')
plt.title('KMeans - Índice de Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# K-Means Clustering
silhouette_scores_kmeans = []
range_k = range(2, 10)

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters_kmeans = kmeans.fit_predict(dados_normalizados)
    silhouette_avg = silhouette_score(dados_normalizados, clusters_kmeans)
    silhouette_scores_kmeans.append(silhouette_avg)

# Visualizando o Método do Cotovelo
# METRICA INERCIA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Lista para armazenar a inércia de cada K
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clean)  # X_clean é seu dataset já pré-processado
    inertias.append(kmeans.inertia_)

# Plotando o gráfico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Soma das distâncias quadradas)')
plt.grid(True)
plt.xticks(k_values)
print('\nMetodo do cotovelo utilizando a metrica da inercia')
plt.show()

# Mede a soma das distâncias quadradas entre os pontos dentro de um cluster e o centróide do cluster.
# Quanto menor a inércia, mais compactos os clusters.
# Você visualiza o ponto onde a redução da inércia começa a diminuir drasticamente, indicando o número ideal de clusters

# METRICA INDICE DE SILHUETA
plt.figure(figsize=(9, 6))
plt.plot(range_k, silhouette_scores_kmeans, marker='o')
plt.title("Método do Cotovelo - K-Means", fontsize=10)
plt.xlabel("Número de Clusters (k)", fontsize=10)
plt.ylabel("Índice de Silhueta", fontsize=10)
plt.grid()
print('\nMetodo do cotovelo utilizando a metrica do indice de silhueta')
plt.show()

# Mede a qualidade dos clusters baseando-se na separação entre eles.
# O índice de silhueta varia de -1 a 1: quanto mais próximo de 1, melhor a separação dos clusters.


## Algoritmo DBSCAN
eps_values = [0.3, 0.5, 0.7, 0.9, 1.2]
min_samples = 5
silhouette_scores_dbscan = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_clean)

    # Filtrar ruído
    if len(set(labels)) > 1:
        score = silhouette_score(X_clean, labels)
    else:
        score = -1
    silhouette_scores_dbscan.append(score)

# Gráfico DBSCAN
plt.plot(eps_values, silhouette_scores_dbscan, marker='x', color='green')
plt.title('DBSCAN - Índice de Silhueta')
plt.xlabel('EPS')
plt.ylabel('Silhouette Score')
plt.show()


# DBSCAN Clustering
silhouette_scores_dbscan = []
eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters_dbscan = dbscan.fit_predict(dados_normalizados)

    # Evitar clusters únicos ou não rotulados para cálculo do silhouette
    if len(set(clusters_dbscan)) > 1:
        silhouette_avg = silhouette_score(dados_normalizados, clusters_dbscan)
        silhouette_scores_dbscan.append(silhouette_avg)
    else:
        silhouette_scores_dbscan.append(-1)

# Visualizando os resultados do DBSCAN
plt.figure(figsize=(10, 6))
plt.plot(eps_values, silhouette_scores_dbscan, marker='o', color='orange')
plt.title("Variação do EPS - DBSCAN", fontsize=14)
plt.xlabel("EPS", fontsize=12)
plt.ylabel("Índice de Silhueta", fontsize=12)
plt.grid()
plt.show()


## Resultados
# BASEADO NA INERCIA
# Kmeans
print('Resultados Kmeans:')
for indice, valor in zip(range_n_clusters, silhouette_scores_kmeans):
    print(f"Índice: {indice}, Valor: {valor}")

print('\n')

# DBSCAN
print('Resultados DBSCAN:')
for indice, valor in zip(eps_values, silhouette_scores_dbscan):
    print(f"Índice: {indice}, Valor: {valor}")

# Metodo cotovelo
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Lista para armazenar a inércia de cada K
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clean)  # X_clean é seu dataset já pré-processado
    inertias.append(kmeans.inertia_)


# K-MEANS: Melhor resultado é onde o cotovelo "quebra" no 2
# DBSCAN: Melhor resultado é no 0,9

# Melhor valor de EPS (com base no gráfico do DBSCAN)
# Note: eps_values used here was the second list defined for DBSCAN plotting
eps_otimo = eps_values[np.argmax(silhouette_scores_dbscan)]
dbscan_final = DBSCAN(eps=eps_otimo, min_samples=5)
df['Cluster_DBSCAN'] = dbscan_final.fit_predict(dados_normalizados) # This part uses dados_normalizados, be careful if you intended to use X_clean

# Determine the best K from the KMeans silhouette scores
# Note: range_n_clusters was used for the first KMeans silhouette plot
best_k = range_n_clusters[np.argmax(silhouette_scores_kmeans)]

# Apply KMeans with the best k to the cleaned data and get labels
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10) # Add n_init to avoid warning
# We need to apply these labels back to the original df, aligning with X_clean
# Create a new column for K-Means clusters and initialize with None or a placeholder
df['Cluster_KMeans'] = np.nan

# Fit KMeans on X_clean and get labels for the clean data
labels_kmeans = kmeans_final.fit_predict(X_clean)

# Assign the KMeans labels to the corresponding rows in the original df
# The mask created earlier from Isolation Forest indicates which rows in df correspond to rows in X_clean
df.loc[mask, 'Cluster_KMeans'] = labels_kmeans

# Convert cluster column to string to ensure it's treated as categorical for hue
df['Cluster_KMeans'] = df['Cluster_KMeans'].astype(str)
df['Cluster_DBSCAN'] = df['Cluster_DBSCAN'].astype(str)


# Visualizando os Clusters
# Ensure the columns for plotting exist in df
plot_vars = ['deforestation_soma', 'mining_soma', 't_total']
if all(col in df.columns for col in plot_vars + ['Cluster_KMeans']):
    sns.pairplot(df.dropna(subset=['Cluster_KMeans']), vars=plot_vars, hue='Cluster_KMeans', palette='Set2') # Drop rows with NaN in Cluster_KMeans for plotting
    plt.title("Clusters - K-Means", fontsize=16)
    plt.show()
else:
    print("One or more columns for K-Means pairplot not found in DataFrame.")

if all(col in df.columns for col in plot_vars + ['Cluster_DBSCAN']):
    sns.pairplot(df.dropna(subset=['Cluster_DBSCAN']), vars=plot_vars, hue='Cluster_DBSCAN', palette='Set1') # Drop rows with NaN in Cluster_DBSCAN for plotting
    plt.title("Clusters - DBSCAN", fontsize=16)
    plt.show()
else:
     print("One or more columns for DBSCAN pairplot not found in DataFrame.")


# Análise Final
# Need k_otimo from the previous KMeans section, assuming it was determined there
# If not, use best_k calculated above
print("Número de Clusters K-Means:", best_k) # Using best_k calculated here
print("Número de Clusters DBSCAN:", len(set(df['Cluster_DBSCAN'].dropna())) - (1 if '-1.0' in df['Cluster_DBSCAN'].astype(str).unique() else 0)) # Account for '-1' as string after conversion


# # Melhor valor de EPS (com base no gráfico do DBSCAN)
# eps_otimo = eps_values[np.argmax(silhouette_scores_dbscan)]
# dbscan_final = DBSCAN(eps=eps_otimo, min_samples=5)
# df['Cluster_DBSCAN'] = dbscan_final.fit_predict(dados_normalizados)

# # Visualizando os Clusters
# sns.pairplot(df, vars=['deforestation_soma', 'mining_soma', 't_total'], hue='Cluster_KMeans', palette='Set2')
# plt.title("Clusters - K-Means", fontsize=16)
# plt.show()

# sns.pairplot(df, vars=['deforestation_soma', 'mining_soma', 't_total'], hue='Cluster_DBSCAN', palette='Set1')
# plt.title("Clusters - DBSCAN", fontsize=16)
# plt.show()

# # Análise Final
# print("Número de Clusters K-Means:", k_otimo)
# print("Número de Clusters DBSCAN:", len(set(df['Cluster_DBSCAN'])) - (1 if -1 in df['Cluster_DBSCAN'] else 0))

# Encontrando o melhor clustering
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X_clean)

# KMeans com melhor cluster
best_k = range_n_clusters[np.argmax(silhouette_scores_kmeans)]
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels_kmeans = kmeans.fit_predict(X_clean)

plt.figure(figsize=(8, 5))
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels_kmeans, cmap='viridis')
plt.title(f'KMeans com {best_k} clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# Dendograma
# Usar apenas duas features para simplificar visualização do dendrograma
sample_df = df[[ 'deforestation_soma', 'mining_soma', 'focos_ativos',
       't_infecciosas', 't_neoplasias', 't_sangue', 't_endocrinas',
       't_nervoso', 't_olho', 't_ouvido', 't_cardiovascular', 't_respiratorio',
       't_digestivo', 't_pele', 't_osteomuscular', 't_genitourinario',
       't_malformacoes', 't_causas_externas', 't_influencia', 't_comunicaveis',
       't_nao_comunicaveis', 't_malaria', 't_cancer_mama',
       't_cancer_colo_do_utero', 't_srag', 't_total']]
scaled_sample = MinMaxScaler().fit_transform(sample_df)

Z = sch.linkage(scaled_sample, method='ward')

plt.figure(figsize=(12, 6))
sch.dendrogram(Z, p=12, truncate_mode='level')
plt.title('Dendrograma - Cluster Hierárquico')
plt.xlabel('doencas')
plt.ylabel('Distância')
plt.show()

Z = sch.linkage(df[['deforestation_soma', 'mining_soma', 'focos_ativos',
       't_infecciosas', 't_neoplasias', 't_sangue', 't_endocrinas',
       't_nervoso', 't_olho', 't_ouvido', 't_cardiovascular', 't_respiratorio',
       't_digestivo', 't_pele', 't_osteomuscular', 't_genitourinario',
       't_malformacoes', 't_causas_externas', 't_influencia', 't_comunicaveis',
       't_nao_comunicaveis', 't_malaria', 't_cancer_mama',
       't_cancer_colo_do_utero', 't_srag', 't_total']], method='centroid')

# p = altura máxima que será exibida
sch.dendrogram(Z, p = 6, truncate_mode = "level")


# Analise do Clustering
# Reatribuir cluster ao DataFrame original (sem outliers)
df_clean = df[mask].copy()
df_clean['cluster_kmeans'] = labels_kmeans

# Visualizar 3 itens de cada cluster
for cluster in sorted(df_clean['cluster_kmeans'].unique()):
    print(f"\nCluster {cluster}")
    print(df_clean[df_clean['cluster_kmeans'] == cluster].sample(10))
