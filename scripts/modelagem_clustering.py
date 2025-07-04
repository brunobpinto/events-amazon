import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# 1) Carregar dados
path = 'resultados/eventos-doencas_amazonia.csv'
df = pd.read_csv(path)
print(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas")

# 2) Definir colunas para análise
numerical_feats = [
    'ano', 'deforestation_soma', 'mining_soma', 'focos_ativos',
    't_infecciosas', 't_neoplasias', 't_sangue', 't_endocrinas',
    't_nervoso', 't_olho', 't_ouvido', 't_cardiovascular',
    't_respiratorio', 't_digestivo', 't_pele', 't_osteomuscular',
    't_genitourinario', 't_malformacoes', 't_causas_externas',
    't_influencia', 't_comunicaveis', 't_nao_comunicaveis',
    't_malaria', 't_cancer_mama', 't_cancer_colo_do_utero',
    't_srag', 't_total'
]
categorical_feats = ['uf']

# 3) Pré-processamento
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])
preprocessor = ColumnTransformer([
    ('nums', num_pipe, numerical_feats),
    ('cats', cat_pipe, categorical_feats)
])

X = preprocessor.fit_transform(df)
print("Pré-processamento concluído")

# 4) Remover outliers
iso = IsolationForest(contamination=0.05, random_state=42)
mask = iso.fit_predict(X) != -1
X_clean = X[mask]
df_clean = df.loc[mask].reset_index(drop=True)
print(f"Outliers removidos: {np.sum(~mask)} registros")

# 5) Avaliar K-Means
k_values = list(range(2, 11))
inertias, sils = [], []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clean)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_clean, labels) if len(set(labels)) > 1 else np.nan)

# Visualizar método do cotovelo e silhueta
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, 'o-', label='Inércia')
plt.title('Cotovelo - KMeans')
plt.xlabel('k'); plt.ylabel('Inércia'); plt.grid(); plt.show()

# Definir melhor k segundo o Cotovelo
drops = np.diff(inertias)
best_k = k_values[int(np.argmax(drops)) + 1]
print(f"Melhor k segundo o Cotovelo: {best_k}")

plt.figure(figsize=(8, 4))
plt.plot(k_values, sils, 'o-', label='Silhueta')
plt.title('Silhouette - KMeans')
plt.xlabel('k'); plt.ylabel('Score'); plt.grid(); plt.show()

best_k = k_values[int(np.nanargmax(sils))]
print(f"Melhor k segundo Silhouette: {best_k}")

# 6) Avaliar DBSCAN
eps_list = np.linspace(0.2, 1.0, 9)
db_sils = []
for eps in eps_list:
    db = DBSCAN(eps=eps, min_samples=5)
    lbl = db.fit_predict(X_clean)
    score = silhouette_score(X_clean, lbl) if len(set(lbl)) > 1 and -1 not in set(lbl) else np.nan
    db_sils.append(score)

plt.figure(figsize=(8, 4))
plt.plot(eps_list, db_sils, 'o-', label='Silhueta DBSCAN')
plt.title('Silhouette - DBSCAN')
plt.xlabel('eps'); plt.ylabel('Score'); plt.grid(); plt.show()

best_eps = eps_list[int(np.nanargmax(db_sils))]
print(f"Melhor eps segundo Silhouette: {best_eps:.2f}")

# 7) Aplicar clusters finais
df_clean['Cluster_KMeans'] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_clean)
df_clean['Cluster_DBSCAN'] = DBSCAN(eps=best_eps, min_samples=5).fit_predict(X_clean)

# 8) Visualização com PCA
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_clean)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=df_clean['Cluster_KMeans'].astype(str), palette='tab10')
plt.title(f'KMeans (k={best_k})'); plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=df_clean['Cluster_DBSCAN'].astype(str), palette='tab10')
plt.title(f'DBSCAN (eps={best_eps:.2f})'); plt.show()

# 9) Dendrograma Hierárquico
link = sch.linkage(X_clean, method='ward')
sch.dendrogram(link, truncate_mode='level', p=12)
plt.title('Dendrograma Hierárquico'); plt.ylabel('Distância'); plt.show()

# 10) Salvar resultados
out_path = 'resultados/clusters_amazonia.csv'
df_clean.to_csv(out_path, index=False)
print(f"Clusters salvos em: {out_path}")