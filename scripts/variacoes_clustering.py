import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Ajuste este caminho para o seu arquivo
data_path = 'resultados/eventos-doencas_amazonia.csv'

# 1) Carregar dados e one-hot em 'uf'
df_raw = pd.read_csv(data_path)
ohe = OneHotEncoder(drop='first', sparse_output=False)
uf_ohe = pd.DataFrame(
    ohe.fit_transform(df_raw[['uf']]),
    columns=ohe.get_feature_names_out(['uf']),
    index=df_raw.index
)
df = pd.concat([df_raw.drop(columns=['uf']), uf_ohe], axis=1)

# Colunas numéricas para clustering
def_cols = [
    'ano','deforestation_soma','mining_soma','focos_ativos',
    't_infecciosas','t_neoplasias','t_sangue','t_endocrinas',
    't_nervoso','t_olho','t_ouvido','t_cardiovascular',
    't_respiratorio','t_digestivo','t_pele','t_osteomuscular',
    't_genitourinario','t_malformacoes','t_causas_externas',
    't_influencia','t_comunicaveis','t_nao_comunicaveis',
    't_malaria','t_cancer_mama','t_cancer_colo_do_utero',
    't_srag','t_total'
]

# 2) Pré-processamentos (8 combinações)
preps = [
    {'normalizacao':'x','padronizacao':'','media':'x','remocao':'','pca':'x'},
    {'normalizacao':'x','padronizacao':'','media':'x','remocao':'','pca':''},
    {'normalizacao':'x','padronizacao':'','media':'','remocao':'x','pca':'x'},
    {'normalizacao':'x','padronizacao':'','media':'','remocao':'x','pca':''},
    {'normalizacao':'','padronizacao':'x','media':'x','remocao':'','pca':'x'},
    {'normalizacao':'','padronizacao':'x','media':'x','remocao':'','pca':''},
    {'normalizacao':'','padronizacao':'x','media':'','remocao':'x','pca':'x'},
    {'normalizacao':'','padronizacao':'x','media':'','remocao':'x','pca':''},
]

# 3) Mineração (18 combinações por bloco)
mining = []
for algo in ['kmeans','dbscan','hierarquico']:
    for clusters in [2,5,8]:
        for ms in [5,10]:
            mining.append({
                'kmeans':'x' if algo=='kmeans' else '',
                'dbscan':'x' if algo=='dbscan' else '',
                'hierarquico':'x' if algo=='hierarquico' else '',
                '2clusters':'x' if clusters==2 else '',
                '5clusters':'x' if clusters==5 else '',
                '8clusters':'x' if clusters==8 else '',
                '5samples':'x' if ms==5 else '',
                '10samples':'x' if ms==10 else ''
            })

# 4) Montagem da tabela e cálculo das métricas
columns = [
    'Cenario','normalizacao','padronizacao','media','remocao','pca',
    'kmeans','dbscan','hierarquico','2clusters','5clusters','8clusters',
    '5samples','10samples','Silhouette_Score','Calinski-Harabasz_Index','Davies-Bouldin_Index'
]
rows = []
cenario = 1

for prep in preps:
    for mine in mining:
        # Pré-processar cópia
        df_p = df.copy()
        if prep['media']=='x':
            df_p[def_cols] = SimpleImputer(strategy='mean').fit_transform(df_p[def_cols])
        else:
            df_p = df_p.dropna(subset=def_cols)
        if prep['normalizacao']=='x':
            df_p[def_cols] = MinMaxScaler().fit_transform(df_p[def_cols])
        else:
            df_p[def_cols] = StandardScaler().fit_transform(df_p[def_cols])
        if prep['pca']=='x':
            comps = PCA(n_components=0.95, random_state=42).fit_transform(df_p[def_cols])
            pc_cols = [f'PC{i+1}' for i in range(comps.shape[1])]
            df_p = pd.concat([
                pd.DataFrame(comps, columns=pc_cols, index=df_p.index),
                df_p.drop(columns=def_cols)
            ], axis=1)
            features = pc_cols + [c for c in df_p.columns if c.startswith('uf_')]
        else:
            features = def_cols + [c for c in df_p.columns if c.startswith('uf_')]

        # Aplicar algoritmo e obter labels
        if mine['kmeans']=='x':
            n = 2 if mine['2clusters']=='x' else 5 if mine['5clusters']=='x' else 8
            labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(df_p[features])
        elif mine['dbscan']=='x':
            ms = 5 if mine['5samples']=='x' else 10
            labels = DBSCAN(eps=0.5, min_samples=ms).fit_predict(df_p[features])
        else:
            n = 2 if mine['2clusters']=='x' else 5 if mine['5clusters']=='x' else 8
            labels = AgglomerativeClustering(n_clusters=n).fit_predict(df_p[features])

        # Cálculo das métricas: incluir ruído como cluster para DBSCAN
        if len(set(labels))>1:
            sil = silhouette_score(df_p[features], labels)
            ch  = calinski_harabasz_score(df_p[features], labels)
            db  = davies_bouldin_score(df_p[features], labels)
        else:
            sil = ch = db = np.nan

        # Construir linha
        row = {'Cenario':cenario}
        row.update(prep)
        row.update(mine)
        row.update({'Silhouette_Score':sil,
                    'Calinski-Harabasz_Index':ch,
                    'Davies-Bouldin_Index':db})
        rows.append(row)
        cenario += 1

# 5) Criar DataFrame e salvar
import pandas as pd

df_res = pd.DataFrame(rows, columns=columns)
df_res.to_csv('resultados/experimentos_cluster_completo.csv', index=False)
df_res.to_excel('resultados/experimentos_cluster_completo.xlsx', index=False)

print("Tabela completa (144 cenários) gerada em CSV e XLSX.")