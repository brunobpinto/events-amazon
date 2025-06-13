import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Criando o dataframe e juntando as tres partes
df1 = pd.read_csv('dados/datasus_2006_2010.csv')
df2 = pd.read_csv('dados/datasus_2011_2015.csv')
df3 = pd.read_csv('dados/datasus_2016_2019.csv')

df = pd.concat([df1, df2, df3], ignore_index=True)

# Localizando dados faltantes
df.isna().sum()

# Limpeza de informação
df=df.dropna(subset=['code_muni'])
df=df[df['legal_amazon'] == 1.0]
df.isna().sum()

df.drop(['code_muni_6','t_covid_identificada','t_covid_nao_identificada',
    't_covid_historico','t_pos_covid','t_inflamacao_covid','t_cigarro_eletronico',
    't_aids','t_mentais','t_gravidez','t_perinatal','t_lesoes','t_especial',
    't_anormais','t_diabetes','t_autoprovocadas','t_transporte','t_agressoes',
    't_acidentes','t_pedestre', 't_ciclista', 't_motociclista'],
    axis=1, inplace=True)

# Atualizando o total de obitos
colunas_para_somar = [ 't_infecciosas', 't_neoplasias', 't_sangue',
    't_endocrinas',    't_nervoso', 't_olho', 't_ouvido',
    't_cardiovascular', 't_respiratorio',    't_digestivo', 't_pele',
    't_osteomuscular', 't_genitourinario',    't_malformacoes',
    't_causas_externas', 't_influencia']

df['t_total_atualizada'] = df[colunas_para_somar].sum(axis=1)

# Verificar resultado final
print(df.shape)
print(df.columns)
print(df.head())

# Carregando os arquivos
df_datasus = df

# Padronizar o nome das colunas em comum de todas as tabelas
# Função para padronizar nomes de colunas comuns
def padronizar_colunas(df):
    df = df.rename(columns={
        'name_muni': 'municipio',
        'abbrev_state': 'uf',
        'code_muni': 'cod_municipio',
        'dtobito' : 'ano'
    })
    return df

# Aplicar padronização a todos os DataFrames
df_datasus = padronizar_colunas(df_datasus)

# Filtrar dados necessarios (Transformar data em ano)
# Remover colunas específicas de cada DataFrame
df_datasus = df_datasus.drop(columns=['code_state'], errors='ignore')
df_datasus['ano'] = df_datasus['ano'].str[:4].astype(int)


# Re-ordenar as colunas pelo padrão map_biomas
nova_ordem = ['uf','municipio','cod_municipio','ano','t_infecciosas', 't_neoplasias', 't_sangue',
            't_endocrinas', 't_nervoso', 't_olho', 't_ouvido', 't_cardiovascular',
            't_respiratorio', 't_digestivo', 't_pele', 't_osteomuscular',
            't_genitourinario', 't_malformacoes', 't_causas_externas',
            't_influencia', 't_total', 't_comunicaveis', 't_nao_comunicaveis',
            't_malaria', 't_cancer_mama', 't_cancer_colo_do_utero', 't_srag',
            't_total_atualizada']
df_datasus = df_datasus[nova_ordem]

# Transformar cod_municipio de float para int
df_datasus["cod_municipio"] = df_datasus["cod_municipio"].astype(int)

# Agrupar por UF, município, código e ano, somando todas as colunas numéricas
df_datasus = df_datasus.groupby(["uf", "municipio", "cod_municipio", "ano"], as_index=False).sum(numeric_only=True)

# Remover coluna t_total
df_datasus = df_datasus.drop(columns=['t_total', 'municipio', 'cod_municipio'], errors='ignore')

# Recalcular t_total_atualizada como soma de todas as colunas t_ exceto ela mesma
colunas_t = [col for col in df_datasus.columns if col.startswith("t_") and col != "t_total_atualizada"]
df_datasus["t_total_atualizada"] = df_datasus[colunas_t].sum(axis=1)

# Renomear t_total_atualizada para t_total
df_datasus = df_datasus.rename(columns={'t_total_atualizada': 't_total'})

# Filtrar apenas os anos entre 2006 a 2019
df_datasus = df_datasus[df_datasus['ano'].between(2006, 2019)]

# Agrupar por UF e ano, somando todas as colunas numéricas
df_datasus = df_datasus.groupby(['uf', 'ano'], as_index=False).sum(numeric_only=True)

# Verificar resultado final
print(df_datasus.shape)
print(df_datasus.columns)
print(df_datasus.head())

df_datasus.to_csv("resultados/datasus_amazonia.csv", index=False)