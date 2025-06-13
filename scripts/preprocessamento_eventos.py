import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### DESMATAMENTO E MINERAÇÃO
# Carregando os arquivos
df_deforestation = pd.read_csv('dados/mapbiomas_deforestation.csv')
df_mining = pd.read_csv('dados/mapbiomas_mining.csv')

# Função para padronizar nomes de colunas comuns
def padronizar_colunas(df):
    df = df.rename(columns={
        'municipio': 'municipio',
        'uf': 'uf',
        'cod_municipio': 'cod_municipio',
        'bioma': 'bioma',
        'ano': 'ano',
        'valor': 'valor',
    })
    return df

df_deforestation = padronizar_colunas(df_deforestation)
df_mining = padronizar_colunas(df_mining)

# Renomear colunas específicas
df_deforestation = df_deforestation.rename(columns={"valor": "deforestation_valor"})
df_mining = df_mining.rename(columns={
    "level_1": "mining_level_1",
    "level_2": "mining_level_2",
    "level_3": "mining_level_3",
    "valor": "mining_valor"
})

# Remover colunas desnecessárias
df_deforestation = df_deforestation.drop(columns=['level_0','level_1','level_2','level_3','level_4','classe_desmatamento'], errors='ignore')
df_mining = df_mining.drop(columns=['name_pt_br'], errors='ignore')

# Agrupar desmatamento
df_sum_deforestation = df_deforestation.groupby(["uf", "municipio", "cod_municipio", "ano", "bioma"])['deforestation_valor'].sum().reset_index()

# Filtrar anos de interesse
anos_selecionados = range(2006, 2020)
df_sum_deforestation = df_sum_deforestation[df_sum_deforestation['ano'].isin(anos_selecionados)]
df_mining = df_mining[df_mining['ano'].isin(anos_selecionados)]

# Filtrar apenas UFs da Amazônia Legal
ufs_amazonia_legal = ['AC', 'AP', 'AM', 'MA', 'MT', 'PA', 'RO', 'RR', 'TO']
df_sum_deforestation = df_sum_deforestation[df_sum_deforestation['uf'].isin(ufs_amazonia_legal)]
df_mining = df_mining[df_mining['uf'].isin(ufs_amazonia_legal)]

# Criar tabela base
colunas_base = ["uf", "municipio", "cod_municipio", "ano", "bioma"]
tabela_base = pd.concat([
    df_sum_deforestation[colunas_base],
    df_mining[colunas_base]
]).drop_duplicates().reset_index(drop=True)

# Unificar dados
eventos_amazonia = tabela_base.copy()
eventos_amazonia = eventos_amazonia.merge(df_sum_deforestation, on=colunas_base, how="left")
eventos_amazonia = eventos_amazonia.merge(df_mining, on=colunas_base, how="left")

# Remover colunas desnecessárias
eventos_amazonia = eventos_amazonia.drop(columns=['municipio', 'cod_municipio', 'bioma', 'mining_level_1', 'mining_level_2', 'mining_level_3'])

# Agrupar por estado e ano
eventos_amazonia = eventos_amazonia.groupby(['uf', 'ano']).agg({
    'deforestation_valor': 'sum',
    'mining_valor': 'sum'
}).reset_index()

# Renomear colunas
eventos_amazonia = eventos_amazonia.rename(columns={
    'deforestation_valor': 'deforestation_soma',
    'mining_valor': 'mining_soma'
})

### QUEIMADAS
# Lista de arquivos e suas respectivas siglas de estado
arquivos_estados = {
    'queimadas_acre.csv': 'AC',
    'queimadas_amapa.csv': 'AP',
    'queimadas_amazonas.csv': 'AM',
    'queimadas_maranhao.csv': 'MA',
    'queimadas_mato-grosso.csv': 'MT',
    'queimadas_para.csv': 'PA',
    'queimadas_rondonia.csv': 'RO',
    'queimadas_roraima.csv': 'RR',
    'queimadas_tocantins.csv': 'TO'
}

# Colunas dos meses a remover
colunas_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

# Lista para armazenar todos os DataFrames processados
dfs_tratados = []

# Processar cada arquivo
for arquivo, sigla_uf in arquivos_estados.items():
    caminho = f'dados/{arquivo}'
    df = pd.read_csv(caminho)

    # Renomear colunas
    df = df.rename(columns={'Unnamed: 0': 'ano', 'Total': 'focos_ativos'})

    # Remover colunas dos meses
    df = df.drop(columns=colunas_meses, errors='ignore')

    # Remover linhas não numéricas (como "Máximo*", "Média*" etc.)
    df = df[df['ano'].astype(str).str.isnumeric()]

    # Converter ano para inteiro
    df['ano'] = df['ano'].astype(int)

    # Filtrar anos desejados
    df = df[df['ano'].between(2006, 2019)]

    # Adicionar a coluna de UF
    df['uf'] = sigla_uf

    # Reorganizar colunas
    df = df[['uf', 'ano', 'focos_ativos']]

    # Armazenar
    dfs_tratados.append(df)

# Concatenar todos os estados em um único DataFrame
df_queimadas_amazonia = pd.concat(dfs_tratados, ignore_index=True)

# Juntar com os dados de queimadas
eventos_completos = eventos_amazonia.merge(df_queimadas_amazonia, on=['uf', 'ano'], how='left')

# Ver resultado
print(eventos_completos.shape)
print(eventos_completos.columns)
print(eventos_completos.head())

# Salvar resultado final
eventos_completos.to_csv("resultados/eventos_amazonia.csv", index=False)
