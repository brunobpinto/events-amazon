import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv("resultados/eventos-doencas_amazonia.csv")

# Identificar apenas colunas numéricas (exceto 'ano', que será mantido como identificação)
numeric_cols = df.select_dtypes(include='number').columns.difference(['ano'])

# ----------- Dataset com Remoção de Outliers -----------
df_sem_outliers = df.copy()

# Remover outliers por IQR em cada coluna numérica
for col in numeric_cols:
    Q1 = df_sem_outliers[col].quantile(0.25)
    Q3 = df_sem_outliers[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filtra apenas os valores dentro dos limites
    df_sem_outliers = df_sem_outliers[(df_sem_outliers[col] >= lower_bound) & (df_sem_outliers[col] <= upper_bound)]


# ----------- Dataset Discretizado em Tercis -----------
df_tercis = df.copy()
df_tercis_outliers = df_sem_outliers.copy()

for col in numeric_cols:
    try:
        df_tercis[col] = pd.qcut(df[col], q=3, labels=["baixo", "medio", "alto"], duplicates='drop')
        df_tercis_outliers[col] = pd.qcut(df[col], q=3, labels=["baixo", "medio", "alto"], duplicates='drop')
    except ValueError as e:
        pass
        # print(f"[TERCIS] Erro ao discretizar a coluna {col}: {e}")


# ----------- Dataset Discretizado em Quintis -----------
df_quintis = df.copy()
df_quintis_outliers = df_sem_outliers.copy()

for col in numeric_cols:
    try:
        df_quintis[col] = pd.qcut(df[col], q=5, labels=["baixo", "baixo_medio", "medio", "medio_alto", "alto"], duplicates='drop')
        df_quintis_outliers[col] = pd.qcut(df[col], q=5, labels=["baixo", "baixo_medio", "medio", "medio_alto", "alto"], duplicates='drop')
    except ValueError as e:
        pass
        # print(f"[QUINTIS] Erro ao discretizar a coluna {col}: {e}")


# Remover colunas que não puderam ser discretizadas
colunas_problema_tercis = ['t_cancer_mama', 't_cancer_colo_do_utero', 't_srag', 't_influencia', 't_olho']
colunas_problema_quintis = ['t_cancer_mama', 't_cancer_colo_do_utero', 't_srag', 't_influencia', 't_olho', 't_malaria', 't_ouvido']
df_tercis = df_tercis.drop(columns=colunas_problema_tercis)
df_quintis = df_quintis.drop(columns=colunas_problema_quintis)
df_tercis_outliers = df_tercis_outliers.drop(columns=colunas_problema_tercis)
df_quintis_outliers = df_quintis_outliers.drop(columns=colunas_problema_quintis)

# Visualizar as discretizações
# print("Dataset em Tercis:")
# print(df_tercis.head())
# print(df_tercis_outliers.head())

# print("\nDataset em Quintis:")
# print(df_quintis.head())
# print(df_quintis_outliers.head())

# ----------------- Regras Apriori -------------------
def gerar_regras_apriori(df_discretizado, min_support=0.1, min_confidence=0.6, max_len=3):
    # Converter para formato transacional
    transactions = []
    for _, row in df_discretizado.iterrows():
        transacao = []
        for col in df_discretizado.columns:
            if col not in ['uf', 'ano']:
                transacao.append(f"{col}={row[col]}")
        transactions.append(transacao)

    # Transformar em estrutura binarizada para Apriori
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Aplicar Apriori
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True, max_len=max_len)

    # Gerar regras
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return rules


# ----------------- Datasets e Parâmetros -------------------
support = [0.10, 0.15]
confidence = [0.60, 0.70]

df_tercis_outliers_apriori_10_60 = gerar_regras_apriori(df_tercis_outliers, min_support=support[0], min_confidence=confidence[0])
df_tercis_outliers_apriori_10_70 = gerar_regras_apriori(df_tercis_outliers, min_support=support[0], min_confidence=confidence[1])
df_tercis_outliers_apriori_15_60 = gerar_regras_apriori(df_tercis_outliers, min_support=support[1], min_confidence=confidence[0])
df_tercis_outliers_apriori_15_70 = gerar_regras_apriori(df_tercis_outliers, min_support=support[1], min_confidence=confidence[1])
df_quintis_outliers_apriori_10_60 = gerar_regras_apriori(df_quintis_outliers, min_support=support[0], min_confidence=confidence[0])
df_quintis_outliers_apriori_10_70 = gerar_regras_apriori(df_quintis_outliers, min_support=support[0], min_confidence=confidence[1])
df_quintis_outliers_apriori_15_60 = gerar_regras_apriori(df_quintis_outliers, min_support=support[1], min_confidence=confidence[0])
df_quintis_outliers_apriori_15_70 = gerar_regras_apriori(df_quintis_outliers, min_support=support[1], min_confidence=confidence[1])
df_tercis_apriori_10_60 = gerar_regras_apriori(df_tercis, min_support=support[0], min_confidence=confidence[0])
df_tercis_apriori_10_70 = gerar_regras_apriori(df_tercis, min_support=support[0], min_confidence=confidence[1])
df_tercis_apriori_15_60 = gerar_regras_apriori(df_tercis, min_support=support[1], min_confidence=confidence[0])
df_tercis_apriori_15_70 = gerar_regras_apriori(df_tercis, min_support=support[1], min_confidence=confidence[1])
df_quintis_apriori_10_60 = gerar_regras_apriori(df_quintis, min_support=support[0], min_confidence=confidence[0])
df_quintis_apriori_10_70 = gerar_regras_apriori(df_quintis, min_support=support[0], min_confidence=confidence[1])
df_quintis_apriori_15_60 = gerar_regras_apriori(df_quintis, min_support=support[1], min_confidence=confidence[0])
df_quintis_apriori_15_70 = gerar_regras_apriori(df_quintis, min_support=support[1], min_confidence=confidence[1])


# ----------------- Regras FP-Growth -------------------
def gerar_regras_fpgrowth(df_discretizado, min_support=0.1, min_confidence=0.6, max_len=3):
    # Converter para formato transacional
    transactions = []
    for _, row in df_discretizado.iterrows():
        transacao = []
        for col in df_discretizado.columns:
            if col not in ['uf', 'ano']:
                transacao.append(f"{col}={row[col]}")
        transactions.append(transacao)

    # Transformar em estrutura binarizada
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Aplicar FP-Growth
    frequent_itemsets = fpgrowth(df_trans, min_support=min_support, use_colnames=True, max_len=max_len)

    # Gerar regras
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return rules

# ----------------- Datasets e Parâmetros -------------------
df_tercis_outliers_fpgrowth_10_60 = gerar_regras_fpgrowth(df_tercis_outliers, min_support=support[0], min_confidence=confidence[0])
df_tercis_outliers_fpgrowth_10_70 = gerar_regras_fpgrowth(df_tercis_outliers, min_support=support[0], min_confidence=confidence[1])
df_tercis_outliers_fpgrowth_15_60 = gerar_regras_fpgrowth(df_tercis_outliers, min_support=support[1], min_confidence=confidence[0])
df_tercis_outliers_fpgrowth_15_70 = gerar_regras_fpgrowth(df_tercis_outliers, min_support=support[1], min_confidence=confidence[1])
df_quintis_outliers_fpgrowth_10_60 = gerar_regras_fpgrowth(df_quintis_outliers, min_support=support[0], min_confidence=confidence[0])
df_quintis_outliers_fpgrowth_10_70 = gerar_regras_fpgrowth(df_quintis_outliers, min_support=support[0], min_confidence=confidence[1])
df_quintis_outliers_fpgrowth_15_60 = gerar_regras_fpgrowth(df_quintis_outliers, min_support=support[1], min_confidence=confidence[0])
df_quintis_outliers_fpgrowth_15_70 = gerar_regras_fpgrowth(df_quintis_outliers, min_support=support[1], min_confidence=confidence[1])
df_tercis_fpgrowth_10_60 = gerar_regras_fpgrowth(df_tercis, min_support=support[0], min_confidence=confidence[0])
df_tercis_fpgrowth_10_70 = gerar_regras_fpgrowth(df_tercis, min_support=support[0], min_confidence=confidence[1])
df_tercis_fpgrowth_15_60 = gerar_regras_fpgrowth(df_tercis, min_support=support[1], min_confidence=confidence[0])
df_tercis_fpgrowth_15_70 = gerar_regras_fpgrowth(df_tercis, min_support=support[1], min_confidence=confidence[1])
df_quintis_fpgrowth_10_60 = gerar_regras_fpgrowth(df_quintis, min_support=support[0], min_confidence=confidence[0])
df_quintis_fpgrowth_10_70 = gerar_regras_fpgrowth(df_quintis, min_support=support[0], min_confidence=confidence[1])
df_quintis_fpgrowth_15_60 = gerar_regras_fpgrowth(df_quintis, min_support=support[1], min_confidence=confidence[0])
df_quintis_fpgrowth_15_70 = gerar_regras_fpgrowth(df_quintis, min_support=support[1], min_confidence=confidence[1])


# ----------------- Filtros para Eventos e Doenças -------------------
eventos = ['deforestation_soma', 'mining_soma', 'focos_ativos']
doenca_prefix = 't_'
valores_altos = ['=alto', '=medio_alto']

def antecedente_evento_alto_ou_medio_alto(itemset):
    return all(
        any(ev in str(i) for ev in eventos) and any(v in str(i) for v in valores_altos)
        for i in itemset
    )

def consequente_doenca_alta_ou_media_alta(itemset):
    return all(
        str(i).startswith(doenca_prefix) and any(v in str(i) for v in valores_altos)
        for i in itemset
    )


# ----------------- Aplicar filtros a todos os DataFrames -------------------
# Lista com nomes das variáveis de regras Apriori e FP-Growth
nomes_regras = [
    "df_tercis_outliers_apriori_10_60", "df_tercis_outliers_apriori_10_70",
    "df_tercis_outliers_apriori_15_60", "df_tercis_outliers_apriori_15_70",
    "df_tercis_outliers_fpgrowth_10_60", "df_tercis_outliers_fpgrowth_10_70",
    "df_tercis_outliers_fpgrowth_15_60", "df_tercis_outliers_fpgrowth_15_70",

    "df_quintis_outliers_apriori_10_60", "df_quintis_outliers_apriori_10_70",
    "df_quintis_outliers_apriori_15_60", "df_quintis_outliers_apriori_15_70",
    "df_quintis_outliers_fpgrowth_10_60", "df_quintis_outliers_fpgrowth_10_70",
    "df_quintis_outliers_fpgrowth_15_60", "df_quintis_outliers_fpgrowth_15_70",

    "df_tercis_apriori_10_60", "df_tercis_apriori_10_70",
    "df_tercis_apriori_15_60", "df_tercis_apriori_15_70",
    "df_tercis_fpgrowth_10_60", "df_tercis_fpgrowth_10_70",
    "df_tercis_fpgrowth_15_60", "df_tercis_fpgrowth_15_70",

    "df_quintis_apriori_10_60", "df_quintis_apriori_10_70",
    "df_quintis_apriori_15_60", "df_quintis_apriori_15_70",
    "df_quintis_fpgrowth_10_60", "df_quintis_fpgrowth_10_70",
    "df_quintis_fpgrowth_15_60", "df_quintis_fpgrowth_15_70"
]

# Loop para aplicar os filtros e imprimir os resultados
for nome in nomes_regras:
    regras = globals().get(nome)
    if regras is not None and not regras.empty:
        regras_filtradas = regras[
            regras['antecedents'].apply(antecedente_evento_alto_ou_medio_alto) &
            regras['consequents'].apply(consequente_doenca_alta_ou_media_alta)
        ]

        regras_boas = regras_filtradas[
            (regras_filtradas['support'] >= 0.10) &
            (regras_filtradas['confidence'] >= 0.6) &
            (regras_filtradas['lift'] >= 2)
        ].sort_values(by='lift', ascending=False)

        print(f"\n🔍 Top 10 regras para {nome} (filtradas):")
        if not regras_boas.empty:
            print(regras_boas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        else:
            print("Nenhuma regra relevante encontrada com os critérios.")
    else:
        print(f"\n{nome} não existe ou está vazio.")


# ----------------- Avaliação dos Cenários (completos, mesmo sem regras) -------------------

avaliacoes = []

for nome in nomes_regras:
    regras = globals().get(nome)
    if regras is not None and not regras.empty:
        regras_filtradas = regras[
            regras['antecedents'].apply(antecedente_evento_alto_ou_medio_alto) &
            regras['consequents'].apply(consequente_doenca_alta_ou_media_alta)
        ]

        regras_boas = regras_filtradas[
            (regras_filtradas['support'] >= 0.10) &
            (regras_filtradas['confidence'] >= 0.6) &
            (regras_filtradas['lift'] >= 2)
        ].sort_values(by='lift', ascending=False).head(10)

        if not regras_boas.empty:
            media_support = regras_boas['support'].mean()
            media_confidence = regras_boas['confidence'].mean()
            media_lift = regras_boas['lift'].mean()
            qtd = len(regras_boas)
        else:
            media_support = np.nan
            media_confidence = np.nan
            media_lift = np.nan
            qtd = 0
    else:
        media_support = np.nan
        media_confidence = np.nan
        media_lift = np.nan
        qtd = 0

    avaliacoes.append({
        "cenario": nome,
        "qtd_regras": qtd,
        "media_support": media_support,
        "media_confidence": media_confidence,
        "media_lift": media_lift
    })

# Converter para DataFrame
df_avaliacoes = pd.DataFrame(avaliacoes)

# Garantir que os cenários apareçam na ordem da lista original
df_avaliacoes = df_avaliacoes.set_index("cenario").reindex(nomes_regras).reset_index()

# Exibir resultados
print("\n📊 Avaliação completa dos 32 cenários (na ordem definida):")
print(df_avaliacoes.to_string(index=False))

# Salvar como CSV
df_avaliacoes.to_csv("resultados/experimentacao_regras.csv", index=False)