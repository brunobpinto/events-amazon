import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv("resultados/eventos-doencas_amazonia.csv")

# Identificar apenas colunas numéricas (exceto 'ano', que será mantido como identificação)
numeric_cols = df.select_dtypes(include='number').columns.difference(['ano'])

# Copiar o DataFrame para preservar o original
df_discretized = df.copy()

# Discretizar colunas numéricas em tercis (baixo, medio, alto)
for col in numeric_cols:
    try:
        df_discretized[col] = pd.qcut(df[col], q=3, labels=["baixo", "medio", "alto"], duplicates='drop')
    except ValueError as e:
        print(f"Erro ao discretizar a coluna {col}: {e}")

# Remover colunas que não puderam ser discretizadas
colunas_problema = ['t_cancer_mama', 't_cancer_colo_do_utero', 't_srag', 't_influencia', 't_olho']
df_discretized = df_discretized.drop(columns=colunas_problema)

# Exibir as primeiras linhas para verificação
print(df_discretized.head())

# Converter para formato transacional
transactions = []
for _, row in df_discretized.iterrows():
    transacao = []
    for col in df_discretized.columns:
        if col not in ['uf', 'ano']:
            transacao.append(f"{col}={row[col]}")
    transactions.append(transacao)

# Transformar em estrutura binarizada para Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar Apriori
frequent_itemsets = apriori(df_trans, min_support=0.1, use_colnames=True, max_len=3)

# Gerar regras de associação
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Listas para identificar eventos e doenças
eventos = ['deforestation_soma', 'mining_soma', 'focos_ativos']
doenca_prefix = 't_'

# Função: verificar se o antecedente tem apenas eventos com valor "alto"
def antecedente_evento_alto(itemset):
    return all(
        any(ev in str(i) for ev in eventos) and '=alto' in str(i)
        for i in itemset
    )

# Função: verificar se o consequente tem apenas doenças com valor "alto"
def consequente_doenca_alta(itemset):
    return all(
        str(i).startswith(doenca_prefix) and '=alto' in str(i)
        for i in itemset
    )

# Aplicar filtro nas regras
regras_eventos_altos = rules[
    rules['antecedents'].apply(antecedente_evento_alto) &
    rules['consequents'].apply(consequente_doenca_alta)
]

# Ordenar pelas mais fortes (lift alto)
regras_eventos_altos = regras_eventos_altos.sort_values(by='lift', ascending=False)

# Filtro de regras fortes e relevantes
regras_boas = regras_eventos_altos[
    (regras_eventos_altos['support'] >= 0.15) &
    (regras_eventos_altos['confidence'] >= 0.8) &
    (regras_eventos_altos['lift'] >= 2)
]

# Ordenar pelas regras mais interessantes (lift alto)
regras_boas = regras_boas.sort_values(by='lift', ascending=False)

# Exibir as 10 melhores regras
print(regras_boas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Dados das 10 melhores regras
dados = [
    (("deforestation_soma=alto", "focos_ativos=alto"), ("t_cardiovascular=alto",), 0.198413, 1.0, 3.0),
    (("focos_ativos=alto", "mining_soma=alto"), ("t_cardiovascular=alto",), 0.198413, 1.0, 3.0),
    (("deforestation_soma=alto", "mining_soma=alto"), ("t_causas_externas=alto",), 0.238095, 0.909091, 2.727273),
    (("deforestation_soma=alto", "mining_soma=alto"), ("t_respiratorio=alto",), 0.238095, 0.909091, 2.727273),
    (("focos_ativos=alto",), ("t_nao_comunicaveis=alto", "t_cardiovascular=alto"), 0.269841, 0.809524, 2.684211),
    (("focos_ativos=alto",), ("t_cardiovascular=alto",), 0.293651, 0.880952, 2.642857),
    (("deforestation_soma=alto", "focos_ativos=alto"), ("t_causas_externas=alto",), 0.174603, 0.88, 2.64),
    (("deforestation_soma=alto", "focos_ativos=alto"), ("t_nao_comunicaveis=alto",), 0.174603, 0.88, 2.64),
    (("deforestation_soma=alto", "focos_ativos=alto"), ("t_digestivo=alto",), 0.174603, 0.88, 2.64),
    (("deforestation_soma=alto", "focos_ativos=alto"), ("t_nervoso=alto",), 0.174603, 0.88, 2.64)
]

# Criar DataFrame
df_regras = pd.DataFrame(dados, columns=["antecedents", "consequents", "support", "confidence", "lift"])
df_regras['antecedents'] = df_regras['antecedents'].apply(lambda x: ', '.join(x))
df_regras['consequents'] = df_regras['consequents'].apply(lambda x: ', '.join(x))

# Criar imagem da tabela com ajustes de largura de coluna
fig, ax = plt.subplots(figsize=(18, 6))
ax.axis('off')

tabela = ax.table(cellText=df_regras.values,
                  colLabels=df_regras.columns,
                  cellLoc='center',
                  loc='center')

tabela.auto_set_font_size(False)
tabela.set_fontsize(10)

# Ajustar largura das colunas: aumentar 'antecedents' e 'consequents'
col_widths = [0.4, 0.4, 0.1, 0.1, 0.1]  # proporcionalmente maiores para texto
for i, width in enumerate(col_widths):
    for j in range(len(df_regras) + 1):  # +1 por causa do header
        cell = tabela[j, i]
        cell.set_width(width)

# Salvar como PNG
plt.savefig("resultados/regras_eventos-doencas.png", bbox_inches='tight')
plt.show()
