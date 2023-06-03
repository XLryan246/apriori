import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Carregando a base de dados
dados = pd.read_csv('compradores_bicicleta.csv')

# Tratando a tabela, excluindo a coluna Home Owner, ID
dados = dados.drop(['ID'], axis=1)

# Tratando a tabela, diferenciando as colunas de quantidade de crianças e carros
dados['Children'] = dados['Children'].apply(lambda x: str(x) + ' Children')
dados['Cars'] = dados['Cars'].apply(lambda x: str(x) + ' Cars')
dados['Home Owner'] = dados['Home Owner'].apply(lambda x: str(x) + '_Home Owner')
dados['Purchased Bike'] = dados['Purchased Bike'].apply(lambda x: str(x) + '_Purchased Bike')

# Transformando os dados em uma lista de listas
usuarios = []
for linha in dados.values:
    usuarios.append([str(item) for item in linha])

# Codificação das transações
transaction_encoder = TransactionEncoder()
transaction_encoder_array = transaction_encoder.fit(usuarios).transform(usuarios)
df = pd.DataFrame(transaction_encoder_array, columns=transaction_encoder.columns_)

# Aplicação do algoritmo PRIORI
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True, verbose=1)
frequent_itemsets = frequent_itemsets.sort_values(by=['support'], ascending=False)
frequent_itemsets = frequent_itemsets.reset_index(drop=True)

# Criação das regras de associação
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

#Filtrar para mostrar apenas 'antecedents', 'consequents', 'confidence'
rules = rules[['antecedents', 'consequents', 'confidence']]

#Filtrar para mostrar apenas regras com 'consequents' = 'Yes_Purchased Bike' e 'No_Purchased Bike'
rules_yes = rules[rules['consequents'].astype(str).str.contains('Yes_Purchased Bike')]
rules_yes = rules_yes.sort_values(by=['confidence'], ascending=False)
rules_yes = rules_yes.reset_index(drop=True)

rules_no = rules[rules['consequents'].astype(str).str.contains('No_Purchased Bike')]
rules_no = rules_no[~rules_no['consequents'].astype(str).str.contains('North')]
rules_no = rules_no.sort_values(by=['confidence'], ascending=False)
rules_no = rules_no.reset_index(drop=True)


# Exibição dos resultados
print("Associações frequentes:")
print(frequent_itemsets)
print("\nRegras de associação, com consequente = 'Yes':")
print(rules_yes)
print("\nRegras de associação, com consequente = 'No':")
print(rules_no)