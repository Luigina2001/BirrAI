import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

beer_dataset = pd.read_csv('../dataset_uniti/ricette.csv')
pd.set_option('display.max_columns', None)

# selezione dei campi pertinenti per l'analisi delle regole di associazione
fields = ['category', '[HOPS]_name', '[FERMENTABLES]_name', '[YEASTS]_name', '[MISCS]_name']
#fields = ['category', '[FERMENTABLES]_type']
#fields = ['category', '[YEASTS]_type']
#fields= ['category', '[MISCS]_type']
#fields= ['category', '[MISCS]_use']
#fields= ['category', '[HOPS]_user_hop_use']

#fields = ['category', '[HOPS]_user_hop_use', '[FERMENTABLES]_type', '[YEASTS]_type', '[MISCS]_type', '[MISCS]_use']



subset_data = beer_dataset[fields]

# trasformo il subset di dati in una lista di transazioni
transactions = subset_data.values.tolist()

# codifico il dataset in una matrice binaria
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)


# ------------------------------- ALGORITMO APRIORI -------------------------------
# algoritmo Apriori per trovare gli itemset frequenti
'''frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)''' #risulta meno efficiente per DB di grandi dimensioni

# ------------------------------- ALGORITMO FP-GROWTH -------------------------------
frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)
print("Itemset frequenti:")
print(frequent_itemsets)

# generazione delle regole di associazione
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
print("Regole di associazione:")
print(rules)
