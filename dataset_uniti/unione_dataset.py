import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import random


def disegna_grafico(dataset, numeric_columns, algoritmo):
    # figura e set di assi
    fig, ax = plt.subplots()

    # boxplot
    boxplot = ax.boxplot(dataset[numeric_columns].values, widths=0.5, patch_artist=True, showfliers=False)

    # modifiche grafiche
    colors = ['lightblue', 'lightgreen', 'lightpink']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    for whisker in boxplot['whiskers']:
        whisker.set(color='gray', linewidth=0.5)
    for cap in boxplot['caps']:
        cap.set(color='gray', linewidth=0.5)
    for median in boxplot['medians']:
        median.set(color='red', linewidth=0.5)

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.7)

    # etichette assi
    ax.set_xticklabels(numeric_columns)
    ax.set_ylabel('Valore')
    ax.set_title('Boxplot dopo ' + algoritmo + ' normalization')

    # dimensione della figura per evitare sovrapposizioni
    fig.tight_layout()
    plt.show()

def aggiungi_prefisso (dataset, prefisso):

    colonna_da_escludere = 'recipe_id'

    for colonna in dataset.columns:
        if colonna != colonna_da_escludere:
            nuovo_nome_colonna = prefisso + colonna
            dataset = dataset.rename(columns={colonna: nuovo_nome_colonna})
    return dataset

hops = pd.read_csv('../dataset_cleaned/hops_dataset_cleaned.csv')
fermentables = pd.read_csv('../dataset_cleaned/fermentables_dataset_cleaned.csv')
yeasts = pd.read_csv('../dataset_cleaned/yeasts_dataset_cleaned.csv')
miscs = pd.read_csv('../dataset_cleaned/miscs_dataset_cleaned.csv')
recipes = pd.read_csv('../dataset_cleaned/recipes_dataset_cleaned.csv')

# aggiunta prefissi ai campi dei vari dataset per distinguerli anche dopo l'operazione di join
hops = aggiungi_prefisso(hops, "[HOPS]_")
fermentables = aggiungi_prefisso(fermentables, "[FERMENTABLES]_")
yeasts = aggiungi_prefisso(yeasts, "[YEASTS]_")
miscs = aggiungi_prefisso(miscs, "[MISCS]_")

# unione dei dataset in base alla chiave recipe_id
merged_data = pd.merge(recipes, hops, on='recipe_id')
merged_data = pd.merge(merged_data, fermentables, on='recipe_id')
merged_data = pd.merge(merged_data, yeasts, on='recipe_id')
merged_data = pd.merge(merged_data, miscs, on='recipe_id')
pd.set_option('display.max_columns', None)  # Mostra tutte le colonne

#print(merged_data.head())

#merged_data.to_csv("dataset_uniti/ricette.csv", index=False)
#print("------------------------------------------------------------------------------------------------------------------------------------------")

#print('lunghezza dataset iniziale: ', len(merged_data))

# ------------------------------- DATA CLEANING -------------------------------

# calcolo i valori mancanti per colonna
missing_values = merged_data.isnull().sum()
#print(missing_values) #0

# calcolo il numero totale di valori mancanti nel dataset
total_missing = merged_data.isnull().sum().sum()
#print("Totale valori mancanti:", total_missing) #0



# ------------------------------- FEATURE SCALING -------------------------------
'''
# per ogni colonna della tabella mostra le statistiche (per verificare valori min e max di ogni campo della tabella)
for col in merged_data:
    print(col, "\n",merged_data[col].describe(), "\n\n")'''

numeric_columns = merged_data.select_dtypes(include=[np.number]).columns

# rimuovo la colonna "recipe_id" se presente
if 'recipe_id' in numeric_columns:
    numeric_columns = numeric_columns.drop('recipe_id')

# [1]Boxplot prima del feature scaling
'''for col in numeric_columns:
    values = dataset[col].values  # Estraggo i valori della colonna
    plt.plot(values, label=col)  # Crea un grafico a linea dei valori

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.title('Boxplot prima del feature scaling')
plt.show()'''

# Z-score normalization
scaler = StandardScaler()
merged_data[numeric_columns] = scaler.fit_transform(merged_data[numeric_columns])

# [3]Boxplot dopo z-score normalization
#disegna_grafico(merged_data, numeric_columns, "z-score")



# ------------------------------- FEATURE SELECTION -------------------------------

nomi_campi = merged_data.columns
#print("\nNomi dei campi prima del feature selection")
#print(nomi_campi)

non_numeric_columns = merged_data.select_dtypes(include=['object', 'bool']).columns
# aggiungo la colonna "recipe_id" se non presente
if 'recipe_id' not in non_numeric_columns:
    non_numeric_columns = non_numeric_columns.append(pd.Index(['recipe_id']))


# oggetto VarianceThreshold con soglia di varianza zero
selector = VarianceThreshold(threshold=0)
# selezione delle feature alle sole colonne numeriche del dataset
dataset_selected = selector.fit_transform(merged_data[numeric_columns])

# indici delle feature selezionate
feature_indices = selector.get_support(indices=True)
# nomi delle feature selezionate
selected_features = [feature for i, feature in enumerate(numeric_columns) if i in feature_indices]

# nomi delle feature selezionate
'''print("Feature selezionate:")
for feature in selected_features:
    print(feature)'''

# creo un nuovo dataset con le sole feature selezionate
dataset_new = merged_data[selected_features].copy()

# aggiungo le colonne non numeriche al nuovo DataFrame
for column in non_numeric_columns:
    dataset_new[column] = merged_data[column]

merged_data= dataset_new
#print(merged_data)

nomi_campi = merged_data.columns
#print("\nNomi dei campi dopo del feature selection")
#print(nomi_campi)
#eliminato il campo [YEASTS]_amount
#eliminare anche [YEASTS]_amount_is_weight?


# ------------------------------- DATA BALANCING (Undersampling) in base alla colonna category -------------------------------
'''
# numero di campioni per ogni categoria
category_counts = merged_data['category'].value_counts()
print(category_counts)
# numero minimo di campioni tra le categorie
min_count = category_counts.min()
'''

'''
# Undersampling
balanced_data = pd.DataFrame()
for category in category_counts.index:
    category_data = merged_data[merged_data['category'] == category].sample(min_count, random_state=42)
    balanced_data = pd.concat([balanced_data, category_data])

# numero elementi del nuovo dataset bilanciato
#print(len(balanced_data))

#balanced_data.to_csv("ricette.csv", index=False)
'''

# numero desiderato di voci per ogni categoria
num_entries_desiderate = 1000

# creo un nuovo DataFrame vuoto per il dataset bilanciato
balanced_data = pd.DataFrame(columns=merged_data.columns)

for category in merged_data['category'].unique():
    entries = merged_data[merged_data['category'] == category]
    count = len(entries)

    if count > num_entries_desiderate:
        # eseguo l'undersampling per le categorie con più di 1000 voci
        entries = entries.sample(num_entries_desiderate, replace=False)
    else:
        # eseguo l'oversampling per le categorie con meno di 1000 voci
        num_duplicates = num_entries_desiderate - count
        duplicates = entries.sample(num_duplicates, replace=True)
        entries = pd.concat([entries, duplicates])

    # aggiungo le voci al DataFrame modificato
    balanced_data = pd.concat([balanced_data, entries])

# verifico se il numero di voci di ogni categoria è uguale al num_entries_desiderate (1000)
category_counts = balanced_data['category'].value_counts()
print(category_counts)
print(balanced_data)
print(len(balanced_data))

balanced_data.to_csv('ricette.csv', index=False)

'''nomi_campi = merged_data.columns
print("\nNomi campi")
print(nomi_campi)

range_valori = merged_data['est_abv'].agg(['min', 'max'])
print("est_abv -->", range_valori)

range_valori = merged_data['ibu'].agg(['min', 'max'])
print("ibu -->", range_valori)

range_valori = merged_data['[FERMENTABLES]_color'].agg(['min', 'max'])
print("[FERMENTABLES]_color -->", range_valori)'''

