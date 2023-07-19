import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #per grafici e visualizzazioni
from matplotlib import rcParams #per configurare i parametri di visualizzazione,
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


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


# parametri di visualizzazione
plt.style.use("ggplot") # stile dei grafici
rcParams['figure.figsize'] = (12,  6) # dimensione figure dei grafici in pollici (altezza e larghezza)
pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne

yeasts = "../dataset/yeasts.csv"
dataset = pd.read_csv(yeasts, sep=';')
''' print(dataset.head())

print("------------------------------------------------------------------------------------------------------------------------------------------")

nomi_campi = dataset.columns
print("\nNomi campi")
print(nomi_campi)

print("------------------------------------------------------------------------------------------------------------------------------------------")

numero_campi = dataset.shape[1]
print("\nNumero campi")
print(numero_campi)

print("------------------------------------------------------------------------------------------------------------------------------------------")

print("\nInfo")
dataset.info()

print("------------------------------------------------------------------------------------------------------------------------------------------")

statistiche = dataset.describe(include='all')
print("\nStatistiche")
print(statistiche)

print("------------------------------------------------------------------------------------------------------------------------------------------")

colonna= "amount_is_weight"
#colonna= "flocculation"
#colonna= "form"
#colonna= "type"
#colonna= "version"

valori_unici = dataset[colonna].unique() # calcolo i valori unici dalla colonna
print("\nValori unici colonna amount_is_weight")
print(valori_unici) 

print("------------------------------------------------------------------------------------------------------------------------------------------")
 '''

# ------------------------------- DATA CLEANING -------------------------------

# calcolo i valori mancanti per colonna
missing_values = dataset.isnull().sum()
print(missing_values)

# calcolo il numero totale di valori mancanti nel dataset
total_missing = dataset.isnull().sum().sum()
print("Totale valori mancanti prima del data cleaning:", total_missing)

#elimino colonne con elevato numero di valori nulli
columns_to_drop = ['attenuation', 'flocculation', 'laboratory', 'max_temperature', 'min_temperature', 'product_id']
dataset = dataset.drop(columns_to_drop, axis=1)

# elimino le righe con valori nulli nelle colonne form e type
dataset = dataset.dropna(subset=['form', 'type'])

# most frequent imputation colonna name
most_frequent_value = dataset['name'].mode()[0]     # calcolo il valore più frequente nella colonna name
print("most frequent value colonna name: " + most_frequent_value)
# sostituisco i valori nulli con il valore più frequente
dataset['name'].fillna(most_frequent_value, inplace=True)

# riempimento valori nulli colonna amount_is_weight
amount_is_not_weight= False
dataset['amount_is_weight'].fillna(amount_is_not_weight, inplace=True)

# [VERIFICA] ricalcolo i valori mancanti per colonna
missing_values = dataset.isnull().sum().sum()
print("Totale valori mancanti dopo il data cleaning: ", missing_values)



# ------------------------------- FEATURE SCALING -------------------------------

'''
# per ogni colonna della tabella mostra le statistiche (per verificare valori min e max di ogni campo della tabella)
for col in dataset:
    print(col, "\n", dataset[col].describe(), "\n\n") '''

numeric_columns = dataset.select_dtypes(include=[np.number]).columns
# rimuovo la colonna "recipe_id" se presente
if 'recipe_id' in numeric_columns:
    numeric_columns = numeric_columns.drop('recipe_id')


# [1]Boxplot prima del feature scaling
for col in numeric_columns:
    values = dataset[col].values     #estraggo i valori della colonna
    plt.plot(values, label=col)      # creo un grafico a linea dei valori

plt.xlim(0, 50)
plt.ylim(0, 50)
plt.legend()
plt.title('Boxplot prima del feature scaling')
plt.show()



# Min-Max normalization
'''min_max_scaler = preprocessing.MinMaxScaler()
dataset[numeric_columns] = min_max_scaler.fit_transform(dataset[numeric_columns].to_numpy())

# [2]Boxplot dopo min-max normalization
disegna_grafico(dataset, numeric_columns, "min-max")'''



# Z-score normalization
scaler = StandardScaler()
dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

# [3]Boxplot dopo z-score normalization
disegna_grafico(dataset, numeric_columns, "z-score")





# ------------------------------- FEATURE SELECTION -------------------------------

non_numeric_columns = dataset.select_dtypes(include=['object', 'bool']).columns
# aggiungo la colonna "recipe_id" se non presente
if 'recipe_id' not in non_numeric_columns:
    non_numeric_columns = non_numeric_columns.append(pd.Index(['recipe_id']))

# oggetto VarianceThreshold con soglia di varianza zero
selector = VarianceThreshold(threshold=0)
# applico la selezione delle feature alle sole colonne numeriche del dataset
dataset_selected = selector.fit_transform(dataset[numeric_columns])

# indici delle feature selezionate
feature_indices = selector.get_support(indices=True)
# nomi delle feature selezionate
selected_features = [feature for i, feature in enumerate(numeric_columns) if i in feature_indices]

# nomi delle feature selezionate
print("Feature selezionate:")
for feature in selected_features:
    print(feature)

# creo un nuovo dataset con le sole feature selezionate
dataset_new = dataset[selected_features].copy()

# aggiungo le colonne non numeriche al nuovo DataFrame
for column in non_numeric_columns:
    dataset_new[column] = dataset[column]

dataset = dataset_new

numero_righe = len(dataset[dataset['name'] == '- -'])
#print("Numero di righe in cui il campo 'nome' è uguale a '--':", numero_righe) #11.945
numero_totale_di_righe= len(dataset)
#print("Numero totale di righe: ", numero_totale_di_righe) #152.772

# Eliminazione di righe in cui il nome è nullo
dataset = dataset[dataset['name'] != '- -']

#print(dataset)
dataset.to_csv("../dataset_cleaned/yeasts_dataset_cleaned.csv", index=False)

