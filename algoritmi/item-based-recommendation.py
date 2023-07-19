"""
    Questo progetto, data in input l'id di una ricetta presente nel database, calcola le ricette più simili
    in base agli ingredienti e agli attributi per poi successivamente combinare loro e creare
    una nuova ricetta di birra.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('../dataset_uniti/ricette.csv')
pd.set_option('display.max_columns', None)

# selezione dei campi rilevanti per la raccomandazione
campi_rilevanti = ['category', '[HOPS]_alpha', '[HOPS]_amount', '[HOPS]_time',
                   '[HOPS]_form', '[HOPS]_name', '[HOPS]_use',
                   '[FERMENTABLES]_amount', '[FERMENTABLES]_color', '[FERMENTABLES]_yield',
                   '[FERMENTABLES]_add_after_boil', '[FERMENTABLES]_name', '[FERMENTABLES]_type',
                   '[YEASTS]_form', '[YEASTS]_name', '[YEASTS]_type', '[MISCS]_amount',
                   '[MISCS]_amount_is_weight', '[MISCS]_name', '[MISCS]_type',
                   '[MISCS]_use']

# encoding numerico che permette di convertire le stringhe in valori numerici per poter essere utilizzati nell'algoritmo
label_encoders = {}
for feature in campi_rilevanti:
    if pd.api.types.is_numeric_dtype(dataset[feature]):
        pass
    else:
        label_encoders[feature] = LabelEncoder()
        dataset[feature] = label_encoders[feature].fit_transform(dataset[feature])


# calcolo la similarità tra le ricette in base ai campi rilevanti
item_similarity = cosine_similarity(dataset[campi_rilevanti], dataset[campi_rilevanti])

def consiglia_ricette_birra(recipe_id, num_recommendations=5):
    # indice della ricetta corrispondente all'id passato come argomento alla funzione
    indice_ricetta = dataset[dataset['recipe_id'] == recipe_id].index[0]

    # calcola la similarità della ricetta corrente con tutte le altre ricette
    similarita_ricetta = item_similarity[indice_ricetta]
    indici_ordinati = similarita_ricetta.argsort() # indici similarità in ordine crescente

    # seleziono gli indici dalla seconda posizione fino a num_recommendations + 1 (incluso) dagli indici ordinati
    indici_selezionati = indici_ordinati[1:num_recommendations + 1]

    #  inverto l'ordine degli indici selezionati per ottenere gli indici delle ricette più simili in cima alla lista
    indici_ricette_simili = indici_selezionati[::-1]

    ricette_consigliate = dataset.loc[indici_ricette_simili]#, 'name']

    return ricette_consigliate

def combina_ricette(ricette):

    # combino gli ingredienti o gli attributi delle ricette per creare una nuova ricetta
    nuova_ricetta = {}

    # per ogni campo feature, calcolo la media dei valori corrispondenti in tutte le ricette fornite come input.
    for feature in campi_rilevanti:
        #nuova_ricetta[feature] = ricette[feature].mean()
        nuova_ricetta[feature] = ricette[feature].median()

    nuova_ricetta = pd.DataFrame(nuova_ricetta, index=[0])

    return nuova_ricetta


recipe_id = 74302.0     # id di una ricetta presente nel dataset
ricette_simili = consiglia_ricette_birra(recipe_id, num_recommendations=5)

nuova_ricetta = combina_ricette(ricette_simili)

print('\n------------------------------------------------------------------\n')

print("Ricetta di partenza:")
print(dataset[dataset['recipe_id'] == recipe_id][['recipe_id', 'name']])

print('\n------------------------------------------------------------------\n')

print(f"Ricette consigliate per la ricetta con id {recipe_id} :")
print(ricette_simili[['recipe_id', 'name']])

print('\n------------------------------------------------------------------\n')

print("\nNuova ricetta creata combinando le ricette simili:")
print(nuova_ricetta)