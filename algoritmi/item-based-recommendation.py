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
campi_rilevanti = ['category', '[HOPS]_alpha', '[HOPS]_name',
                   'est_abv', 'ibu', 'type', '[FERMENTABLES]_color', '[FERMENTABLES]_name',
                   '[YEASTS]_name', '[YEASTS]_type', '[MISCS]_name']

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
    indici_ordinati = indici_ordinati[::-1] # indici similarità in ordine decrescente

    # seleziono gli indici dalla seconda posizione fino a num_recommendations + 1 (incluso) dagli indici ordinati
    indici_selezionati = indici_ordinati[1:num_recommendations + 1]

    ricette_consigliate = dataset.loc[indici_selezionati]#, 'name']

    # similarità corrispondenti alle ricette consigliate
    similarita_consigliate = similarita_ricetta[indici_selezionati]

    for i in range(len(ricette_consigliate)):
        ricetta_nome = ricette_consigliate.iloc[i]['name']
        similarita = similarita_consigliate[i]
        print(f"Ricetta consigliata: {ricetta_nome}, Similarità: {similarita}")

    print('\n------------------------------------------------------------------\n')

    return ricette_consigliate

'''def combina_ricette(ricette):

    # combino gli ingredienti o gli attributi delle ricette per creare una nuova ricetta
    nuova_ricetta = {}

    # per ogni campo feature, calcolo la media dei valori corrispondenti in tutte le ricette fornite come input.
    for feature in campi_rilevanti:
        #nuova_ricetta[feature] = ricette[feature].mean()
        nuova_ricetta[feature] = ricette[feature].median()

    nuova_ricetta = pd.DataFrame(nuova_ricetta, index=[0])

    return nuova_ricetta'''

def combina_ricette(ricette):

    dataset_originale = pd.read_csv('../dataset_uniti/ricette.csv')

    nuova_ricetta = {
        'nome': 'Ricetta combinata',
        'categorie_combinate': [],
        'Luppoli': [],
        'Malti': [],
        'Lieviti': [],
        'Varie': []
    }

    # indici delle ricette consigliate
    indici_ricette_simili = ricette.index

    nuova_ricetta['Luppoli'].extend(dataset_originale.loc[indici_ricette_simili, '[HOPS]_name'].values.flatten().tolist())
    nuova_ricetta['Malti'].extend(dataset_originale.loc[indici_ricette_simili, '[FERMENTABLES]_name'].values.flatten().tolist())
    nuova_ricetta['Lieviti'].extend(dataset_originale.loc[indici_ricette_simili, '[YEASTS]_name'].values.flatten().tolist())
    nuova_ricetta['Varie'].extend(dataset_originale.loc[indici_ricette_simili, '[MISCS]_name'].values.flatten().tolist())

    nuova_ricetta['categorie_combinate'].extend(dataset_originale.loc[indici_ricette_simili, 'category'].values.flatten().tolist())

    # rimuovo gli ingredienti duplicati dalla nuova ricetta
    nuova_ricetta['Luppoli'] = list(set(nuova_ricetta['Luppoli']))
    nuova_ricetta['Malti'] = list(set(nuova_ricetta['Malti']))
    nuova_ricetta['Lieviti'] = list(set(nuova_ricetta['Lieviti']))
    nuova_ricetta['Varie'] = list(set(nuova_ricetta['Varie']))

    nuova_ricetta['categorie_combinate'] = list(set(nuova_ricetta['categorie_combinate']))

    return nuova_ricetta


recipe_id = 83392.0     # id di una ricetta presente nel dataset
print("\nRicetta di partenza:")
print(dataset[dataset['recipe_id'] == recipe_id][['recipe_id', 'name']])

print('\n------------------------------------------------------------------\n')

ricette_simili = consiglia_ricette_birra(recipe_id, num_recommendations=5)
print(f"Ricette consigliate per la ricetta con id {recipe_id} :")
print(ricette_simili[['recipe_id', 'name']])

print('\n------------------------------------------------------------------\n')

nuova_ricetta = combina_ricette(ricette_simili)
print("\nNuova ricetta creata combinando le ricette simili:")
print(nuova_ricetta)