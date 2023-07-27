"""
    Questo progetto implementa un algoritmo genetico per generare nuove ricette basate su
    un dataset di ricette esistente.
"""

import pandas as pd
import random
import numpy as np

dataset = pd.read_csv('../dataset_uniti/ricette.csv')

# parametri dell'algoritmo genetico
size_popolazione = 50
generazioni = 10
size_mating_pool = 5
mutation_rate = 0.1

# funzione per calcolare il punteggio di una ricetta
def calcola_punteggio(ricetta):
    # metrica 1: ABV (Alcohol By Volume)
    abv = ricetta['est_abv']
    abv_score = 0

    # assegno un punteggio in base all'ABV desiderato, si predilige un grado alcolico più basso
    # intervallo di valori assunti d'attributo --> [-3.501565; 23.863067]
    if abv <= 2:
        abv_score = 10
    elif abv > 2 and abv <= 5:
        abv_score = 8
    elif abv > 5 and abv <= 9:
        abv_score = 5
    elif abv > 9 and abv <= 15:
        abv_score = 3
    else:
        abv_score = 1

    # metrica 2: ibu
    ibu = ricetta['ibu']
    ibu_score = 0

    # assegno un punteggio in base all'ibu desiderato
    # intervallo di valori assunti d'attributo --> [-1.298474; 33.314285]
    # si predilige un ibu più basso, quindi una birra più amara
    if ibu >= 22:
        ibu_score = 10
    elif ibu >= 12 and ibu < 22:
        ibu_score = 8
    elif ibu >= 6 and ibu < 12:
        ibu_score = 5
    elif ibu >= 0 and ibu < 6:
        ibu_score = 2
    else:
        ibu_score = 1

    # metrica 3: colore del mosto
    color = ricetta['[FERMENTABLES]_color']
    color_score = 0

    # assegno un punteggio in base al colore desiderato [-0.399302; 30.996475]
    # intervallo di valori assunti d'attributo --> [-0.399302; 30.996475]
    # si predilige un colore più scuro
    if color >= 20:
        color_score = 10
    elif color >= 15 and color < 20:
        color_score = 8
    elif color >= 10 and color <= 15:
        color_score = 8
    elif color >= 5 and color < 10:
        color_score = 5
    else:
        color_score = 2

    # punteggio totale
    score = (abv_score + ibu_score + color_score) / 3
    return score


# funzione per creare una ricetta casuale
def crea_ricetta_casuale():
    ricetta = {}

    # campi delle ricette
    campi_ricetta = ['batch_size', 'boil_size', 'boil_time', 'efficiency', 'est_abv', 'ibu', 'category',
                     'display_batch_size', 'display_boil_size', 'name', 'type']

    for campo in campi_ricetta:
        ricetta[campo] = np.random.choice(dataset[campo].values)

    # campi degli ingredienti dalle colonne corrispondenti nel dataset
    campi_ingredienti = ['[HOPS]_alpha', '[HOPS]_amount', '[HOPS]_time', '[HOPS]_form', '[HOPS]_name',
                         '[HOPS]_use', '[HOPS]_user_hop_use', '[FERMENTABLES]_amount', '[FERMENTABLES]_color',
                         '[FERMENTABLES]_yield', '[FERMENTABLES]_add_after_boil', '[FERMENTABLES]_name',
                         '[FERMENTABLES]_type', '[YEASTS]_amount_is_weight', '[YEASTS]_form',
                         '[YEASTS]_name', '[YEASTS]_type', '[MISCS]_amount', '[MISCS]_amount_is_weight',
                         '[MISCS]_name', '[MISCS]_type', '[MISCS]_use']

    for campo in campi_ingredienti:
        ricetta[campo] = np.random.choice(dataset[campo].values)

    return ricetta

# funzione per creare la popolazione iniziale di ricette
def crea_popolazione_iniziale():
    popolazione = []
    for _ in range(size_popolazione):
        recipe = crea_ricetta_casuale()
        popolazione.append(recipe)
    return popolazione

# funzione per selezionare il mating pool dalla popolazione
def seleziona_mating_pool_popolazione(popolazione, size_mating_pool):
    popolazione_ordinata = sorted(popolazione, key=lambda x: calcola_punteggio(x), reverse=True)
    return popolazione_ordinata[:size_mating_pool]

# funzione per eseguire l'operazione di crossover su due genitori per creare un figlio
def crossover(parent1, parent2):
    figlio = {}

    # crossover dei campi dell'indice delle ricette
    campi_ricetta = ['batch_size', 'boil_size', 'boil_time', 'efficiency', 'est_abv', 'ibu', 'category',
                     'display_batch_size', 'display_boil_size', 'name', 'type']

    for campo in campi_ricetta:
        if random.random() < 0.5:
            figlio[campo] = parent1[campo]
        else:
            figlio[campo] = parent2[campo]

    # crossover dei campi degli ingredienti
    campi_ingredienti = ['[HOPS]_alpha', '[HOPS]_amount', '[HOPS]_time', '[HOPS]_form', '[HOPS]_name',
                         '[HOPS]_use', '[HOPS]_user_hop_use', '[FERMENTABLES]_amount', '[FERMENTABLES]_color',
                         '[FERMENTABLES]_yield', '[FERMENTABLES]_add_after_boil', '[FERMENTABLES]_name',
                         '[FERMENTABLES]_type', '[YEASTS]_amount_is_weight', '[YEASTS]_form',
                         '[YEASTS]_name', '[YEASTS]_type', '[MISCS]_amount', '[MISCS]_amount_is_weight',
                         '[MISCS]_name', '[MISCS]_type', '[MISCS]_use']

    for campo in campi_ingredienti:
        if random.random() < 0.5:
            figlio[campo] = parent1[campo]
        else:
            figlio[campo] = parent2[campo]

    return figlio

# funzione per eseguire l'operazione di mutazione su una ricetta
def mutazione(ricetta):
    ricetta_mut = ricetta.copy()
    for campo in ricetta_mut:
        if campo != 'recipe_id':  # evito di mutare l'attributo 'recipe_id'
            ricetta_mut[campo] = random.choice(dataset[campo].tolist())
    return ricetta_mut

# funzione per calcolare la prossima generazione di ricette
def calcola_generazione_successiva(popolazione):
    mating_pool = seleziona_mating_pool_popolazione(popolazione, size_mating_pool)

    generazione_successiva = mating_pool.copy()

    while len(generazione_successiva) < size_popolazione:
        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)

        figlio = crossover(parent1, parent2)
        figlio = mutazione(figlio)

        generazione_successiva.append(figlio)

    return generazione_successiva


# creazione di nuove ricette
popolazione = crea_popolazione_iniziale()

for generazione in range(generazioni):
    print(f"Generazione {generazione + 1}")
    popolazione = calcola_generazione_successiva(popolazione)

# cerco la ricetta migliore nella popolazione finale
miglior_ricetta = max(popolazione, key=lambda x: calcola_punteggio(x))
print("Nuova ricetta generata:")
print(miglior_ricetta)