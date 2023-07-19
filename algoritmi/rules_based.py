"""
    Questo progetto implementa l'applicazione di regole specifiche per combinare gli ingredienti,
    in base alle caratteristiche della categoria di birra desiderata.
"""

import pandas as pd
import numpy as np

# Dataset di ingredienti
hops = pd.read_csv('../dataset_cleaned/hops_dataset_cleaned.csv')
fermentables = pd.read_csv('../dataset_cleaned/fermentables_dataset_cleaned.csv')
yeasts = pd.read_csv('../dataset_cleaned/yeasts_dataset_cleaned.csv')
miscs = pd.read_csv('../dataset_cleaned/miscs_dataset_cleaned.csv')
recipes = pd.read_csv('../dataset_cleaned/recipes_dataset_cleaned.csv', usecols=['category', 'recipe_id'])

hops = pd.merge(recipes, hops, on='recipe_id')
fermentables = pd.merge(recipes, fermentables, on='recipe_id')
yeasts = pd.merge(recipes, yeasts, on='recipe_id')
miscs = pd.merge(recipes, miscs, on='recipe_id')


# controllo che la categoria data in input sia presente nel dataset delle ricette
def controllo_categoria_ricetta(categoria):
    if recipes['category'].str.lower().isin([categoria.lower()]).any():
        print("Categoria valida.")
        return True
    print("Categoria NON valida.")
    return False


# controllo che l'ingrediente selezionato sia della giusta categoria
def controllo_categoria_ingrediente(categoria, nome_ingrediente, tabella):
    for index, row in tabella.iterrows():
        if (row['name'].lower() == pd.Series(nome_ingrediente).astype(str).str.lower()).any() and (
                row['category'].lower() == categoria.lower()):
            return True
    return False


# seleziono una riga del dataset di ingredienti passato come argomento alla funzione
def ottieni_ingrediente(dataset, categoria, tipologia_ingrediente):
    if tipologia_ingrediente == 'hops':
        if categoria == 'ipa':
            selezionabili = dataset[(dataset['alpha'] >= 10) & (dataset['alpha'] <= 21.15)] # scelgo luppoli con magiore amarezza
        elif categoria == 'stout' or categoria == 'porter':
            selezionabili = dataset[(dataset['alpha'] <= 5)]                                # scelgo luppoli con amarezza moderata
        elif categoria == 'belgian ale':
            selezionabili = dataset[(dataset['alpha'] <= 2)]                                # scelgo luppoli con amarezza bassa
        else:
            selezionabili= dataset
    else:
        selezionabili = dataset

    if len(selezionabili)>0:
        ingredient = selezionabili.sample(n=1)['name'].values[0]
        # dataset.drop(ingredient.index, inplace=False)  # per evitare di utilizzare due volte lo stesso ingrediente
        return ingredient                                # ['name'].values[0]
    else:
        return None
    # return np.random.choice(dataset['name'].values)


# creazione della ricetta di birra in base alla categoria passata come argomento alla funzione
def crea_ricetta(categoria):
    recipe = {}
    while True:
        recipe["hops1"] = ottieni_ingrediente(hops, categoria, 'hops')
        if controllo_categoria_ingrediente(categoria, recipe["hops1"], hops):
            break
    while True:
        recipe["hops2"] = ottieni_ingrediente(hops, categoria, 'hops')
        if controllo_categoria_ingrediente(categoria, recipe["hops2"], hops):
            break

    # IPA: birra ad alta fermentazione caratterizzata dall’impiego di una grande quantità di luppoli
    if (categoria.lower() == "ipa") or (categoria.lower() == "india pale ale"):
        while True:
            recipe["hops3"] = ottieni_ingrediente(hops, categoria, 'hops')
            if controllo_categoria_ingrediente(categoria, recipe["hops3"], hops):
                break
    else:
        recipe["hops3"] = ""

    while True:
        recipe["fermentables1"] = ottieni_ingrediente(fermentables, categoria, 'fermentables')
        if controllo_categoria_ingrediente(categoria, recipe["fermentables1"], fermentables):
            break
    while True:
        recipe["fermentables2"] = ottieni_ingrediente(fermentables, categoria, 'fermentables')
        if controllo_categoria_ingrediente(categoria, recipe["fermentables2"], fermentables):
            break
    while True:
        recipe["yeasts"] = ottieni_ingrediente(yeasts, categoria, 'yeasts')
        if controllo_categoria_ingrediente(categoria, recipe["yeasts"], yeasts):
            break

    if categoria.lower() == "pilsner":
        recipe["miscs"] = "-"

    else:
        while True:
            recipe["miscs"] = ottieni_ingrediente(miscs, categoria, 'miscs')
            if controllo_categoria_ingrediente(categoria, recipe["miscs"], miscs):
                break
    return recipe


# utilizzo dell'algoritmo per creare una ricetta di birra
while True:
    categoria = input("Inserisci la categoria di birra desiderata (es. IPA, Stout, ecc.) --> ")
    if (controllo_categoria_ricetta(categoria)):
        break
    else:
        print("Mi dispiace, non sono in grado di creare una birra di questa categoria.")

recipe = crea_ricetta(categoria.lower())
print("\nRicetta di birra - stile", categoria)
print("Luppoli:", recipe["hops1"], ",", recipe["hops2"], ",", recipe["hops3"])
print("Malti:", recipe["fermentables1"], ",", recipe["fermentables2"])
print("Lieviti:", recipe["yeasts"])
print("Varie:", recipe["miscs"])
