"""
    Questo progetto implementa un algoritmo random forest
    per predire il valore dell'ibu di una ricetta data in input
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Importa il modello di random forest

dataset = pd.read_csv('../dataset_uniti/ricette.csv')

# definizione delle features e del target
features = ["[HOPS]_alpha", "[HOPS]_amount", "[HOPS]_time", "[FERMENTABLES]_amount", "[FERMENTABLES]_color"]
target = "ibu"  # attributo target da prevedere

# divisione dei dati in set di addestramento e test
train_data = dataset.sample(frac=0.8)  # 80% dei dati per l'addestramento
test_data = dataset.drop(train_data.index)  # rimuovo i dati di addestramento dal set di test

# creazione del modello
model = RandomForestRegressor(n_estimators=100, max_depth=10)

# addestramento del modello
X_train = train_data[features]  # selezione delle features del set di addestramento
y_train = train_data[target]    # selezione del target del set di addestramento
model.fit(X_train, y_train)     # addestramento del modello utilizzando le features e il target

# valutazione del modello
X_test = test_data[features]            # selezione delle features del set di test
y_test = test_data[target]              # selezione del target del set di test
accuracy = model.score(X_test, y_test)  # calcolo l'accuracy del modello sul set di test
print("Accuracy:", accuracy)

new_recipe_features = {                 # Esempio di valori per le features
    "[HOPS]_alpha": [-1],               #G
    "[HOPS]_amount": [-0.17],           #H
    "[HOPS]_time": [-0.0023],           #I
    "[FERMENTABLES]_amount": [-0.16],   #J
    "[FERMENTABLES]_color": [-0.27]     #K
}
new_recipe_df = pd.DataFrame(new_recipe_features)  # creo un DataFrame con le features
ibu_stimato = model.predict(new_recipe_df)         # previsione in base alle features fornite
print("ibu stimato:", ibu_stimato)