"""
    Questo progetto implementa un algoritmo random forest
    per predire il valore dell'ibu di una ricetta data in input
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


dataset = pd.read_csv('../dataset_uniti/ricette.csv')

# definizione delle features e del target
features = ['batch_size', 'boil_size', 'boil_time', 'efficiency', 'est_abv',
       'category', 'display_batch_size', 'display_boil_size', 'name', 'type',
       'recipe_id', '[HOPS]_alpha', '[HOPS]_amount', '[HOPS]_time',
       '[HOPS]_form', '[HOPS]_name', '[HOPS]_use', '[HOPS]_user_hop_use',
       '[FERMENTABLES]_amount', '[FERMENTABLES]_color', '[FERMENTABLES]_yield',
       '[FERMENTABLES]_add_after_boil', '[FERMENTABLES]_name',
       '[FERMENTABLES]_type', '[YEASTS]_amount', '[YEASTS]_amount_is_weight',
       '[YEASTS]_form', '[YEASTS]_name', '[YEASTS]_type', '[MISCS]_amount',
       '[MISCS]_amount_is_weight', '[MISCS]_name', '[MISCS]_type',
       '[MISCS]_use']

target = "ibu"  # attributo target da prevedere

# encoding numerico che permette di convertire le stringhe in valori numerici per poter essere utilizzati nell'algoritmo
label_encoders = {}
for feature in features:
    if pd.api.types.is_numeric_dtype(dataset[feature]):
        pass
    else:
        label_encoders[feature] = LabelEncoder()
        dataset[feature] = label_encoders[feature].fit_transform(dataset[feature])


# divisione dei dati in set di addestramento e test (80% addestramento, 20% test)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# creazione del modello
model = RandomForestRegressor(n_estimators=100, max_depth=10)

# addestramento del modello
X_train = train_data[features]  # selezione delle features del set di addestramento
y_train = train_data[target]    # selezione del target del set di addestramento
model.fit(X_train, y_train)     # addestramento del modello utilizzando le features e il target

# salvo il modello addestrato in un file (opzionale)
joblib.dump(model, "modello_random_forest.pkl")

# valutazione del modello sul set di test
X_test = test_data[features]            # selezione delle features del set di test
y_test = test_data[target]              # selezione del target del set di test

# valutazione del modello sul set di test
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# errore medio assoluto (Mean Absolute Error, MAE)
y_pred = model.predict(X_test)
mae = sum(abs(y_test - y_pred)) / len(y_test)
print("Mean Absolute Error:", mae)

# errore quadratico medio (Mean Squared Error, MSE)
mse = sum((y_test - y_pred) ** 2) / len(y_test)
print("Mean Squared Error:", mse)

'''
# Trova gli indici degli esempi correttamente predetti
correct_indices = (y_test - y_pred).abs() < 0.1  # Consideriamo corretta una predizione con errore inferiore a 0.1

# Conta i valori corretti e errati
num_correct = correct_indices.sum()
num_wrong = len(y_test) - num_correct

# Creazione del grafico a barre
plt.bar(['Corretti', 'Errati'], [num_correct, num_wrong], color=['blue', 'red'], alpha=0.5)
plt.xlabel('Predizione')
plt.ylabel('Numero di Esempi')
plt.title('Distribuzione dei Valori Predetti')
plt.show()'''