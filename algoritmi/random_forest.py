"""
    Questo progetto implementa un algoritmo random forest
    per predire il valore dell'ibu di una ricetta data in input
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder
import random



dataset = pd.read_csv('../dataset_uniti/ricette.csv')

# definizione delle features e del target
features = ['batch_size', 'boil_size', 'boil_time', 'efficiency', 'est_abv',
       'category', 'display_batch_size', 'display_boil_size', 'name', 'type',
       '[HOPS]_alpha', '[HOPS]_amount', '[HOPS]_time',
       '[HOPS]_form', '[HOPS]_name', '[HOPS]_use', '[HOPS]_user_hop_use',
       '[FERMENTABLES]_amount', '[FERMENTABLES]_color', '[FERMENTABLES]_yield',
       '[FERMENTABLES]_add_after_boil', '[FERMENTABLES]_name',
       '[FERMENTABLES]_type', '[YEASTS]_amount', '[YEASTS]_amount_is_weight',
       '[YEASTS]_form', '[YEASTS]_name', '[YEASTS]_type']#, '[MISCS]_amount',
       #'[MISCS]_amount_is_weight', '[MISCS]_name', '[MISCS]_type',
       #'[MISCS]_use']

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
model = RandomForestRegressor(n_estimators=100, max_depth=30)

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
print("Mean Absolute Error (MAE):", mae)

# errore quadratico medio (Mean Squared Error, MSE)
mse = sum((y_test - y_pred) ** 2) / len(y_test)
print("Mean Squared Error (MSE):", mse, "\n\n")

# ------------------------ stampe risultati ------------------------

# numero di esempi da visualizzare
num_examples = 5

# righe casuali dal set di test
random_indices = random.sample(range(len(test_data)), num_examples)
sample_data = test_data.iloc[random_indices]

# seleziono solo le features per l'esempio
X_sample = sample_data[features]

# predizioni utilizzando il modello
predictions = model.predict(X_sample)

# confronto delle predizioni con i valori reali
for i in range(num_examples):
    real_value = sample_data.iloc[i][target]
    predicted_value = predictions[i]
    print(f"Esempio {i + 1}:")
    print(f"IBU stimata: {predicted_value:.2f}")
    print(f"IBU reale: {real_value:.2f}")
    print("-" * 20)

'''import matplotlib.pyplot as plt

# Creazione del grafico di confronto
plt.figure(figsize=(10, 6))
plt.title("Confronto IBU stimati vs. IBU reali")
plt.xlabel("Esempio")
plt.ylabel("Valore IBU")
plt.grid(True)

# Etichette degli esempi
labels = [f"Esempio {i + 1}" for i in range(num_examples)]

# Valori stimati e reali dell'IBU per gli esempi selezionati
predicted_values = predictions
real_values = [sample_data.iloc[i][target] for i in range(num_examples)]

# Plot dei valori predetti e reali
plt.plot(labels, predicted_values, label="IBU stimati", marker="o")
plt.plot(labels, real_values, label="IBU reali", marker="x")

# Aggiungi legenda e mostra il grafico
plt.legend()
plt.tight_layout()
plt.show()'''


import matplotlib.pyplot as plt

# valori stimati e reali dell'IBU per tutto il set di test
predicted_values = y_pred
real_values = y_test

# grafico a dispersione
plt.figure(figsize=(10, 6))
plt.title("Confronto IBU stimati vs. IBU reali (Set di Test)")
plt.xlabel("IBU stimati")
plt.ylabel("IBU reali")
plt.grid(True)

# plot dei valori predetti rispetto ai valori reali
plt.scatter(predicted_values, real_values, alpha=0.7, color='b')

# linea diagonale per aiutare nella valutazione visiva
plt.plot([-1.5, 1.5], [-1.5, 1.5], color='r', linestyle='--')

plt.tight_layout()
plt.show()


