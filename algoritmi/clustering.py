"""
    Questo progetto mira ad organizzare (tramite l'algoritmo di clustering K-means) e generare ricette
    in base alle loro caratteristiche, aprendo la possibilità di scoprire nuove combinazioni.
"""
import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("../dataset_uniti/ricette.csv")

# selezione delle colonne rilevanti per l'algoritmo di clustering
selected_features = ['category', '[HOPS]_name', '[FERMENTABLES]_name', '[YEASTS]_name', '[MISCS]_name']
X = dataset[selected_features]

# encoding numerico che permette di convertire le stringhe in valori numerici per poter essere utilizzati nell'algoritmo di clustering
label_encoders = {}
for feature in selected_features:
    label_encoders[feature] = LabelEncoder()
    X[feature] = label_encoders[feature].fit_transform(X[feature])

num_clusters = 8    # numero di cluster desiderati
kmeans = KMeans(init='k-means++', algorithm='elkan', n_clusters=num_clusters, random_state=0, n_init='auto')
kmeans.fit(X)

# etichette di clustering per ogni ricetta
labels = kmeans.labels_
dataset['cluster_label'] = labels

# ricette appartenenti a ciascun cluster
for cluster in range(num_clusters):
    recipes_in_cluster = dataset[dataset['cluster_label'] == cluster]
    print(f"\nRicette nel cluster {cluster + 1}:")
    print(recipes_in_cluster[['name', 'category']])

print('---------------------------------------------------------------------------')

# calcolo le caratteristiche medie di ciascun cluster
cluster_centers = kmeans.cluster_centers_

# generazione di nuove ricette basate sulle caratteristiche medie di ciascun cluster
new_recipes = []
for cluster_center in cluster_centers:
    new_recipe = {
        'category': cluster_center[0],
        '[HOPS]_name': cluster_center[1],
        '[FERMENTABLES]_name': cluster_center[2],
        '[YEASTS]_name': cluster_center[3],
        '[MISCS]_name': cluster_center[4],
    }
    new_recipes.append(new_recipe)

# nuove ricette generate
for i, recipe in enumerate(new_recipes):
    print(f"\nNuova ricetta {i+1}:")
    for feature, value in recipe.items():
        print(f"{feature}: {label_encoders[feature].inverse_transform([int(value)])}")



# ---------------------------------- Elbow point ----------------------------------
elbow_scores = []

for num_clusters in range(2, 15):
    kmeans = KMeans(init='k-means++', algorithm='elkan', n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(X)
    labels = kmeans.labels_
    elbow_scores.append(kmeans.inertia_)

# Plot dell'elbow point
plt.plot(range(2, 15), elbow_scores, marker='o')
plt.title("Elbow point")
plt.xlabel("Numero di cluster")
plt.ylabel("Varianza")
plt.show()
