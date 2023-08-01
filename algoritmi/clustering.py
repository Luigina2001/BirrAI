"""
    Questo progetto mira ad organizzare (tramite l'algoritmo di clustering K-means) e generare ricette
    in base alle loro caratteristiche, aprendo la possibilità di scoprire nuove combinazioni.
"""
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


dataset = pd.read_csv("../dataset_uniti/ricette.csv")

# selezione delle colonne rilevanti per l'algoritmo di clustering
selected_features = ['category', '[HOPS]_name', '[FERMENTABLES]_name', '[YEASTS]_name', '[MISCS]_name', 'name']
#X = dataset[selected_features]
X = dataset.loc[:, selected_features].copy()


# encoding numerico che permette di convertire le stringhe in valori numerici per poter essere utilizzati nell'algoritmo di clustering
label_encoders = {}
for feature in selected_features:
    label_encoders[feature] = LabelEncoder()
    X[feature] = label_encoders[feature].fit_transform(X[feature])

num_clusters = 8    # numero di cluster desiderati
kmeans = KMeans(init='k-means++', algorithm='elkan', n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(X)
iterazioni = kmeans.n_iter_
print(iterazioni)


# etichette di clustering per ogni ricetta
labels = kmeans.labels_
dataset['cluster_label'] = labels

# calcolo le caratteristiche medie di ciascun cluster
cluster_centers = kmeans.cluster_centers_


# ricette appartenenti a ciascun cluster
for cluster in range(num_clusters):
    recipes_in_cluster = dataset[dataset['cluster_label'] == cluster]
    print(f"\nRicette nel cluster {cluster + 1}:")
    print(recipes_in_cluster[['name', 'category']])

print('---------------------------------------------------------------------------')


# generazione di nuove ricette basate sulle caratteristiche medie di ciascun cluster
new_recipes = []
for cluster_center in cluster_centers:
    new_recipe = {
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


print('---------------------------------------------------------------------------')

'''silhouette_avg = silhouette_score(X, labels)
print(f"\nCoefficiente di silhouette: {silhouette_avg}")'''

# indice di Davies-Bouldin
davies_bouldin = davies_bouldin_score(X, labels)
print(f"\nIndice di Davies-Bouldin: {davies_bouldin}")

# indice di Calinski-Harabasz
calinski_harabasz = calinski_harabasz_score(X, labels)
print(f"\nIndice di Calinski-Harabasz: {calinski_harabasz}")

# ---------------------------------- Grafico complessivo cluster ----------------------------------

'''pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

kmeans = KMeans(init='k-means++', algorithm='elkan', n_clusters=num_clusters, random_state=0, n_init='auto')
kmeans.fit(X)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap='viridis', edgecolor='k', alpha=0.7)

# Aggiungi la leggenda con il numero del cluster e il colore corrispondente
legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc='upper right', bbox_to_anchor=(1.18, 1), prop={'size': 15})
ax.add_artist(legend)

ax.set_xlabel('Componente principale 1', fontsize=16)
ax.set_ylabel('Componente principale 2', fontsize=16)
ax.set_zlabel('Componente principale 3', fontsize=16)
plt.grid(True)

legend.set_title("Cluster", prop={"size": 16})
plt.show()

'''
# ---------------------------------- Elbow point ----------------------------------
'''elbow_scores = []

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
plt.show()'''

# ---------------------------- Silhouette coefficient ----------------------------------------

'''X = dataset.loc[:, selected_features].copy()

# encoding numerico che permette di convertire le stringhe in valori numerici per poter essere utilizzati nell'algoritmo di clustering
label_encoders = {}
for feature in selected_features:
    label_encoders[feature] = LabelEncoder()
    X[feature] = label_encoders[feature].fit_transform(X[feature])

num_clusters = 8    # numero di cluster desiderati

silhouette_scores = []

# Calcolo del coefficiente di silhouette per diversi valori di K
for num_clusters in range(5, 11):
    kmeans = KMeans(init='k-means++', algorithm='elkan', n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# Plot del coefficiente di silhouette al variare di K
plt.plot(range(5, 11), silhouette_scores, marker='o')
plt.title("Coefficienti di Silhouette al variare del numero di cluster")
plt.xlabel("Numero di cluster (K)")
plt.ylabel("Coefficiente di Silhouette")
plt.show()'''



# ---------------------------- Numero elementi in ciascun cluster ----------------------------------------
'''
# numero di istanze in ogni cluster
num_elements_in_cluster = [8030, 7106, 10004, 6011, 6187, 7051, 6117, 6494]

# etichette dei cluster
cluster_labels = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7", "Cluster 8"]

# grafico a barre
plt.figure(figsize=(10, 6))
plt.bar(cluster_labels, num_elements_in_cluster, color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Numero di Elementi')
plt.title('Numero di Elementi in Ciascun Cluster')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

'''