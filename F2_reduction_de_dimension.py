import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fonction de réduction de dimension, on choisi arbitrairement a l'aide du tableau de corrélation
# quelle colonne garder.
# Prend en parametre le tableau des accident et retourne un tableau réduit à 4 colonnes
def F2_Reduction_Dimension_Manuelle(data):
    # Affichage du tableau de corrélation comparé a descr_grav
    data_only_numeric = data.select_dtypes(include=['float64', 'int64', 'int32'])
    print(data_only_numeric.corr()['descr_grav'])

    # On ne garde que les colonnes qui on =t un coefficient de corrélation superieur à 0.2 avec descr_grav
    data_tri = data[['descr_grav', 'descr_cat_veh', 'descr_agglo', 'descr_dispo_secu']] #0.2

    return data_tri

# Fonction de réduction de dimension, on utilise le module PCA de sklearn pour créer de nouvelles
# dimensions correspondant aux combinaisons linéaire de nos colonnes.
# Prend en parametre le tableau des accident et retourne un tableau réduit à 2 nouvelles dimensions
def F2_Reduction_Dimension_PCA(data):
    # Extraction des valeurs numériques de la DataFrame
    data_only_numeric = data.select_dtypes(include=['float64', 'int64', 'int32'])

    # Centrer et réduire les données
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_only_numeric)

    # Créer une instance de PCA avec le nombre de composantes souhaité
    pca = PCA(n_components=2)

    # Effectuer l'analyse en composantes principales
    data_tri_pca = pca.fit_transform(data_scaled)
    data_tri_pca = pd.DataFrame(data=data_tri_pca, columns=['PC1', 'PC2'])     #A modifier en fonction du nombre de composantes souhaité

    # Afficher les résultats
    print("Variance expliquée par chaque composante :", pca.explained_variance_ratio_)
    print("Composantes principales :", pca.components_)
    print("Données projetées sur les composantes principales :", data_tri_pca)

    # Eboulis des valeurs propres
    explained_variance = pca.explained_variance_ratio_
    plt.bar(range(len(explained_variance)), explained_variance)
    plt.xlabel('Composantes principales')
    plt.ylabel('Variance expliquée')
    plt.title('Eboulis des valeurs propres')
    plt.show()

    # Cercle de corrélation
    features = data_only_numeric.columns
    pcs = pca.components_
    scale = np.sqrt(pca.explained_variance_)

    plt.figure(figsize=(8, 8))
    plt.scatter(pcs[0], pcs[1], alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    for i, (x, y) in enumerate(zip(pcs[0], pcs[1])):
        plt.text(x, y, features[i], fontsize='9', ha='center', va='center')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title('Cercle de corrélation')
    plt.show()

    return data_tri_pca