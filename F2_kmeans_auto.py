import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

from F2_evaluation import *

def F2_Preparation_Data_Pour_Classification(data, nombre_de_valeur):
    # On prépare le tableau qu'on va utiliser pour k-Means
    data_pos =pd.DataFrame(data=data, columns= ["latitude", "longitude"]).head(nombre_de_valeur)
    return data_pos

def F2_Affichage_kMeans_Auto(data_pos, nombre_clusters):
    # Créer un objet KMeans avec le nombre de clusters souhaité
    kmeans = KMeans(n_clusters=nombre_clusters)

    # Appliquer K-means sur les données
    kmeans.fit(data_pos)

    # Obtenir les labels des clusters pour chaque échantillon
    labels = kmeans.labels_

    # Obtenir les coordonnées des centroïdes de chaque cluster
    centroides = kmeans.cluster_centers_

    # Afficher les labels des clusters et les coordonnées des centroïdes
    # print("Labels des clusters :", labels)
    # print("Coordonnées des centres :", centres)

    # On rajoute la variable cluster pour un affichage couleur de chaque cluster
    data_pos.loc[:, 'Cluster'] = labels.astype(str)

    # On fais une data frame des centroides pour pouvoir les afficher
    data_centroides = pd.DataFrame(centroides, columns=['latitude', 'longitude'])

    # On prépare les informations qu'on met en sortie
    data_kMeans = [labels, data_centroides]

    # Carte de chaque accidents coloré par cluster
    fig = px.scatter_geo(data_pos, locationmode='country names',
                         lat='latitude', lon='longitude',
                         projection='natural earth', color='Cluster',
                         color_discrete_sequence=px.colors.qualitative.Pastel)

    # Ajout des centroïdes sur la carte
    fig.add_trace(go.Scattergeo(locationmode='country names',
                                lat=data_centroides['latitude'], lon=data_centroides['longitude'],
                                marker=dict(size=15,color='red')))

    # Personnaliser l'apparence de la carte
    fig.update_geos(fitbounds="locations", showcountries=True)
    fig.update_layout(title='Clusters accident en France en 2019 (K-Means)')

    # Afficher la carte
    fig.show()

    return data_kMeans

