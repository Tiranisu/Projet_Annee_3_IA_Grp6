import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def F2_Preparation_Data_Pour_Classification(data, nombre_de_valeur):
    # On prépare le tableau qu'on va utiliser pour k-Means
    data = data.head(nombre_de_valeur)
    data_pos = data.copy()
    data_pos = data_pos[['latitude', 'longitude']]

    # Mettre la date qui est au format (MM/DD/YY hh:mm:ss) au format numérique : YMMDDhh
    data["date"] = (data["date"].str.split('/').str[2].str.split(" ").str[0] +
                    data["date"].str.split('/').str[0].str.split("(").str[1] + data["date"].str.split('/').str[1] +
                    data["date"].str.split('/').str[2].str.split(" ").str[1].str.split(":").str[0]).astype(int)

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
    #fig.show()

    return data_kMeans

def F2_Silhouette_Coefficient(data_pos, labels):
    # Calculer le coefficient de silhouette
    silhouette_coefficient = silhouette_score(data_pos, labels)

    return silhouette_coefficient

def F2_Calinski_Harabasz_Index(data_pos, labels):
    # Calculer le coefficient de silhouette
    calinski_harabasz_index = calinski_harabasz_score(data_pos, labels)

    return calinski_harabasz_index

def F2_Davies_Bouldin_Index(data_pos, labels):
    # Calculer le coefficient de silhouette
    davies_bouldin_index = davies_bouldin_score(data_pos, labels)

    return davies_bouldin_index

# Lecture du CSV exporté du projet Big Data
data = pd.read_csv('export.csv', dtype={0: str})

# Reduction de dimension
#data_tri_MAN = F2_Reduction_Dimension_Manuelle(data)
#data_tri_PCA = F2_Reduction_Dimension_PCA(data)

# On créer une base de donnée avec suelement la latitude et la longitude
data_pos = F2_Preparation_Data_Pour_Classification(data, 70000)

# On effectue k-Means et on récupere les labels de toute nos latitude/longitude
data_kMeans = F2_Affichage_kMeans_Auto(data_pos, 4)
labels_kMeans = data_kMeans[0]
centroids_kMeans = data_kMeans[1]
print(centroids_kMeans)

# Calcul des coeffs
silhouette_coefficient = F2_Silhouette_Coefficient(data_pos, labels_kMeans)
calinski_harabasz_index = F2_Calinski_Harabasz_Index(data_pos, labels_kMeans)
davies_bouldin_index = F2_Davies_Bouldin_Index(data_pos, labels_kMeans)

# Afficher les coeffs
print("Coefficient de silhouette :", silhouette_coefficient)                    #Proche de 1 = bien (entre -1 et 1 la prise de valeur)
print("Coefficient de Calsinki-Harabasz :", calinski_harabasz_index)            #Proche de +infini = bien
print("Coefficient de Davies-Bouldin :", davies_bouldin_index)                  #Proche de 0 = bien
