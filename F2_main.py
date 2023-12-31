import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from F2_reduction_de_dimension import *
from F2_evaluation import *
from F2_kmeans_auto import *
from F2_kmeans_scratch import *

#pourcentage de la BDD utilisée (pour réduire temps de calcul si besoin)
pourcent = float(1)

# nombre de clusters 
clusters = int(13) 

# Lecture du CSV exporté du projet Big Data
data = pd.read_csv('export.csv', dtype={0: str})
data = F2_Format_Date(data) 

# Reduction de dimension
data_tri_MAN = F2_Reduction_Dimension_Manuelle(data)
data_tri_PCA = F2_Reduction_Dimension_PCA(data)

# On créer une base de donnée avec suelement la latitude et la longitude
data_pos = F2_Preparation_Data_Pour_Classification(data, 70000)
#frac permet de choisir le pourcentage des datas qu'on veut utiliser
data_pos = data_pos.sample(frac=pourcent).reset_index(drop=True) 

# On effectue k-Means et on récupere les labels de toute nos latitude/longitude
data_kMeans = F2_Affichage_kMeans_Auto(data_pos, clusters)
labels_kMeans = data_kMeans[0]
centroids_kMeans = data_kMeans[1]
print(centroids_kMeans)


print('kmeans fini')

print('calcul des coefficients pour kmeans')
# Calcul des coeffs
silhouette_coefficient = F2_Silhouette_Coefficient(data_pos, labels_kMeans)
calinski_harabasz_index = F2_Calinski_Harabasz_Index(data_pos, labels_kMeans)
davies_bouldin_index = F2_Davies_Bouldin_Index(data_pos, labels_kMeans)

# Afficher les coeffs
print("Coefficient de silhouette :", silhouette_coefficient)                    #Proche de 1 = bien (entre -1 et 1 la prise de valeur)
print("Coefficient de Calsinki-Harabasz :", calinski_harabasz_index)            #Proche de +infini = bien
print("Coefficient de Davies-Bouldin :", davies_bouldin_index)                  #Proche de 0 = bien

##Kmean from scratch

#Import des données du csv
coord = data #on reprend les données brut du CSV de la partie précédent

#coord = pd.DataFrame(data=coord, columns= ["latitude", "longitude"]).head(70000) #récupère les 70 000 premières lignes de lat et lon
coord = F2_Preparation_Data_Pour_Classification(coord, 70000)
#frac permet de choisir le pourcentage des datas qu'on veut utiliser
coord = coord.sample(frac=pourcent).reset_index(drop=True) 

#on effectue le kmeans from scratch avec 100 itérations (précision) et nb clusters
Ndata = kmeansScratch(coord,clusters,100)

#print(np.unique(Ndata["centroid"])) # on vérifie qu'on a bien 13 clusters différents

# converion en tring pour affichage couleur
Ndata["centroid"] = Ndata["centroid"].astype(str) 

#Contient tous les centroids 
centroids = Ndata.groupby('centroid').agg('mean').reset_index(drop = True)

#print(centroids)


#affichage sur une carte
fig = px.scatter_geo(Ndata, locationmode='country names',
                         lat='latitude', lon='longitude',
                         projection='natural earth', color='centroid',
                         color_discrete_sequence=px.colors.qualitative.Alphabet)
fig.update_geos(fitbounds="locations", showcountries=True)
fig.update_layout(title='Clusters accident en France en 2019 (K-Means scratch)')
#ajout des centroids
fig.add_trace(go.Scattergeo(locationmode='country names',
                                lat=centroids.iloc[:,0], lon=centroids.iloc[:,1],
                                marker=dict(size=15,color='red')))
fig.show()
print('kmeans from scratch fini')


print('calcul des coefficients du from scratch')
# Calcul des coeffs
silhouette_coefficient2 = F2_Silhouette_Coefficient(Ndata[['latitude','longitude']] , Ndata['centroid'])
calinski_harabasz_index2 = F2_Calinski_Harabasz_Index(Ndata[['latitude','longitude']] , Ndata['centroid'])
davies_bouldin_index2 = F2_Davies_Bouldin_Index(Ndata[['latitude','longitude']] , Ndata['centroid'])

# Afficher les coeffs
print("Coefficient de silhouette :", silhouette_coefficient2)                    #Proche de 1 = bien (entre -1 et 1 la prise de valeur)
print("Coefficient de Calsinki-Harabasz :", calinski_harabasz_index2)            #Proche de +infini = bien
print("Coefficient de Davies-Bouldin :", davies_bouldin_index2)   
