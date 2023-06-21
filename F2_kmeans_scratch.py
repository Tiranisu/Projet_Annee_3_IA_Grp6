import numpy as np
from scipy.spatial.distance import cdist 
import pandas as pd
import random as rand
import plotly.express as px
import plotly.graph_objects as go

from F2_evaluation import F2_Silhouette_Coefficient
from F2_evaluation import F2_Calinski_Harabasz_Index
from F2_evaluation import F2_Davies_Bouldin_Index

pourcent = float(1) #pourcentage de la BDD utilisée (pour réduire temps de calcul si besoin)
clusters = int(96) # nombre de clusters

#retourne pour k clusters, k centroids aléatoirement choisi parmis tous les points de data
def initialisationCentroids(k, data):
    centroids = []
    for centroid in range(k):
        centroid = all_vals[rand.randint(0, len(data))] #prise aléatoire du point 
        while any(np.all(l == centroid) for l in centroids): # pour éviter d'avoir deux fois les memes
            #print("deja present")
            centroid = all_vals[rand.randint(0, len(data))]
       
        centroids.append(centroid)
    
    centroids = pd.DataFrame(centroids, columns = data.columns)
    return centroids

#prend en paramètre les datas ([latitude, longitude], le nombre de clusters souhaités, le nombre de répétition(précision))
def kmeansScratch(data,k, nb):
    
    #Choix aléatoire des centroids de départ
    centroids = initialisationCentroids(k, data)
    print(centroids)
    
    #Récupération distance entre les datas et les centroids (eucliedian ou Manhattan)
    distances = cdist(data, centroids ,'euclidean') 
    #distances = cdist(data, centroids ,'cityblock') #ditance cityblock = distance de manhattan = L1

    #Création du tableau points de la taille de la data
    #Contient le numéro du centroid le plus proche
    points = np.array([np.argmin(i) for i in distances])
    #print(np.unique(points))
    #print(points)


    #On répète l'opération nb fois pour augmenter précision
    print('----------')
    for l in range(nb):
        print(round((l/nb)*100, 2)  ,'%') #affichage du pourcentage d'avancement du kmeans
        NCentroids = []
        for j in range(k):
            #Mise à jour des centroids en prenant la moyenne de tous les points y appartenant
            temp_cent = data[points==j].mean(axis=0) 
            NCentroids.append(temp_cent)
 
        centroids = np.array(NCentroids)
                     
        #calcul distance encore (euclidean ou manhattan)
        distances = cdist(data, centroids ,'euclidean')
        #distances = cdist(data, centroids ,'cityblock') #ditance cityblock = distance de manhattan = L1

        #Maj du numéro du centroid le plus proche
        points = np.array([np.argmin(i) for i in distances])
        
        #ajout à la data du numéro du centroid que le point appartient
    data['centroid'] = points
    return data
 
#Import des données du csv
coord = pd.read_csv('export.csv', dtype={0: str})

coord = pd.DataFrame(data=coord, columns= ["latitude", "longitude"]).head(70000)
coord = coord.sample(frac=pourcent).reset_index(drop=True) #frac permet de choisir le pourcentage des datas qu'on veut utiliser
all_vals = coord.values

Ndata = kmeansScratch(coord,clusters,100)

#print(np.unique(Ndata["centroid"])) # on vérifie qu'on a bien 13 clusters différents

Ndata["centroid"] = Ndata["centroid"].astype(str) # converion pour affichage couleur

#Contient tous les centroids 
centroids = Ndata.groupby('centroid').agg('mean').reset_index(drop = True)

print(centroids)

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
print('kmeans from scratch finished')

print('calcul des coefficients')
# Calcul des coeffs
silhouette_coefficient2 = F2_Silhouette_Coefficient(Ndata[['latitude','longitude']] , Ndata['centroid'])
calinski_harabasz_index2 = F2_Calinski_Harabasz_Index(Ndata[['latitude','longitude']] , Ndata['centroid'])
davies_bouldin_index2 = F2_Davies_Bouldin_Index(Ndata[['latitude','longitude']] , Ndata['centroid'])

# Afficher les coeffs
print("Coefficient de silhouette :", silhouette_coefficient2)                    #Proche de 1 = bien (entre -1 et 1 la prise de valeur)
print("Coefficient de Calsinki-Harabasz :", calinski_harabasz_index2)            #Proche de +infini = bien
print("Coefficient de Davies-Bouldin :", davies_bouldin_index2)   
