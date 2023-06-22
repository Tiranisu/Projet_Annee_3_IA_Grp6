import numpy as np
from scipy.spatial.distance import cdist 
import pandas as pd
import random as rand
import plotly.express as px
import plotly.graph_objects as go

from F2_evaluation import *

#retourne pour k clusters, k centroids aléatoirement choisi parmis tous les points de data
def initialisationCentroids(k, data):
    centroids = []
    all_vals = data.values
    for centroid in range(k):
        #prise aléatoire du point 
        centroid = all_vals[rand.randint(0, len(data))]
        # pour éviter d'avoir deux fois les memes
        while any(np.all(l == centroid) for l in centroids): 
            #print("deja present")
            centroid = all_vals[rand.randint(0, len(data))]

        centroids.append(centroid)
    
    centroids = pd.DataFrame(centroids, columns = data.columns)
    return centroids

#prend en paramètre les datas ([latitude, longitude], le nombre de clusters souhaités, le nombre de répétition(précision))
def kmeansScratch(data,k, nb):
    
    #Choix aléatoire des centroids de départ
    centroids = initialisationCentroids(k, data)
    
    #Récupération distance entre les datas et les centroids (eucliedian ou Manhattan)
    distances = cdist(data, centroids ,'euclidean') 
    #distances = cdist(data, centroids ,'cityblock') #ditance cityblock = distance de manhattan = L1

    #Création du tableau points de la taille de la data
    #Contient le numéro du centroid le plus proche
    points = np.array([np.argmin(i) for i in distances])

    #On répète l'opération nb fois pour augmenter précision
    print('----------')
    for l in range(nb):
        #affichage du pourcentage d'avancement du kmeans
        print(round((l/nb)*100, 2)  ,'% de kmeans from scratch') 
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
 