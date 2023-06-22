import sys
import json
import numpy as np

# Le format d'entrée du tableau des centroides doit etre JSON
def Apprentissage_Non_Supervise(latitude_accident, longitude_accident, tab_centroides):
    # Tableau latitude/longitude pour calcul de distance
    tab_position = [float(latitude_accident), float(longitude_accident)]

    # On transforme le JSON en tableau
    tab_centroides = np.array(json.loads(tab_centroides))

    # On calcule la distance de notre nouveau point avec chaque centroides
    tab_distances = np.linalg.norm(tab_centroides - tab_position, axis=1)

    # Récuperation de l'incdice du centroide le plus proche de notre nouveau point
    indice_centroide_plus_proche = np.argmin(tab_distances)

    # Création de la sortie au format JSON
    cluster_accident = {
        "latitude": latitude_accident,
        "longitude": longitude_accident,
        "cluster": indice_centroide_plus_proche
    }

    return cluster_accident

# Exemples d'utilisation
# python apprentissage_non_supervise.py 45.858844 0.2943506 "[[48.8589, 2.2943], [51.5074, -0.1278], [37.7749, -122.4194]]"
# python apprentissage_non_supervise.py 45.858844 0.2943506 "[[43.610710, 3.751613], [48.169447, -2.221695], [48.804392, 2.320895], [48.559753, 6.792344], [43.996617, -0.668129], [45.705642, 4.708645], [42.196570, 9.201395], [48.018978, 4.505430], [49.061031, 0.566178], [46.537850, -0.175911], [43.953026, 1.404269], [50.488611, 2.851553], [43.311947, 5.580863]]"

resultat = Apprentissage_Non_Supervise(sys.argv[1], sys.argv[2], sys.argv[3])
print(resultat)