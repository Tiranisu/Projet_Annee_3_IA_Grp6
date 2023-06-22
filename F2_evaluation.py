from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

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