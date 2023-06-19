import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from F1 import *

df = pd.read_csv("export.csv")


#---------------------------------#
#                F1               #
#---------------------------------#

print("Les valeurs cible sont : " + str(df["descr_grav"].unique()))
print("La longeur de la base est de : " + str(len(df)))

# affichage_nombre_instances_par_classe(df)

# affichage_taille_features(df)



#---------------------------------#
#                F2               #
#---------------------------------#