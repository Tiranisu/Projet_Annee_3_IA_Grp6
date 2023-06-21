import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from F1 import *
from F3 import *


df = pd.read_csv("export.csv")


#---------------------------------#
#                F1               #
#---------------------------------#

print("Les valeurs cible sont : " + str(df["descr_grav"].unique()))
print("La longeur de la base est de : " + str(len(df)))

affichage_nombre_instances_par_classe(df)

# affichage_taille_features(df)



#---------------------------------#
#                F3               #
#---------------------------------#

#data_reduit = df[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","nom_departement","descr_grav"]]
#data_reduit.loc[:,"date"]=(data_reduit.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]
#data_reduit.loc[:,"age"]=round((data_reduit.loc[:,"age"])/10)
#repartition_donnees(data_reduit)