import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from F1_main import *
from F3_main import *





#---------------------------------#
#                F1               #
#---------------------------------#




#---------------------------------#
#                F3               #
#---------------------------------#

#data_reduit = df[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","nom_departement","descr_grav"]]
#data_reduit.loc[:,"date"]=(data_reduit.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]
#data_reduit.loc[:,"age"]=round((data_reduit.loc[:,"age"])/10)
#repartition_donnees(data_reduit)