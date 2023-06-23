from library import *

def affichage_nombre_instances_par_classe(df):
    #Affichage du pie chart du nombre d'instance par classe (gravité des accidents)
    gravite_list = df['descr_grav'].tolist()
    gravite_list = [int(x) for x in gravite_list if str(x) != 'nan']
    compte_gravite = [[x,gravite_list.count(x)] for x in set(gravite_list)]
    print(compte_gravite)
    E2=[]
    E3=[]

    for i in compte_gravite:
        E2.append(i[0])
        E3.append(i[1])

    E=[E2,E3]
    pie=px.pie(compte_gravite, values=E[1], names=E[0],title='Répartition gravité accidents',color_discrete_sequence=px.colors.sequential.RdBu)
    pie.update_layout(margin={"r":500})
    pie.show()



def affichage_taille_features(df):
    #Affiche un histogramme des features
    df.hist(bins=110, figsize=(15,10))
    plt.tight_layout()
    plt.show()
   
   
   
df = pd.read_csv("export.csv") 
    
print("Les valeurs cible sont : " + str(df["descr_grav"].unique()))
print("La longeur de la base est de : " + str(len(df)))

#Affichage des types de données
print(df.dtypes)

# affichage_nombre_instances_par_classe(df)

affichage_taille_features(df)
