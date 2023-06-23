from library import *
from F3_KNN import *
from F3_algorithmes_haut_niveau import *



#Initialisation des variables pour les variables globals
sum_accur_score_svm=0
sum_accur_score_rf=0
sum_accur_score_mlp=0

sum_prec_score_svm=0
sum_prec_score_rf=0
sum_prec_score_mlp=0

sum_rec_score_svm=0
sum_rec_score_rf=0
sum_rec_score_mlp=0

sum_accur_score_svm_upgrade=0
sum_accur_score_rf_upgrade=0
sum_accur_score_mlp_upgrade=0

sum_prec_score_svm_upgrade=0
sum_prec_score_rf_upgrade=0
sum_prec_score_mlp_upgrade=0

sum_rec_score_svm_upgrade=0
sum_rec_score_rf_upgrade=0
sum_rec_score_mlp_upgrade=0



def holdout_method(data):
    #Répartition de la base en Train et Test (80/20)
    nb_split = len(data)//5
    
    data_temp=data
    global_H_score=0
    global_E_score=0
    K=10
    
    #Préparation des variables globales
    global sum_accur_score_svm
    global sum_accur_score_rf
    global sum_accur_score_mlp
    
    global sum_prec_score_svm
    global sum_prec_score_rf
    global sum_prec_score_mlp
    
    global sum_rec_score_svm
    global sum_rec_score_rf
    global sum_rec_score_mlp
    
    global sum_accur_score_svm_upgrade
    global sum_accur_score_rf_upgrade
    global sum_accur_score_mlp_upgrade
    
    global sum_prec_score_svm_upgrade
    global sum_prec_score_rf_upgrade
    global sum_prec_score_mlp_upgrade
    
    global sum_rec_score_svm_upgrade
    global sum_rec_score_rf_upgrade
    global sum_rec_score_mlp_upgrade
    
    
    for _ in range(5):
        #PREP DONNEE END
        #Mélange des données
        data_temp=shuffle(data_temp)
        #Répartition des données
        X_test, X_train,y_test,y_train = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
        #PREP DONNEE END


        #KNN
        total_size=len(X_test) 
        NOMBRE_ITER = total_size #on peut mettre un nombre à la main pour réduire le temps d'execution
        score_H_temp,score_E_temp = KNN_scratch(X_train, X_test, y_train, y_test,NOMBRE_ITER,K)
        global_H_score+=score_H_temp
        global_E_score+=score_E_temp
        
        
        
        
        #Algo haut niveau
        #Test des modèles normaux
        sum_accur_score_svm_temp, sum_prec_score_svm_temp, sum_rec_score_svm_temp = Support_Vector_Machine(X_train, X_test, y_train, y_test)
        sum_accur_score_rf_temp, sum_prec_score_rf_temp, sum_rec_score_rf_temp = Random_Forest(X_train, X_test, y_train, y_test)
        sum_accur_score_mlp_temp, sum_prec_score_mlp_temp, sum_rec_score_mlp_temp = Multilayer_Perceptron(X_train, X_test, y_train, y_test)
        
        #Test des modèles upgradés
        sum_accur_score_svm_upgrade_temp, sum_prec_score_svm_upgrade_temp, sum_rec_score_svm_upgrade_temp = Support_Vector_Machine_upgrade(X_train, X_test, y_train, y_test)
        sum_accur_score_rf_upgrade_temp, sum_prec_score_rf_upgrade_temp, sum_rec_score_rf_upgrade_temp = Random_Forest_upgrade(X_train, X_test, y_train, y_test)
        sum_accur_score_mlp_upgrade_temp, sum_prec_score_mlp_upgrade_temp, sum_rec_score_mlp_upgrade_temp = Multilayer_Perceptron_upgrade(X_train, X_test, y_train, y_test)


        #Afin de contourner l'erruer : SyntaxError: 'tuple' is an illegal expression for augmented assignment
        #Ajoute le score actuel au score total
        sum_accur_score_svm += sum_accur_score_svm_temp
        sum_prec_score_svm += sum_prec_score_svm_temp
        sum_rec_score_svm += sum_rec_score_svm_temp
        
        sum_accur_score_rf += sum_accur_score_rf_temp
        sum_prec_score_rf += sum_prec_score_rf_temp 
        sum_rec_score_rf += sum_rec_score_rf_temp   
        
        sum_accur_score_mlp += sum_accur_score_mlp_temp 
        sum_prec_score_mlp += sum_prec_score_mlp_temp   
        sum_rec_score_mlp += sum_rec_score_mlp_temp 
        
        sum_accur_score_svm_upgrade += sum_accur_score_svm_upgrade_temp
        sum_prec_score_svm_upgrade += sum_prec_score_svm_upgrade_temp
        sum_rec_score_svm_upgrade += sum_rec_score_svm_upgrade_temp
        
        sum_accur_score_rf_upgrade += sum_accur_score_rf_upgrade_temp
        sum_prec_score_rf_upgrade += sum_prec_score_rf_upgrade_temp 
        sum_rec_score_rf_upgrade += sum_rec_score_rf_upgrade_temp   
        
        sum_accur_score_mlp_upgrade += sum_accur_score_mlp_upgrade_temp 
        sum_prec_score_mlp_upgrade += sum_prec_score_mlp_upgrade_temp   
        sum_rec_score_mlp_upgrade += sum_rec_score_mlp_upgrade_temp 
        print("\n")
        


    global_H_score/=5
    global_E_score/=5
    print("Global Hamming Score (5-Holdout): "+str(global_H_score))
    print("Global Euclidian Score (5-Holdout): "+str(global_E_score))
    
    #Affichage des résultats des algos
    print("*---------------------------------*")
    print("Moyenne des scores pour SVM : ", sum_accur_score_svm/5)
    print("Moyenne des scores pour RF : ", sum_accur_score_rf/5)
    print("Moyenne des scores pour MLP : ", sum_accur_score_mlp/5)
    print("*---------------------------------*")
    print("Moyenne des scores pour SVM upgrade : ", sum_accur_score_svm_upgrade/5)
    print("Moyenne des scores pour RF upgrade : ", sum_accur_score_rf_upgrade/5)
    print("Moyenne des scores pour MLP upgrade : ", sum_accur_score_mlp_upgrade/5)
    print("*---------------------------------*")
    print("*---------------------------------*")
    print("Moyenne des precisions pour SVM : ", sum_prec_score_svm/5)   
    print("Moyenne des precisions pour RF : ", sum_prec_score_rf/5) 
    print("Moyenne des precisions pour MLP : ", sum_prec_score_mlp/5)   
    print("*---------------------------------*")
    print("Moyenne des precisions pour SVM upgrade : ", sum_prec_score_svm_upgrade/5)
    print("Moyenne des precisions pour RF upgrade : ", sum_prec_score_rf_upgrade/5)
    print("Moyenne des precisions pour MLP upgrade : ", sum_prec_score_mlp_upgrade/5)
    print("*---------------------------------*")
    print("*---------------------------------*")
    print("Moyenne des rappels pour SVM : ", sum_rec_score_svm/5)
    print("Moyenne des rappels pour RF : ", sum_rec_score_rf/5)
    print("Moyenne des rappels pour MLP : ", sum_rec_score_mlp/5)
    print("*---------------------------------*")
    print("Moyenne des rappels pour SVM upgrade : ", sum_rec_score_svm_upgrade/5)   
    print("Moyenne des rappels pour RF upgrade : ", sum_rec_score_rf_upgrade/5) 
    print("Moyenne des rappels pour MLP upgrade : ", sum_rec_score_mlp_upgrade/5)   
    print("*---------------------------------*")
    
    
    data_temp=shuffle(data_temp)
    X_test, X_train, y_test, y_train = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    
    #Lancer l'optimisation des paramètres
    # Support_Vector_Machine_GridSearch(X_train, X_test, y_train, y_test)
    # Random_Forest_GridSearch(X_train, X_test, y_train, y_test)
    # Multilayer_Perceptron_GridSearch(X_train, X_test, y_train, y_test)
    
    Support_Vector_Machine_matrix_save(X_train, X_test, y_train, y_test)
    Random_Forest_matrix_save(X_train, X_test, y_train, y_test)  
    Multilayer_Perceptron_matrix_save(X_train, X_test, y_train, y_test)
    plt.show()
    
    #Lanceemnt du test de fusion
    fusion(X_train, X_test, y_train, y_test)


    
def loo_method(data):
    #Préparation des variables globales
    global sum_accur_score_svm
    global sum_accur_score_rf
    global sum_accur_score_mlp
    
    global sum_prec_score_svm
    global sum_prec_score_rf
    global sum_prec_score_mlp
    
    global sum_rec_score_svm
    global sum_rec_score_rf
    global sum_rec_score_mlp
    
    global sum_accur_score_svm_upgrade
    global sum_accur_score_rf_upgrade
    global sum_accur_score_mlp_upgrade
    
    global sum_prec_score_svm_upgrade
    global sum_prec_score_rf_upgrade
    global sum_prec_score_mlp_upgrade
    
    global sum_rec_score_svm_upgrade
    global sum_rec_score_rf_upgrade
    global sum_rec_score_mlp_upgrade
    
    df_reduced=pd.DataFrame()

    for j in range(0,4):
        df_reduced = pd.concat([df_reduced, data[data['descr_grav'] == j].sample(frac = 0.2)])

    size_loo = len(df_reduced)
    NOMBRE_ITER = 100 #on peut mettre un nombre à la main pour réduire le temps d'execution

    data_temp=df_reduced
    global_H_score=0
    global_E_score=0
    K=10
    for i in range(NOMBRE_ITER):
        data_temp=shuffle(data_temp)
        data_train=data_temp[:].drop(data_temp.iloc[[i]].index)

        X_test=data_temp[:].drop(["descr_grav"],axis=1).iloc[i]
        y_test=data_temp[:]["descr_grav"].iloc[i]
        y_train=data_train[:]["descr_grav"]
        X_train=data_train[:].drop(["descr_grav"],axis=1)
        X_test_reshape = np.reshape(X_test.to_numpy(), (-1,8))  
        y_test_reshape = np.reshape(y_test, (-1,1))

        #KNN
        predict = predict_scratch(X_train, X_test, y_train, y_test,K)

        global_E_score+=(predict==y_test)
        
        
        #Algo haut niveau
        #Test des modèles normaux
        sum_accur_score_svm_temp, sum_prec_score_svm_temp, sum_rec_score_svm_temp = Support_Vector_Machine(X_train, X_test_reshape, y_train, y_test_reshape)
        sum_accur_score_rf_temp, sum_prec_score_rf_temp, sum_rec_score_rf_temp = Random_Forest(X_train, X_test_reshape, y_train, y_test_reshape)
        sum_accur_score_mlp_temp, sum_prec_score_mlp_temp, sum_rec_score_mlp_temp = Multilayer_Perceptron(X_train, X_test_reshape, y_train, y_test_reshape)
        
        #Test des modèles upgradés
        sum_accur_score_svm_upgrade_temp, sum_prec_score_svm_upgrade_temp, sum_rec_score_svm_upgrade_temp = Support_Vector_Machine_upgrade(X_train, X_test_reshape, y_train, y_test_reshape)
        sum_accur_score_rf_upgrade_temp, sum_prec_score_rf_upgrade_temp, sum_rec_score_rf_upgrade_temp = Random_Forest_upgrade(X_train, X_test_reshape, y_train, y_test_reshape)
        sum_accur_score_mlp_upgrade_temp, sum_prec_score_mlp_upgrade_temp, sum_rec_score_mlp_upgrade_temp = Multilayer_Perceptron_upgrade(X_train, X_test_reshape, y_train, y_test_reshape)


        #Afin de contourner l'erruer : SyntaxError: 'tuple' is an illegal expression for augmented assignment
        #Ajoute le score actuel au score total
        sum_accur_score_svm += sum_accur_score_svm_temp
        sum_prec_score_svm += sum_prec_score_svm_temp
        sum_rec_score_svm += sum_rec_score_svm_temp
        
        sum_accur_score_rf += sum_accur_score_rf_temp
        sum_prec_score_rf += sum_prec_score_rf_temp 
        sum_rec_score_rf += sum_rec_score_rf_temp   
        
        sum_accur_score_mlp += sum_accur_score_mlp_temp 
        sum_prec_score_mlp += sum_prec_score_mlp_temp   
        sum_rec_score_mlp += sum_rec_score_mlp_temp 
        
        sum_accur_score_svm_upgrade += sum_accur_score_svm_upgrade_temp
        sum_prec_score_svm_upgrade += sum_prec_score_svm_upgrade_temp
        sum_rec_score_svm_upgrade += sum_rec_score_svm_upgrade_temp
        
        sum_accur_score_rf_upgrade += sum_accur_score_rf_upgrade_temp
        sum_prec_score_rf_upgrade += sum_prec_score_rf_upgrade_temp 
        sum_rec_score_rf_upgrade += sum_rec_score_rf_upgrade_temp   
        
        sum_accur_score_mlp_upgrade += sum_accur_score_mlp_upgrade_temp 
        sum_prec_score_mlp_upgrade += sum_prec_score_mlp_upgrade_temp   
        sum_rec_score_mlp_upgrade += sum_rec_score_mlp_upgrade_temp 
        
        print("\n")

    global_E_score/=NOMBRE_ITER
    print("Global Euclidian Score (LeaveOneOut) with k="+str(K)+" : "+str(global_E_score))
    #print(df_reduced)
    
    #Affichage des résultats des algos
    print("*---------------------------------*")
    print("Moyenne des scores pour SVM : ", sum_accur_score_svm/NOMBRE_ITER)
    print("Moyenne des scores pour RF : ", sum_accur_score_rf/NOMBRE_ITER)
    print("Moyenne des scores pour MLP : ", sum_accur_score_mlp/NOMBRE_ITER)
    print("*---------------------------------*")
    print("Moyenne des scores pour SVM upgrade : ", sum_accur_score_svm_upgrade/NOMBRE_ITER)
    print("Moyenne des scores pour RF upgrade : ", sum_accur_score_rf_upgrade/NOMBRE_ITER)
    print("Moyenne des scores pour MLP upgrade : ", sum_accur_score_mlp_upgrade/NOMBRE_ITER)
    print("*---------------------------------*")
    print("*---------------------------------*")
    print("Moyenne des precisions pour SVM : ", sum_prec_score_svm/NOMBRE_ITER)   
    print("Moyenne des precisions pour RF : ", sum_prec_score_rf/NOMBRE_ITER) 
    print("Moyenne des precisions pour MLP : ", sum_prec_score_mlp/NOMBRE_ITER)   
    print("*---------------------------------*")
    print("Moyenne des precisions pour SVM upgrade : ", sum_prec_score_svm_upgrade/NOMBRE_ITER)
    print("Moyenne des precisions pour RF upgrade : ", sum_prec_score_rf_upgrade/NOMBRE_ITER)
    print("Moyenne des precisions pour MLP upgrade : ", sum_prec_score_mlp_upgrade/NOMBRE_ITER)
    print("*---------------------------------*")
    print("*---------------------------------*")
    print("Moyenne des rappels pour SVM : ", sum_rec_score_svm/NOMBRE_ITER)
    print("Moyenne des rappels pour RF : ", sum_rec_score_rf/NOMBRE_ITER)
    print("Moyenne des rappels pour MLP : ", sum_rec_score_mlp/NOMBRE_ITER)
    print("*---------------------------------*")
    print("Moyenne des rappels pour SVM upgrade : ", sum_rec_score_svm_upgrade/NOMBRE_ITER)   
    print("Moyenne des rappels pour RF upgrade : ", sum_rec_score_rf_upgrade/NOMBRE_ITER) 
    print("Moyenne des rappels pour MLP upgrade : ", sum_rec_score_mlp_upgrade/NOMBRE_ITER)   
    print("*---------------------------------*")
    
    data_temp=shuffle(data_temp)
    data_train=data_temp[:].drop(data_temp.iloc[[1]].index)

    X_test=data_temp[:].drop(["descr_grav"],axis=1).iloc[1]
    y_test=data_temp[:]["descr_grav"].iloc[1]
    y_train=data_train[:]["descr_grav"]
    X_train=data_train[:].drop(["descr_grav"],axis=1)
    X_test_reshape = np.reshape(X_test.to_numpy(), (-1,8))
    y_test_reshape = np.reshape(y_test, (-1,1))
    
    #Lancer l'optimisation des paramètres
    # Support_Vector_Machine_GridSearch(X_train, X_test_reshape, y_train, y_test_reshape)
    # Random_Forest_GridSearch(X_train, X_test_reshape, y_train, y_test_reshape)
    # Multilayer_Perceptron_GridSearch(X_train, X_test_reshape, y_train, y_test_reshape)
    
    Support_Vector_Machine_matrix_save(X_train, X_test_reshape, y_train, y_test_reshape)
    Random_Forest_matrix_save(X_train, X_test_reshape, y_train, y_test_reshape)  
    Multilayer_Perceptron_matrix_save(X_train, X_test_reshape, y_train, y_test_reshape)
    plt.show()
    
    fusion(X_train, X_test_reshape, y_train, y_test_reshape)







df = pd.read_csv("export.csv",low_memory=False)

data_reduit = df[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","descr_grav"]]
data_reduit.loc[:,"date"]=(data_reduit.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]
data_reduit.loc[:,"age"]=round((data_reduit.loc[:,"age"])/10)*10

#Reduction de la base à 5% de sa taille de manière aquivalente pour les 4 classes
df_reduced=pd.DataFrame()
for j in range(0,4):
        df_reduced = pd.concat([df_reduced, data_reduit[data_reduit['descr_grav'] == j].sample(frac = 0.05)])


#Méthode avec 5% de la base
holdout_method(df_reduced)
loo_method(df_reduced)

#Methode avec toute la base
holdout_method(data_reduit)
loo_method(data_reduit)





