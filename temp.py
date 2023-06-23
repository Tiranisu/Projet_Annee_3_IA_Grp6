from library import *

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





def repartition_donnees(data):
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
    
    nb_split = len(data)//5
    data_temp=data


    for i in range(1,6):
        data_temp=shuffle(data_temp)
        X_test, X_train, y_test, y_train = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
        
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
    
    #Affichage des résultats
    affichage()
    
    
    




df = pd.read_csv("export.csv",low_memory=False)
data_reduit = df[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","descr_grav"]]
data_reduit.loc[:,"date"]=(data_reduit.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]

df_reduced=pd.DataFrame()
for j in range(0,4):
        df_reduced = pd.concat([df_reduced, data_reduit[data_reduit['descr_grav'] == j].sample(frac = 0.2)])

repartition_donnees(df_reduced)


