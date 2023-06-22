from library import *



def Support_Vector_Machine_GridSearch(X_train, X_test, y_train, y_test):
    print("########## Support_Vector_Machine_GridSearch ##########")
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 500, 1000, 2000, 5000, 10000], 
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    }

    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, n_jobs=-1)
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)
    
    # print("Le meilleur score est  : ", clf.best_score_)
    # print("Les meileurs param sont : ", clf.best_params_)
    
    pd.DataFrame(clf.cv_results_).to_csv("out/MLP_GridSearch.csv")
    
    print("########## End ##########\n\n")

def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    print("########## Support_Vector_Machine ##########")
    
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)

    accur_score = accuracy_score(y_test, clf_pred)
    prec_score = precision_score(y_test, clf_pred, average='macro', zero_division=1)
    rec_score = recall_score(y_test, clf_pred, average='macro')
    
    # print("Le accurate est de : ", accur_score)
    # print("Le precision est de : ", prec_score)
    # print("Le recall est de : ", rec_score)
    # print("########## End ########## \n\n")
    return accur_score, prec_score, rec_score
 
def Support_Vector_Machine_upgrade(X_train, X_test, y_train, y_test):
    print("########## Support_Vector_Machine_up ##########")
    
    clf = svm.SVC(C=100, gamma=0.001)
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)

    accur_score = accuracy_score(y_test, clf_pred)
    prec_score = precision_score(y_test, clf_pred, average='macro', zero_division=1)
    rec_score = recall_score(y_test, clf_pred, average='macro')

    # print("Le accurate est de : ", accur_score)
    # print("Le precision est de : ", prec_score)
    # print("Le recall est de : ", rec_score)
    # print("########## End ########## \n\n")
    return accur_score, prec_score, rec_score
  
def Support_Vector_Machine_matrix_save(X_train, X_test, y_train, y_test):
    print("########## Support_Vector_Machine_matrix ##########")
    
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    
    dump(clf, 'out/Support_Vector_Machine.joblib') 
    
    cm = confusion_matrix(y_test, clf_pred, normalize="all")
    
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Support Vector Machine :")
    plt.savefig('out/Support_Vector_Machine_matrix.png')
    
   
    
    
def Random_Forest_GridSearch(X_train, X_test, y_train, y_test):
    print("########## Random_Forest_GridSearch ##########")

    #Partie sur la 
    param_grid = {
        'n_estimators':[10, 100, 200],
        'max_features':[1,2,3,4,5,6,7,8],
        'max_depth' :[1, 3 , 5, 10, 15]
    }

    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, param_grid, n_jobs=-1)
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)
 
    # print("Le meilleur score est  : ", clf.best_score_)
    # print("Les meileurs param sont : ", clf.best_params_)
    
    pd.DataFrame(clf.cv_results_).to_csv("out/RF_GridSearch.csv")
    
    # print("########## End ##########\n\n")
        
def Random_Forest(X_train, X_test, y_train, y_test):
    print("########## Random_Forest ########## ")

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)

    accur_score = accuracy_score(y_test, clf_pred)
    prec_score = precision_score(y_test, clf_pred, average='macro', zero_division=1)
    rec_score = recall_score(y_test, clf_pred, average='macro')
 
    # print("Le accurate est de : ", accur_score)
    # print("Le precision est de : ", prec_score)
    # print("Le recall est de : ", rec_score)
    # print("########## End ########## \n\n")
    return accur_score, prec_score, rec_score

def Random_Forest_upgrade(X_train, X_test, y_train, y_test):
    print("########## Random_Forest_up ########## ")

    clf = RandomForestClassifier(max_depth = 5, max_features = 4, n_estimators = 200)
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)

    accur_score = accuracy_score(y_test, clf_pred)
    prec_score = precision_score(y_test, clf_pred, average='macro', zero_division=1)
    rec_score = recall_score(y_test, clf_pred, average='macro')
 
    # print("Le accurate est de : ", accur_score)
    # print("Le precision est de : ", prec_score)
    # print("Le recall est de : ", rec_score)
    # print("########## End ########## \n\n")
    return accur_score, prec_score, rec_score

def Random_Forest_matrix_save(X_train, X_test, y_train, y_test):
    print("########## Random_Forest_matrix ########## ")

    clf = RandomForestClassifier(max_depth = 5, max_features = 4, n_estimators = 200)
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    
    dump(clf, 'out/Random_Forest.joblib')     
    
    cm = confusion_matrix(y_test, clf_pred, normalize="all")
    
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Random Forest :")
    plt.savefig('out/Random_Forest_matrix.png')
    
    
    
    
def Multilayer_Perceptron_GridSearch(X_train, X_test, y_train, y_test):   
    print("########## Multilayer_Perceptron_GridSearch ##########")
    
    param_grid = {
        'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10],
        'max_iter':[500, 1000, 1500],
        'hidden_layer_sizes' :[(50,), (100, 50), (100, 100, 100), (50,50,50), (50,100,50), (100,)]
    }
    
    mlpc = MLPClassifier(random_state=1, max_iter=1000)
    clf = GridSearchCV(mlpc, param_grid, n_jobs=-1)
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    
    # print("Le meilleur score est  : {0:.2f}", clf.best_score_)
    # print("Les meileurs param sont : ", clf.best_params_)
    
    pd.DataFrame(clf.cv_results_).to_csv("out/MP_GridSearch.csv")
    
    print("########## End ##########\n\n") 
    
def Multilayer_Perceptron(X_train, X_test, y_train, y_test):
    print("########## Multilayer_Perceptron ##########")
    
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)

    accur_score = accuracy_score(y_test, clf_pred)
    prec_score = precision_score(y_test, clf_pred, average='macro', zero_division=1)
    rec_score = recall_score(y_test, clf_pred, average='macro')
 
    # print("Le accurate est de : ", accur_score)
    # print("Le precision est de : ", prec_score)
    # print("Le recall est de : ", rec_score)
    # print("########## End ########## \n\n")
    return accur_score, prec_score, rec_score

def Multilayer_Perceptron_upgrade(X_train, X_test, y_train, y_test):
    print("########## Multilayer_Perceptron_up ##########")
    
    clf = MLPClassifier(alpha = 0.0001, max_iter = 500, hidden_layer_sizes=(100, 50))
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)

    accur_score = accuracy_score(y_test, clf_pred)
    prec_score = precision_score(y_test, clf_pred, average='macro', zero_division=1)
    rec_score = recall_score(y_test, clf_pred, average='macro')
 
    # print("Le accurate est de : ", accur_score)
    # print("Le precision est de : ", prec_score)
    # print("Le recall est de : ", rec_score)
    # print("########## End ########## \n\n")
    return accur_score, prec_score, rec_score

def Multilayer_Perceptron_matrix_save(X_train, X_test, y_train, y_test):
    print("########## Multilayer_Perceptron_matrix ##########")
    
    clf = MLPClassifier(alpha = 0.0001, max_iter = 500, hidden_layer_sizes=(100, 50))
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    
    dump(clf, 'out/Multilayer_Perceptro.joblib')    

    cm = confusion_matrix(y_test, clf_pred, normalize="all")
    
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Multilayer Perceptron :")
    plt.savefig('out/Multilayer_Perceptron_matrix.png')
    
    

def affichage():
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


def aux(data_temp):
    data_temp=shuffle(data_temp)
    X_test, X_train, y_test, y_train = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    
    #Lancer l'optimisation des paramètres
    # Support_Vector_Machine_GridSearch(X_train, X_test, y_train, y_test)
    # Random_Forest_GridSearch(X_train, X_test, y_train, y_test)
    # Multilayer_Perceptron_GridSearch(X_train, X_test, y_train, y_test)
    
    # Support_Vector_Machine_matrix_save(X_train, X_test, y_train, y_test)
    # Random_Forest_matrix_save(X_train, X_test, y_train, y_test)  
    # Multilayer_Perceptron_matrix_save(X_train, X_test, y_train, y_test)
    # plt.show()
    
    fusion(X_train, X_test, y_train, y_test)


def fusion(X_train, X_test, y_train, y_test):
    clf1 = svm.SVC(C=100, gamma=0.001)
    clf2 = RandomForestClassifier(max_depth = 5, max_features = 4, n_estimators = 200)
    clf3 = MLPClassifier(alpha = 0.0001, max_iter = 500, hidden_layer_sizes=(100, 50))
    
    #Pour une fusion avec un vote majoritaire
    eclf1 = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('mlp', clf3)], voting='hard', n_jobs=-1, verbose=True) 
    eclf1 = eclf1.fit(X_train, y_train)
    clf_pred1 = eclf1.predict(X_test)
    accur_score1 = accuracy_score(y_test, clf_pred1)
    print("Score de la fusion avec vote majoritaire : ", accur_score1)
    
    clf_pred1 = X_test
    
    #Pour une fusion avec un vote pondéré
    # eclf2 = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('mlp', clf3)], voting='soft', n_jobs=-1, verbose=True)
    # eclf2 = eclf2.fit(X_train, y_train)
    # clf_pred2 = eclf2.predict(X_test)
    # accur_score2 = accuracy_score(y_test, clf_pred2)
    # print("Score de la fusion avec vote pondéré : ", accur_score2)