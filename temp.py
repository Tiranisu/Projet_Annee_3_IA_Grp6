import random
import pandas as pd
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle




def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    print("########## Support_Vector_Machine ##########")
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000], 
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }

    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, n_jobs=-1)
    clf.fit(X_train, y_train)


    clf_pred = clf.predict(X_test)

    score = accuracy_score(y_test, clf_pred)
    print(score)

    cm = confusion_matrix(y_test, clf_pred,labels=[0,1],normalize="all")
    print(cm)
    
    cm_display = ConfusionMatrixDisplay(cm).plot()
    
    print("Le meilleur score est  : ", clf.best_score_)
    print("Les meileurs param sont : ", clf.best_params_)
    print("########## End ##########")
    
    
    
def Random_Forest(X_train, X_test, y_train, y_test):
    print("########## Random_Forest ##########")

    param_grid = {
        'n_estimators':[10,100,200],
        # 'max_features':[1,2,3,4,5,6,7],
    }

    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, param_grid, n_jobs=-1)
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)

    score = accuracy_score(y_test, clf_pred)
    print(score)

    cm = confusion_matrix(y_test, clf_pred, labels=[0,1], normalize="all")
    print(cm)
    cm_display = ConfusionMatrixDisplay(cm).plot()
 
    print("Le meilleur score est  : ", clf.best_score_)
    print("Les meileurs param sont : ", clf.best_params_)
    print("########## End ##########")
    
    
    
def Multilayer_Perceptron(X_train, X_test, y_train, y_test):
    print("########## Multilayer_Perceptron ##########")
    
    param_grid = {
        # 'n_estimators':[10,100,200],
        # 'max_features':[1,2,3,4,5,6,7],
    }
    
    mlpc = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, y_train)
    clf = GridSearchCV(mlpc, param_grid, n_jobs=-1)
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)

    score = accuracy_score(y_test, clf_pred)
    print(score)

    cm = confusion_matrix(y_test, clf_pred,labels=[0,1],normalize="all")
    print(cm)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    
    print("Le meilleur score est  : {0:.2f}", clf.best_score_)
    print("Les meileurs param sont : ", clf.best_params_)
    print("########## End ##########")



#print("TOUS LES CIR3 <- cube world")
def repartition_donnees(data):
    nb_split = len(data)//5

    data_temp=data
       
    # X_test_1, X_train_1,y_test_1,y_train_1 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    # data_temp.sample()
    # X_test_2, X_train_2,y_test_2,y_train_2 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    # data_temp.sample()
    # X_test_3, X_train_3,y_test_3,y_train_3 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    # data_temp.sample()
    # X_test_4, X_train_4,y_test_4,y_train_4 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    # data_temp.sample()
    # X_test_5, X_train_5,y_test_5,y_train_5 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]

    for i in range(1,6):
        data_temp=shuffle(data_temp)
        X_test, X_train, y_test, y_train = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
        
        Support_Vector_Machine(X_train, X_test, y_train, y_test)
        Random_Forest(X_train, X_test, y_train, y_test)
        Multilayer_Perceptron(X_train, X_test, y_train, y_test)





df = pd.read_csv("export.csv",low_memory=False)
data_reduit = df[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","descr_grav"]]
data_reduit.loc[:,"date"]=(data_reduit.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]

df_reduced=pd.DataFrame()
for j in range(0,4):
        df_reduced = pd.concat([df_reduced, data_reduit[data_reduit['descr_grav'] == j].sample(frac = 0.1)])

# print(data_reduit.iloc[2].loc["age"])
repartition_donnees(df_reduced)


