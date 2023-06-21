import random
import pandas as pd
import math
import numpy as np
from statistics import mode

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

def hamming_dist(x,y):
    return np.sum(x!=y,axis=1)


def euclidian_dist(x,y):
    return np.sqrt(np.sum((y - x)**2,axis=1))


def KNN_sklearn(X_train, y_train, X_test):
    clf = KNeighborsClassifier().fit(X_train, y_train)
    # print(clf.predict(X_test))
    
    
def KNN_scratch(X_train, X_test, y_train, y_test, NB_ITER):
    X_train_fix=(X_train.loc[:,["age","date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","descr_dispo_secu","descr_type_col"]]).astype(int)
    X_test_fix=(X_test.loc[:,["age","date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","descr_dispo_secu","descr_type_col"]]).astype(int)
    good_predict_hamming=0
    good_predict_euclidian=0


    for test in range(0,NB_ITER):
        distance_val_hamming=hamming_dist(X_train_fix, X_test_fix.iloc[test])
        distance_val_euclidian=euclidian_dist(X_train_fix, X_test_fix.iloc[test])


        good_predict_hamming += y_test.iloc[test] == mode([y for _, y in sorted(zip(distance_val_hamming, y_train))][:10])
        good_predict_euclidian += y_test.iloc[test] == mode([y for _, y in sorted(zip(distance_val_euclidian, y_train))][:10])

    hamming_score = (good_predict_hamming/NB_ITER)
    euclidian_score = (good_predict_euclidian/NB_ITER)
    print(hamming_score)
    print(euclidian_score)
    return hamming_score,euclidian_score


def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    clf = svm.SVC(decision_function_shape='ovo').fit(X_train, y_train)
    # print(clf.predict(X_test))
    
def Random_Forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    print(clf.predict(X_test))
    
def Multilayer_Perceptron(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    print(clf.predict(X_test))



def holdout_method(data):
    nb_split = len(data)//5

    data_temp=data
    data_temp.sample()
    
    X_test_1, X_train_1,y_test_1,y_train_1 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    data_temp=shuffle(data_temp)
    X_test_2, X_train_2,y_test_2,y_train_2 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    data_temp=shuffle(data_temp)
    X_test_3, X_train_3,y_test_3,y_train_3 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    data_temp=shuffle(data_temp)
    X_test_4, X_train_4,y_test_4,y_train_4 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
    data_temp=shuffle(data_temp)
    X_test_5, X_train_5,y_test_5,y_train_5 = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]

    NOMBRE_ITER = len(X_test_1)
    score_H_1,score_E_1 = KNN_scratch(X_train_1, X_test_1, y_train_1, y_test_1,NOMBRE_ITER)
    score_H_2,score_E_2 = KNN_scratch(X_train_2, X_test_2, y_train_2, y_test_2,NOMBRE_ITER)
    score_H_3,score_E_3 = KNN_scratch(X_train_3, X_test_3, y_train_3, y_test_3,NOMBRE_ITER)
    score_H_4,score_E_4 = KNN_scratch(X_train_4, X_test_4, y_train_4, y_test_4,NOMBRE_ITER)
    score_H_5,score_E_5 = KNN_scratch(X_train_5, X_test_5, y_train_5, y_test_5,NOMBRE_ITER)

    global_H_score = (score_H_1+score_H_2+score_H_3+score_H_4+score_H_5)/5
    global_E_score = (score_E_1+score_E_2+score_E_3+score_E_4+score_E_5)/5
    print("Global Hamming Score (5-Holdout): "+str(global_H_score))
    print("Global Euclidian Score (5-Holdout): "+str(global_E_score))

    
def loo_method(data):
    df_reduced=pd.DataFrame()
    
    for j in range(0,4):
        df_reduced = pd.concat([df_reduced, df[df['descr_grav'] == j].sample(frac = 0.2)])

    #print(df_reduced)




df = pd.read_csv("export.csv",low_memory=False)
data_reduit = df[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","nom_departement","descr_grav"]]
data_reduit.loc[:,"date"]=(data_reduit.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]
data_reduit.loc[:,"age"]=round((data_reduit.loc[:,"age"])/10)
repartition_donnees(data_reduit)


