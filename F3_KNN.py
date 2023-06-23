from library import *


#Distance de Hamming
def hamming_dist(x,y):
    return np.sum(x!=y,axis=1)

#Distance Euclidienne
def euclidian_dist(x,y):
    return np.sqrt(np.sum((y - x)**2,axis=1))

#Fonction de KNN de Sklearn
def KNN_sklearn(X_train, y_train, X_test):
    clf = KNeighborsClassifier().fit(X_train, y_train)
    print(clf.predict(X_test)) 
    
#Fonction de KNN from scratch (utilisé pour les Holdout)
def KNN_scratch(X_train, X_test, y_train, y_test, NB_ITER,k):
    X_train_fix=(X_train.loc[:,["age","date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","descr_dispo_secu","descr_type_col"]]).astype(int)
    X_test_fix=(X_test.loc[:,["age","date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","descr_dispo_secu","descr_type_col"]]).astype(int)
    good_predict_hamming=0
    good_predict_euclidian=0


    for test in range(0,NB_ITER):
        distance_val_hamming=hamming_dist(X_train_fix, X_test_fix.iloc[test])
        distance_val_euclidian=euclidian_dist(X_train_fix, X_test_fix.iloc[test])


        good_predict_hamming += y_test.iloc[test] == mode([y for _, y in sorted(zip(distance_val_hamming, y_train))][:k])
        good_predict_euclidian += y_test.iloc[test] == mode([y for _, y in sorted(zip(distance_val_euclidian, y_train))][:k])

    hamming_score = (good_predict_hamming/NB_ITER)
    euclidian_score = (good_predict_euclidian/NB_ITER)
    return hamming_score,euclidian_score

#Fonction de KNN from scratch (utilisé pour les LeaveOneOut)
def predict_scratch(X_train, X_test, y_train, y_test,k):
    X_train_fix=(X_train.loc[:,["age","date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","descr_dispo_secu","descr_type_col"]]).astype(int)
    X_test_fix=(X_test[:])
    X_test_fix["age"]=int(X_test_fix["age"])
    X_test_fix["date"]=int(X_test_fix["date"])
    X_test_fix["descr_cat_veh"]=int(X_test_fix["descr_cat_veh"])
    X_test_fix["descr_agglo"]=int(X_test_fix["descr_agglo"])
    X_test_fix["descr_athmo"]=int(X_test_fix["descr_athmo"])
    X_test_fix["description_intersection"]=int(X_test_fix["description_intersection"])
    X_test_fix["descr_dispo_secu"]=int(X_test_fix["descr_dispo_secu"])
    X_test_fix["descr_type_col"]=int(X_test_fix["descr_type_col"])

    distance_val_euclidian=(np.sum((X_test_fix - X_train_fix)**2,axis=1))**0.5

    predict_euclidian = mode([y for _, y in sorted(zip(distance_val_euclidian, y_train))][:k])

    return predict_euclidian


def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    clf = svm.SVC(decision_function_shape='ovo').fit(X_train, y_train)
    # print(clf.predict(X_test))
    
def Random_Forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    print(clf.predict(X_test))
    
def Multilayer_Perceptron(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    print(clf.predict(X_test))
