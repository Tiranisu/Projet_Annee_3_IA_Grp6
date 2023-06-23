from library import *

def hamming_dist(x,y):
    return np.sum(x!=y,axis=1)

def euclidian_dist(x,y):
    return np.sqrt(np.sum((y - x)**2,axis=1))


def KNN_sklearn(X_train, y_train, X_test):
    clf = KNeighborsClassifier().fit(X_train, y_train)
    print(clf.predict(X_test)) 
    
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
    print(hamming_score)
    print(euclidian_score)
    return hamming_score,euclidian_score


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


def holdout_method(data):
    nb_split = len(data)//5
    data_temp=data
    global_H_score=0
    global_E_score=0
    K=10
    for _ in range(5):
        #PREP DONNEE END
        data_temp=shuffle(data_temp)
        X_test, X_train,y_test,y_train = data_temp[nb_split:].drop(["descr_grav"],axis=1), data_temp[:nb_split].drop(["descr_grav"],axis=1), data_temp[nb_split:]["descr_grav"], data_temp[:nb_split]["descr_grav"]
        #PREP DONNEE END

       
        total_size=len(X_test) 
        NOMBRE_ITER = total_size #on peut mettre un nombre à la main pour réduire le temps d'execution
        score_H_temp,score_E_temp = KNN_scratch(X_train, X_test, y_train, y_test,NOMBRE_ITER,K)
        global_H_score+=score_H_temp
        global_E_score+=score_E_temp


    global_H_score/=5
    global_E_score/=5
    print("Global Hamming Score (5-Holdout): "+str(global_H_score))
    print("Global Euclidian Score (5-Holdout): "+str(global_E_score))


    
def loo_method(data):
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
        predict = predict_scratch(X_train, X_test, y_train, y_test,K)

        global_E_score+=(predict==y_test)

    global_E_score/=NOMBRE_ITER
    print("Global Euclidian Score (LeaveOneOut) with k="+str(K)+" : "+str(global_E_score))
    #print(df_reduced)




df = pd.read_csv("export.csv",low_memory=False)

data_reduit = df[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","descr_grav"]]
data_reduit.loc[:,"date"]=(data_reduit.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]
data_reduit.loc[:,"age"]=round((data_reduit.loc[:,"age"])/10)*10

#holdout_method(data_reduit)
loo_method(data_reduit)





