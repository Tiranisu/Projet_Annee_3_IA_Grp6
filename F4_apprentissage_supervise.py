from library import *


def knn_for_web(csv,data,k):
    data_reduit_ = csv[["date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","age","descr_dispo_secu","descr_type_col","descr_grav"]]
    data_reduit_.loc[:,"date"]=(data_reduit_.loc[:,"date"].str.split(" ",expand=True)[1]).str.split(":",expand=True)[0]
    data_reduit_.loc[:,"age"]=round((data_reduit_.loc[:,"age"])/10)*10

    data_reduit_=shuffle(data_reduit_)

    y_train=data_reduit_[:]["descr_grav"]
    X_train=data_reduit_[:].drop(["descr_grav"],axis=1)

    X_train_fix=(X_train.loc[:,["age","date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","descr_dispo_secu","descr_type_col"]]).astype(int)
    

    distance_val_euclidian=np.sqrt(np.sum((data - X_train_fix)**2,axis=1))

    predict_euclidian = mode([y for _, y in sorted(zip(distance_val_euclidian, y_train))][:k])
    return predict_euclidian
    

df = pd.read_csv(sys.argv[0]+".csv",low_memory=False)

#"age","date","descr_cat_veh","descr_agglo","descr_athmo","description_intersection","descr_dispo_secu","descr_type_col"

#predict = knn_for_web(CSV,[DATA],K)
#predict=knn_for_web(df,[10,2,1,1,1,14,2,2],10)
predict=knn_for_web(df,[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]],sys.argv[9])



value={"predict":predict}
json_object =json.dumps(value)
with open("out/predict.json", "w") as outfile:
    outfile.write(json_object)
