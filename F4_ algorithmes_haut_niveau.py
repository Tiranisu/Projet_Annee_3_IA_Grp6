from library import *

def classification_haut_niveau_for_web(data, method):
    match method:
        case "SVM":
            clf = load("out/Support_Vector_Machine.joblib")

        case "RF":
            clf = load("out/Random_Forest.joblib")

        case "MLP":
            clf = load("out/Multilayer_Perceptro.joblib")

        case _:
            print("Il y a un probleme avec la methode de classification rentrer !")
            return
        
    pred = clf.predict(data)
    print(pred[0])
    value={"descr_grav":pred[0].astype(str)}
    
    json_object =json.dumps(value)
    with open("out/predict_haut_niveau.json", "w") as outfile:
        outfile.write(json_object)
    
classification_haut_niveau_for_web([[10,2,1,1,1,14,2,2]], 'RF')