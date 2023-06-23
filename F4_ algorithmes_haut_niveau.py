from library import *

def classification_haut_niveau_for_web(data, method):
    match method:
        case "SVM":
            #Récupère le modèle contenu dans le fichier SVM.joblib
            clf = load("out/Support_Vector_Machine.joblib")

        case "RF":
            clf = load("out/Random_Forest.joblib")

        case "MLP":
            clf = load("out/Multilayer_Perceptro.joblib")

        case _:
            print("Il y a un probleme avec la methode de classification rentrer !")
            return
    
    #Prédit la gravité de l'accident avec l'algorithme choisi
    pred = clf.predict(data)
    print(pred[0])
    value={"descr_grav":pred[0].astype(str)}
    
    #Transforme le résultat en json et l'écrit dans le fichier predict_haut_niveau.json
    json_object =json.dumps(value)
    with open("out/predict_haut_niveau.json", "w") as outfile:
        outfile.write(json_object)
    
# Pour tester la fonction, nous retourne bien 1
# classification_haut_niveau_for_web([[10,2,1,1,1,14,2,2]], 'RF')

# Appel de la fonction pour le site web
classification_haut_niveau_for_web([sys.argv[1]], sys.argv[2])