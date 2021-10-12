from typing import NamedTuple

import os
import glob
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def extract_functional(prosodic):
    prosodic = np.array(prosodic)
    stats = []
    stats.append(prosodic.mean())
    stats.append(prosodic.max())
    stats.append(len(prosodic))
    stats.append(prosodic.std())
    stats.append(np.median(prosodic))
    stats.append(np.percentile(prosodic,25))
    stats.append(np.percentile(prosodic,75))
    derivate = np.mean([prosodic[i+1] - prosodic[i] for i in range(len(prosodic)-1)])
    stats.append(derivate)
    
    return stats

def repartition_label(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    print("repartion label = ", dict(zip(unique.astype(int), counts)))
    print("repartition label en % = ", dict(zip(unique.astype(int), (counts/counts.sum()*100).astype("float16"))))


def create_dataset_from_path(path, labels, all):
    """
    Extract features from file of path given and create a dataset
    all: 1 to take all characteristic
         0 to take voiced
         -1 to take unvoiced
    """
    
    files = [file for file in os.listdir(path) if not file.endswith(".mat")]
    
    datasets = []
    
    #prosodic extraction
    
    if path.split('\\')[-1] == 'BabyEars_WAV':
        step_nb = 4
    else:
        step_nb = 3
        
    for i in range(0, len(files), step_nb):
        label = files[i][6:8]
        if(label not in labels):
            continue
        
        label = labels.index(label)
        with open(path+"\\"+files[i], 'r') as file1:
            en = np.array([int(f.split()[1]) for f in file1])
        
        with open(path+"\\"+files[i+1], 'r') as file2:
            f0 = np.array([int(f.split()[1]) for f in file2])
            
        if all == -1:
            mask = (f0 == 0)
            f0_unvoiced = f0[mask]
            en_unvoiced = en[mask]
            
            en_stats = extract_functional(en_unvoiced)
            f0_stats = extract_functional(f0_unvoiced)
        
        elif all == 0:
            mask = (f0 != 0)
            f0_voiced = f0[mask]
            en_voiced = en[mask]
            
            en_stats = extract_functional(en_voiced)
            f0_stats = extract_functional(f0_voiced)
        
        else:
            en_stats = extract_functional(en)
            f0_stats = extract_functional(f0)
        
        datasets.append([label] + en_stats + f0_stats)  
         
    file1.close(); file2.close()   
    return np.array(datasets)

def pipeline(Model, X_train, y_train): 
    pipeline = make_pipeline(StandardScaler(), Model)
    pipeline.fit(X_train, y_train)
    return pipeline

def datasets_fusion(path, all):
    babyears = create_dataset_from_path(path + '\Databases\BabyEars_WAV', ['ap', 'pr', 'at'], all = all)
    kismet = create_dataset_from_path(path + '\Databases\Kismet_data', ['ap', 'pw', 'at'], all = all)
    
    datasets = np.vstack((kismet, babyears))
    return datasets

def test_BabyEars(path):
    
    print("===============Test sur BabyEars Total=============")
    
    print("\n===Test on 2 labels===")
    dataset = create_dataset_from_path(path + '\Databases\BabyEars_WAV', ['ap', 'pr'], all = 1)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%")
    
    print("\n===Test on 3 labels===")
    
    dataset = create_dataset_from_path(path + '\Databases\BabyEars_WAV', ['ap', 'pr', 'at'], all = 1)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))

def test_Kismet(path):
    print("===============Test sur Kismet Total=============")
    
    print("\n===Test on 2 labels===")
    dataset = create_dataset_from_path(path + '\Databases\Kismet_data', ['ap', 'pw'], all = 1)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%")
    
    print("\n===Test on 3 labels===")
    
    dataset = create_dataset_from_path(path + '\Databases\Kismet_data', ['ap', 'pw', 'at'], all = 1)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))
    

def test_Kismet_voiced_unvoiced(path, unvoiced):
    
    print("===============Test sur Kismet Voiced=============")
    
    print("\n===Test on 2 labels===")
    dataset = create_dataset_from_path(path + '\Databases\Kismet_data', ['at', 'pw'], all = 0)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    
    print("\n===Test on 3 labels===")
    
    dataset = create_dataset_from_path(path + '\Databases\Kismet_data', ['ap', 'pw', 'at'], all = 0)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))
    
    if unvoiced:
        print("===============Test sur Kismet Unvoiced=============")
        
        print("\n===Test on 2 labels===")
        dataset = create_dataset_from_path(path + '\Databases\Kismet_data', ['at', 'pw'], all = -1)
        X,y = dataset[:,1:], dataset[:,0]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
        print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
        repartition_label(y_train)
        
        knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
        print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
        
        svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
        print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
        
        print("\n===Test on 3 labels===")
    
        dataset = create_dataset_from_path(path + '\Databases\Kismet_data', ['ap', 'pw', 'at'], all = -1)
        X,y = dataset[:,1:], dataset[:,0]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
        print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
        repartition_label(y_train)
        
        knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
        print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
        print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
        
        svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
        print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
        print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))
        
    
def test_BabyEars_voiced_unvoiced(path, unvoiced):
    
    print("===============Test sur BabyEars Voiced=============")
    
    print("\n===Test on 2 labels===")
    dataset = create_dataset_from_path(path + '\Databases\BabyEars_WAV', ['at', 'pr'], all = 0)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    
    print("\n===Test on 3 labels===")
    
    dataset = create_dataset_from_path(path + '\Databases\BabyEars_WAV', ['ap', 'pr', 'at'], all = 0)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))
    
    
    if unvoiced:
        print("===============Test sur BabyEars Unvoiced=============")
        
        print("\n===Test on 2 labels===")
        dataset = create_dataset_from_path(path + '\Databases\BabyEars_WAV', ['at', 'pr'], all = -1)
        X,y = dataset[:,1:], dataset[:,0]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
        print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
        repartition_label(y_train)
        
        knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
        print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
        
        svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
        print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
        
        print("\n===Test on 3 labels===")
    
        dataset = create_dataset_from_path(path + '\Databases\BabyEars_WAV', ['ap', 'pr', 'at'], all = -1)
        X,y = dataset[:,1:], dataset[:,0]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
        print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
        repartition_label(y_train)
        
        knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
        print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
        print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
        
        svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
        print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
        print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))
    
def test_fusion(path):
    
    print("===============Test sur Fusion Total Intra=============")
    
    
    print("\n===Test on 3 labels===")
    
    dataset = datasets_fusion(path, all = 1)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))
    
    print("===============Test sur Fusion Voiced Intra=============")
    
    
    print("\n===Test on 3 labels===")
    
    dataset = datasets_fusion(path, all = 0)
    X,y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)
    print(f"repartion train = {len(X_train)}, test = {len(X_test)}")
    repartition_label(y_train)
    
    knn_model = pipeline(KNeighborsClassifier(), X_train, y_train)
    print(f"Knn 5 accuracy = {knn_model.score(X_test, y_test)*100:0.2f}%")
    print("confusion matrix = \n", confusion_matrix(y_test, knn_model.predict(X_test)))
    
    svm_model = pipeline(SVC(gamma='auto'), X_train, y_train)
    print(f"Svm accuracy = {svm_model.score(X_test,y_test)*100:0.2f}%\n")
    print("confusion matrix = \n", confusion_matrix(y_test, svm_model.predict(X_test)))
    
if __name__ == '__main__':
    
    path_file = os.path.realpath(__file__)
    real_path = os.path.dirname(path_file)
    
    test_Kismet_voiced_unvoiced(real_path, unvoiced = True)
    
    test_Kismet(real_path)
    
    test_BabyEars_voiced_unvoiced(real_path, unvoiced = False)
    
    test_BabyEars(real_path)
    
    test_fusion(real_path)
    
    
    """
    
    send to lebrun@isir.upmc.fr
    
    Exercice 1:
    
    1. Voire fonction create_dataset_from_path
    
    2. Voire fonction extract_functional
    
    3. En regardant les fonctionnelles sur les parties voisés ou non voisés,
    nous voyons que pour la partie non voisés, nous perdons de l'information
    sur nos features, puisque les fonctionnelles de la fréquence fondamentale
    seront toujours nulles (puisque f0 = 0 partout). Contrairement aux parties
    voisés qui elles contiennent les informations nécessaires pour les fonctionnelles
    de f0. Les segments voisés sont donc les plus adéquats pour le traitement.
    
    4. voir chaque fonction de test. La démarche est toujours la même en utilisant
    la fonction train_test_split de scikit-learn.
    
    5. Sur Kismet, nous obtenons les scores suivants (pour les labels "ap" et "pw"):
    
        Sur la base entière, voisé + non voisés:
        -------------------------
        | KNN |  92    %
        | SVM |  92.67 %
        -------------------------
        Avec seulement voisé:
        -------------------------
        | KNN |  100   %
        | SVM |  100   %
        -------------------------
        Avec seulement non voisé:
        -------------------------
        | KNN |  68.31 %
        | SVM |  64.08 %
        -------------------------
        
        Nous voyons donc que le meilleur score obtenu est lorsque nous prenons en compte
        seulement la partie voisé et ce peu importe le modèle puisque les deux ont obtenus 100%.
        Le problème est donc 100% linéairement séparable en partie voisé. Mais nous obtenons, aussi
        un très bon score avec les deux parties.
        La partie non voisé obtient un bien moins bon score, ce qui montre l'importance de f0 dans nos
        features. 
        Les features les plus importantes qui permettent d'affiner au mieux l'accuracy sont la std, la médiane et
        la moyenne.
        Nous obtenons toujours 100% en voisé à chaque run, ce qui montre bien la séparabilité des données.
        
    Sur BabyEars, nous obtenons les scores suivants:

        Sur la base entière, voisé + non voisés:
        -------------------------
        | KNN |  61.81 %
        | SVM |  62.50 %
        -------------------------
        Avec seulement voisé:
        -------------------------
        | KNN |  73.11 %
        | SVM |  79.83 %
        -------------------------
        Avec seulement non voisé:
        -------------------------
        Pas possible, des valeurs
        provoquent des NaN
        -------------------------
        
        On voit que le Dataset provenant de BabyEars est plus complexe mais 
        qu'encore une fois c'est la partie voisé seul qui permet d'avoir le meilleur
        score sur les modèles mais que cette fois la SVM se démarque.
    
    Exercice 2:
    
    
    1. Nous faisons la même chose avec la même fonction que dans l'exercice 1 mais cette fois
    avec 3 labels
    
    2. Meme type de classifieur que dans l'exercice 1.
    
    3. Sur Kismet, nous obtenons les scores suivants (pour les labels ['ap', 'pw', 'at']):
    
        Sur la base entière, voisé + non voisés:
        -------------------------
        | KNN |  81.94 %
        confusion matrix = 
            [[55  6 10]
            [ 7 72  0]
            [16  0 50]]
        
        | SVM |  83.80 %
        confusion matrix =
            [[57  5  9]
            [ 4 72  3]
            [14  0 52]] 
        -------------------------
        Avec seulement voisé:
        -------------------------
        | KNN |  87.04 %
        confusion matrix = 
            [[58  4  9]
            [ 5 74  0]
            [10  0 56]]
            
        | SVM |  89.81 %
        confusion matrix =
            [[60  4  7]
            [ 2 77  0]
            [ 8  1 57]]
        -------------------------
        Avec seulement non voisé:
        -------------------------
        | KNN |  52.78 %
        confusion matrix = 
            [[43  9 19]
            [27 41 11]
            [25 11 30]]
            
        | SVM |  48.15 %
        confusion matrix = 
            [[60  4  7]
            [50 24  5]
            [39  7 20]]
        -------------------------
        
        On voit encore que le meilleur modèle est le modèle ne prenant en compte que la partie voisé et que c'est le SVM
        qui obtient une meilleure accuracy. Contrairement avec deux labels nous n'obtenons plus une accuracy de 100 %,
        celà vient du fait que le modèle devient plus complexe avec 3 labels avec des sons pouvant se ressembler,
        comme l'atteste la matrice de confusion qui montre bien que que "ap" quelque fois reconnu comme "at" et inversement.
        
    Sur BabyEars, nous obtenons les scores suivants (pour les labels ['ap', 'pr', 'at']):
    
        Sur la base entière, voisé + non voisés:
        -------------------------
        | KNN |  49.51 %
        confusion matrix = 
            [[59 14  8]
            [34 19  9]
            [26 12 23]]
        
        | SVM |  54.90% %
        confusion matrix =
            [[57 11 13]
            [31 21 10]
            [22  5 34]] 
        -------------------------
        Avec seulement voisé:
        -------------------------
        | KNN |  53.92 %
        confusion matrix = 
            [[57 16  8]
            [30 29  3]
            [26 11 24]]
            
        | SVM |  53.43 %
        confusion matrix =
            [[46 20 15]
            [22 29 11]
            [16 11 34]]
        -------------------------
        Avec seulement non voisé:
        -------------------------
        Pareil que exercice 1
        -------------------------

        Même constat que précedement mais les résultats sont bien plus mauvais. On voit que majoritairement 
        les predictions sont classés 'ap', en effet c'est le label présent à 43% dans le X_train contrairement
        aux autres qui sont présents à 28%. Ils faudrait donc réequilibrer le dataset en répartition sur les labels.
        
        
    Exercice 3:
    
    Sur la fusion Intra, nous obtenons les scores suivants (pour les labels ['ap', 'pr' = 'pw', 'at']):
    
        Sur la base entière, voisé + non voisés:
        -------------------------
        | KNN |  62.62 %
        confusion matrix = 
            [[113  28  23]
            [ 39  81   9]
            [ 47  11  69]]
        
        | SVM |  64.52 %
        confusion matrix =
            [[114  28  22]
            [ 30  85  14]
            [ 43  12  72]] 
        -------------------------
        Avec seulement voisé:
        -------------------------
        | KNN |  70.24%
        confusion matrix = 
            [[117  27  20]
            [ 37  90   2]
            [ 29  10  88]]
            
        | SVM |  69.29 %
        confusion matrix =
            [[103  33  28]
            [ 25  94  10]
            [ 25   8  94]]
        -------------------------
        
        Nous voyons que la fusion a permis d'obtenir un score proche de la moyenne des deux en individuel.
        Ce qui montre que la fusion de différents types de dialogues n'ayant pas la même intonnation mais
        ayant le même résultat permet d'augmenter la généralisation du modèle.
        
    """
    
    print('done')
