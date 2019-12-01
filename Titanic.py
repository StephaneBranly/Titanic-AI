import numpy as np
from sklearn import tree

sexe_location=3
embarked_location=1
age_location=4
somme_age=0
nbr_age_info=0
age_moyen=0
fichier = open("train_sortie.csv", "w")
with open("train.csv") as fp:
    print("Traitement des datas en cours")
    print("...")
    header=next(fp)
    header_split= header.split(",") #Split les colonnes
    header_split[sexe_location]="Ho,Fe"
    header_split[embarked_location]="S,C,G"
    fichier.write(",".join(header_split))
    for line3 in fp:
        output2=line3.split(",")
        if not output2[age_location]=="":
            if float(output2[age_location])<1:
                output2[age_location]=""
            else:
                somme_age+=float(output2[age_location])
                nbr_age_info+=1
    age_moyen=somme_age/nbr_age_info
    print("Age moyen a bord:")
    print(age_moyen)
    fp.seek(0)
    next(fp)
    for line in fp:
        output=line.split(",") #Split les colonnes
        if output[sexe_location]=="male": #Convert bin division
            output[sexe_location]="1,0"
        elif output[sexe_location]=="female":
            output[sexe_location]="0,1"
        else:
            output[sexe_location]="0,0"
            
        if output[embarked_location]=="S": #Convert bin division
            output[embarked_location]="1,0,0"
        elif output[embarked_location]=="C":
            output[embarked_location]="0,1,0"
        elif output[embarked_location]=="Q":
            output[embarked_location]="0,0,1"
        else:
            output[embarked_location]="0,0,0"

        if output[age_location]=="":
            output[age_location]=str(age_moyen)
        line2=",".join(output)
        line2=line2.replace(",,", ",0,")
        line2=line2.replace(",,", ",0,")
        line2=line2.replace(",\n", "\n")
        fichier.write(line2)
    print("Traitement des datas fini")
fichier.close()
print("Fichier ferme")
print("")
print("")
print("Ouverture du fichier a traiter")
f = open("train_sortie.csv")
titan = np.loadtxt(f, delimiter=',', skiprows=1)
target_end=11
data_end=11

nombre_test=50
test_idx = []
fin_titan=len(titan)-1
for i in range(0,nombre_test):
    test_idx.append(fin_titan-(1*i))
fin_titan=len(titan)-1
target=titan[:, target_end]
data=titan[:,1:data_end]
# on retire les donnees qu'on veut tester
train_target = np.delete(target, test_idx, axis=0)
train_data = np.delete(data, test_idx, axis=0)


# testing data
test_target = target[test_idx]
test_data = data[test_idx]
# create new classifier
clf = tree.DecisionTreeClassifier()
# train on training data
clf.fit(train_data, train_target)
print("Classifieur entraine")
print("")
# ce que l'on veut
print("Vraies donnees:")
print(test_target)

# ce qui est predit
print("Ce qui est predit:")
result=clf.predict(test_data)
print(result)

prediction_good=0
for i in range(0,nombre_test):
    if result[i]==test_target[i]:
        prediction_good+=1
print("")
print("Precision en %")
print((prediction_good/nombre_test*100))
# visu
print("Creation du PDF de visualisation")
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Prediction.pdf")

