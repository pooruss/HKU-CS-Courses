from Evaluation import Evaluation
import mlp
from datasets_to_approaches import datasets_to_approaches
import numpy as np
import LinearR
import svm
import ID3
import C45
import adaboost
import knn
from kmeans import Kmeans
import matplotlib.pyplot as plt

# Step 1: input a dataset and preprocess the dataset
from Readdata import Readdata
print()

data = Readdata.data
dataset = Readdata.dataset
label = Readdata.label
train_data = Readdata.train_data
train_label = Readdata.train_label
test_data = Readdata.test_data
test_label = Readdata.test_label


# Step 2: select a suitable ML approach for the dataset

ask_dataset = datasets_to_approaches(Readdata.input_word)
approaches = ask_dataset.datasets_to_approach()
ask_dataset.print_approach(approaches)

print()

selected_approach = input("Please choose an approach: ")

# Step 3: train, test and evaluation the ML model

if selected_approach == 'Linear Regression':
    linr = LinearR.LinearR(train_data, test_data, train_label, test_label)
    linr.linearR()
    
elif selected_approach == 'Naive Bayes':
    import nb_number

elif selected_approach == 'Multilayer Perceptron':
    if (Readdata.input_word == 'fashion-mnist.csv'):
        mlp = mlp.MLP(784, 256, 10, 0.3, 1, np.array(train_data),np.array(train_label),np.array(test_data),np.array(test_label))
        mlp.train()
        mlp.test()
    elif (Readdata.input_word == 'iris.csv'):
        mlp = mlp.MLP(4, 3, 3, 0.3, 20, np.array(train_data),np.array(train_label),np.array(test_data),np.array(test_label))
        mlp.train()
        mlp.test()
    elif (Readdata.input_word == 'mobile_price.csv'):
        mlp = mlp.MLP(20, 12, 4, 0.4, 10, np.array(train_data),np.array(train_label),np.array(test_data),np.array(test_label))
        mlp.train()
        mlp.test()

elif selected_approach == 'Density-Based Spatial Clustering of Applications with Noise':
    import DBSCAN

elif selected_approach == 'Principal Component Analysis':
    import PCA

elif selected_approach == 'Supporting Vector Machine':
    choice = input("please enter the dimensional of inputs(2 or 4): ")
    if choice == '2':
        svm = svm.SVM(np.array(dataset)[:,:2], np.array(label), 100, 0.01, 40)
        svm.preprocess()
        w, b = svm.SMO()
        eva = Evaluation()
        eva.show_decision_boundary(w, b, np.array(dataset)[:,:2], np.array(label))
    elif choice == '4':
        svm = svm.SVM(np.array(dataset), np.array(label), 100, 0.01, 40)
        svm.preprocess()
        w, b = svm.SMO()
        print("w: ", w)
        print("b: ", b)

elif selected_approach == 'Iterative Dichotomiser 3':
    id3 = ID3.ID3(data)
    id3.id3()

elif selected_approach == 'C4.5':
    c45 = C45.C45(train_data, test_data)
    c45.c45()

elif selected_approach == 'Adaboost':
    p=adaboost.Adaboost(3)
    p.fit(np.array(train_data),  np.array(train_label))
    y_pred= p.predict(np.array(test_data))

elif selected_approach == 'K-Nearest Neighbor':
    knn=knn.KNN()
    knn.predictandevaluate(np.array(test_data),np.array(train_data),  np.array(train_label))

elif selected_approach =='K-Means Clustering':
    kmeans_clf = Kmeans(k=3)
    dataset=np.array(dataset)
    kmeans_clf.fit(dataset)
    y_preds = kmeans_clf.predict(np.array(dataset))
    centerpoints = kmeans_clf.centerpoints
    eva=Evaluation()
    eva.kmeans_evaluation(dataset,y_preds,centerpoints)
    


