import numpy as np
from Readdata import Readdata
from Evaluation import Evaluation
import distance as dis

realtestlabel=np.array(Readdata.test_label)
realtestlabel = realtestlabel.astype(int)

import numpy as np


class KNN:
    def __init__(self, k=5):
        self.k = k

    def count(self, closest_points_k):
        counts = {}
        for dist, label in closest_points_k:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
        return counts

    # the predict and evaluate method
    def predictandevaluate(self, X_test, X_train, y_train):
        y_pred = []
        for x in X_test:
            # Calculate the distance between the test point and all points in the training set
            distances = [(dis._distance(x, train_point), label) for train_point, label in zip(X_train, y_train)]

            # Sort by distance from nearest to far
            sorted_distances = sorted(distances)
            # Take the first k closest points
            closest_k = sorted_distances[:self.k]

            # closest_k = np.argsort(distances)[:self.k]


            counts = self.count(closest_k)

            max_count = 0
            max_label = None
            for label, count in counts.items():
                if count > max_count:
                    max_count = count
                    max_label = label
            # Add the prediction results to the list
            y_pred.append(max_label)


        y_pred = np.array(y_pred)
        y_pred = y_pred.astype(int)

        eva = Evaluation()
        acc = eva.accuracy(realtestlabel, y_pred)
        pre = eva.precision(realtestlabel, y_pred)
        rec = eva.recall(realtestlabel, y_pred)
        print("Accuracy:", acc)
        print("Precision:", pre)
        print("Recall:", rec)
