# import Readdata as read
# import Evaluation as eval
import distance as dist
import numpy as np


class Kmeans:
    def __init__(self, k=3, iter=10):
        self.k = k
        self.iter = iter
        self.centerpoints = None

    '''
    np.random.choice(len(X), K, replace=False)is used from 0 to len(X)-1 to choose K non-repeating index
    K is the number of selected random points
    replace=False means the selected index is not allowed to be the same
    '''

    def random_initialization(self, X, k):
        indexes = np.random.choice(len(X), k, replace=False)
        random_points = X[indexes]
        return random_points

    def get_distance(self, x, centerpoints, model='manhattan', p=3):
        distances = []
        for i in range(len(centerpoints)):
            distance = dist._distance(x, centerpoints[i], model, p)
            distances.append(distance)
        return distances

    def equal(self, new_centerpoints, centerpoints):
        for i in range(self.k):
            if new_centerpoints[i].any() != centerpoints[i].any():
                return False
        return True

    def fit(self, X):
        # k center points were randomly selected as the original clustering center
        self.centerpoints = self.random_initialization(X, self.k)

        for _ in range(self.iter):
            clusters = [[] for _ in range(self.k)]
            # For each data point, its distance from each cluster center is calculated and divided into the category to which the nearest cluster center belongs
            for x in X:
                distances = []
                distances = self.get_distance(x, self.centerpoints)
                label = np.argmin(distances)
                clusters[label].append(x)

            # For each cluster, the mean of all the data points in it is calculated and used as the new cluster center
            new_centerpoints = np.zeros((self.k, X.shape[1]))
            for i in range(self.k):
                cluster = clusters[i]
                if len(cluster)>0:
                    new_centerpoints[i] = np.mean(cluster)
                else:
                    new_centerpoints[i] = self.centerpoints[i]

            # Repeat steps 2 and 3 until the clustering center no longer changes or the maximum number of iterations is reached
            if self.equal(new_centerpoints, self.centerpoints):
                break

            self.centerpoints = new_centerpoints

    def predict(self, X):
        y_preds = [[] for _ in range(self.k)]
        for x in X:
            distances = self.get_distance(x, self.centerpoints)
            label = np.argmin(distances)
            y_preds[label].append(x)
        return y_preds





