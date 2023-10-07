import numpy as np
from Readdata import Readdata
from Evaluation import Evaluation

class DBSCAN:



    # get the index of all points that are no more than eps away from point x
    def find_all_neighbors(self,dataset, idx, eps):
        neighbors = []
        for i in range(len(dataset)):
            # Calculate the Euclidean distance between two points
            if np.sqrt(np.sum((dataset[i]-dataset[idx])**2)) <= eps:
                neighbors.append(i)
        return neighbors


    # DBSCAN
    def dbscan(self, dataset, eps, min_samples):
        # The label of all data is marked as -1(noise) in the begining
        labels = [-1] * len(dataset)
        # the label of first cluster is 0
        cluster_label = 0

        for i in range(len(dataset)):
            # if this piece of data has a label, continue
            if labels[i] != -1:
                continue

            # find_all_neighbors is a function to find all points within
            neighbors = self.find_all_neighbors(dataset, i, eps)

            # if the number of points is smaller than minPts, it’s label is equal to -2
            if len(neighbors) < min_samples:
                labels[i] = -2
                continue

            labels[i] = cluster_label
            # traverse all other points within
            for neighbor_i in neighbors:
                # if it’s label is -2, it is a border point
                if labels[neighbor_i] == -2:
                    labels[neighbor_i] = cluster_label
                # if it is not the noise, continue
                if labels[neighbor_i] != -1:
                    continue
                labels[neighbor_i] = cluster_label
                # For these points, find all points within
                new_neighbors = self.find_all_neighbors(dataset, neighbor_i, eps)
                # if within , the new points still has at least minPts points, it is a new core point
                if len(new_neighbors) >= min_samples:
                    neighbors.extend(new_neighbors)

            # complete the discovery of a cluster, add one to find a new cluster
            cluster_label += 1

        return np.array(labels)

dbs=DBSCAN()
# load dataset
X = Readdata.dataset
X = np.array(X)

labels = dbs.dbscan(X, eps=0.5, min_samples=5)
# Use the dbscan to get the result

eva=Evaluation()
eva.plotscatter2d(labels,X)
# Evaluate the result

