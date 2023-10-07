import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluation:

    def MSE(self, y_test, y_pred):
        return np.sum(np.square(y_pred - y_test)) / y_pred.shape[0]

    def r2_score(self, y_test, y_pred):
        # Test set label mean
        y_avg = np.mean(y_test)
        # Sum of squares of total dispersion
        ss_tot = np.sum((y_test - y_avg) ** 2)
        # Sum of squares of residuals
        ss_res = np.sum((y_test - y_pred) ** 2)
        # R calculation
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def MAE(self, y_test, y_pred):
        return np.sum(abs(y_pred - y_test)) / y_pred.shape[0]

    def RMSE(self, y_test, y_pred):
        return pow(np.sum(np.square(y_pred - y_test)) / y_pred.shape[0], 0.5)

    def precision(self, y_true, y_pred):
        cnt = Counter(y_true)
        prema = np.zeros((len(cnt), 2))
        precision = np.zeros(len(cnt))

        for i in range(len(y_pred)):
            if (y_pred[i] == y_true[i]):
                prema[y_pred[i], 0] += 1
                prema[y_pred[i], 1] += 1
            else:
                prema[y_pred[i], 1] += 1

        for i in range(len(precision)):
            precision[i] = float(prema[i, 0]) / float(prema[i, 1])

        return precision

    def recall(self, y_true, y_pred):
        cnt = Counter(y_true)
        rec = np.zeros((len(cnt), 2))
        recall = np.zeros(len(cnt))

        for i in range(len(y_pred)):
            if (y_pred[i] == y_true[i]):
                rec[y_true[i], 0] += 1
                rec[y_true[i], 1] += 1
            else:
                rec[y_true[i], 1] += 1

        for i in range(len(recall)):
            recall[i] = float(rec[i, 0]) / float(rec[i, 1])


        return recall

    def accuracy(self, y_true, y_pred):
        countacc=0
        for i in range(len(y_pred)):
            if (y_pred[i] == y_true[i]):
                countacc += 1
        acc = countacc / len(y_pred)
        return acc

    def err_rate(self, y_true, y_pred):
        TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
        FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
        TN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
        FN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
        err_rate = (FP + FN) / (TP + TN + FP + FN)
        return err_rate

    def F1(self, y_true, y_pred):
        cnt = Counter(y_true)
        prema = np.zeros((len(cnt), 2))
        precision = np.zeros(len(cnt))

        for i in range(len(y_pred)):
            if (y_pred[i] == y_true[i]):
                prema[y_pred[i], 0] += 1
                prema[y_pred[i], 1] += 1
            else:
                prema[y_pred[i], 1] += 1

        for i in range(len(precision)):
            precision[i] = float(prema[i, 0]) / float(prema[i, 1])

        rec = np.zeros((len(cnt), 2))
        recall = np.zeros(len(cnt))

        for i in range(len(y_pred)):
            if (y_pred[i] == y_true[i]):
                rec[y_true[i], 0] += 1
                rec[y_true[i], 1] += 1
            else:
                rec[y_true[i], 1] += 1

        for i in range(len(recall)):
            recall[i] = float(rec[i, 0]) / float(rec[i, 1])

        F1 = 2 * precision * recall / (precision + recall)
        return F1

    def ConfusionMatrix(self, y_true, y_pred, num_classes):


        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)


        for i in range(len(y_pred)):
            pred = y_pred[i]
            true = y_true[i]
            confusion_matrix[pred][true] += 1

        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds")
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
    def variance_ratio(self,eig_vals,eigenvalue_sum,k):
        eig_vals=eig_vals[:k]
        variance_ratio = eig_vals / eigenvalue_sum
        return variance_ratio

    def plotscatter2d(self,y_true,y_pred):
        plt.scatter(y_pred[:, 0], y_pred[:, 1], c=y_true)
        plt.xlabel('dimension1')
        plt.ylabel('dimension2')
        plt.show()

    def show_decision_boundary(self, w, b, X, y):
        point_a = np.zeros((int(X.shape[0] / 2), X.shape[1]))
        point_b = np.zeros((int(X.shape[0] / 2), X.shape[1]))

        j = 0
        k = 0
        for i in range(X.shape[0]):
            if y[i] == 1:
                point_a[j] = X[i]
                j += 1
            else:
                point_b[k] = X[i]
                k += 1

        x = np.linspace(4, 6, 5)
        y = -(w[0] * x + b) / w[1]
        print("w: ", w)
        print("b: ", b)
        plt.scatter(point_a[:,0], point_a[:,1], color='orange')
        plt.scatter(point_b[:,0], point_b[:,1], color='green')
        plt.plot(x, y, color='purple')
        plt.show()
    
    def kmeans_evaluation(self,x,y,centerpoints):

        plt.scatter(x[:, 0], x[:, 1], c='black', label='Data Points')

        for i, cluster_points in enumerate(y):
            cluster_points = np.array(cluster_points)
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')


        plt.scatter(centerpoints[:, 0], centerpoints[:, 1], c='red', label='Centerpoints')

        plt.legend()
        plt.title('K-means Clustering')
        plt.show()