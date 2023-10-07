import numpy as np


'''
这个弱分类器是为adaboost算法服务的，我们所选择的弱分类器是单层决策树算法。
每一个弱分类器被看作一个stump，adaboost是由一组stumps组成的。
每一个weak classifier会被用于处理一个feature，因此feature index和best threshold被用来表示这个stump。

'''

class SingleDecisionTrees:
    def __init__(self):
        # self.feature_index = None
        self.threshold = None
        self.best_threshold = None

    def error_rate(self, y_pred, y, weights):
        overall_error_weight = 0
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                overall_error_weight = overall_error_weight + weights[i]
        return overall_error_weight

    # 通过train获得best_threshold
    def train(self, X, y, weights):
        threshold_list = np.unique(X)
        y_pred = np.ones(len(X))
        mini_error = np.inf

        for threshold in threshold_list:
            for i in range(len(X)):
                if X[i] < threshold:
                    y_pred[i] = -1
                else:
                    y_pred[i] = 1

            error = self.error_rate(y_pred, y, weights)
            if error < mini_error:
                mini_error = error
                self.best_threshold = threshold


    def temp_predict(self, X):
        y_pred = np.ones(len(X))
        for i in range(len(X)):
            if X[i] < self.best_threshold:
                y_pred[i] = -1
            else:
                y_pred[i] = 1
        return y_pred

    def predict(self, X):
        y_pred = np.ones(len(X))
        for i in range(len(X)):
            if X[i] < self.threshold:
                y_pred[i] = -1
            else:
                y_pred[i] = 1
        return y_pred


