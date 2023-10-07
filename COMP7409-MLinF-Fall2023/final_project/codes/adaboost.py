import numpy as np
from weakclassifier import SingleDecisionTrees
from Evaluation import Evaluation
from Readdata import Readdata

realtestlabel=np.array(Readdata.test_label)
realtestlabel = realtestlabel.astype(int)
class Adaboost:
    def __init__(self, iter=20):
        self.iter = iter
        self.stumps = []
        self.amounts = []

    def error_rate(self, y_pred, y, weights):
        overall_error_weight = 0
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                overall_error_weight = overall_error_weight + weights[i]
        return overall_error_weight

    def update_weights(self, amount, y, y_pred, weights):
        weights *= np.exp(-amount * y * y_pred)
        weights /= np.sum(weights)
        return weights

    def amount_of_says(self, error):
        amount_of_say = 0.5 * np.log((1 - error) / (error + 1e-10))
        return amount_of_say

    def fit(self, X, y):
        # 1. Initialize sample weights as 1/n sample.
        y= np.where(y > 0, 1, -1)
        n_sample, n_feature = X.shape
        weights = np.ones(n_sample)/n_sample

        for _ in range(self.iter):
            # 2. Train a basic classifier: Train a basic classifier with the current sample weight.
            base_clf = SingleDecisionTrees()
            # adaboost consists of a set of stumps, each base classifier being a stump, each affected by the previous stump.
            stump = base_clf

            # The weak classifier with the lowest error rate is selected as the basic classifier
            mini_error = np.inf
            for feature_index in range(n_feature):
                # 3. Calculate the classification error rate: For each sample, if it is correctly classified, the error is 0; otherwise, the error is 1. The error rate of the basic classifier is the weighted average of all sample errors.
                X_feature_column = X[:, feature_index]

                base_clf.train(X_feature_column, y, weights)
                y_pred = base_clf.temp_predict(X_feature_column)
                err = self.error_rate(y_pred, y, weights)

                if err < mini_error:
                    mini_error = err
                    # In this case, the error rate of the weak classifier is the smallest, so this weak classifier is taken as the basic classifier we need, that is, a stump
                    stump.feature_index = feature_index
                    stump.threshold = base_clf.best_threshold

            # 4. Calculate the amount based on the error rate of the basic classifier
            amount = self.amount_of_says(mini_error)

            # 5. Update sample weights
            weights = self.update_weights(amount, y, stump.predict(X[:, stump.feature_index]), weights)

            # 6. Continue training until the preset number of basic classifiers or certain accuracy requirements are reached.

            self.stumps.append(stump)
            self.amounts.append(amount)



    def predict(self, X):
        # The predictions of all basic classifiers are weighted and summed
        n_sample, n_feature = X.shape
        y_pred = np.zeros(n_sample)

        for i in range(self.iter):
            y_pred += self.amounts[i] * self.stumps[i].predict(X[:, self.stumps[i].feature_index])

        for i in range(len(y_pred)):
            if y_pred[i]<0:
                y_pred[i]=0
            else:
                y_pred[i]=1

        y_pred = y_pred.astype(int)
        print(y_pred)
        print(realtestlabel)

        eva = Evaluation()
        acc = eva.accuracy(realtestlabel, y_pred)
        pre = eva.precision(realtestlabel, y_pred)
        rec = eva.recall(realtestlabel, y_pred)
        F1 = eva.F1(realtestlabel, y_pred)
        print("Accuracy:", acc)
        print("Precision:", pre)
        print("Recall:", rec)
        print("F1:",F1)











