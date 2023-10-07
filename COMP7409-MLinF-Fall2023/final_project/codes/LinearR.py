import numpy as np


from Evaluation import Evaluation


class LinearR:
    """
    read the dataset

    """
    def __init__(self, X_train1,X_test1,y_train1,y_test1):
        self.X_train = np.array(X_train1)
        self.X_test = np.array(X_test1)
        self.y_train2 = np.array(y_train1)
        self.y_test2=np.array(y_test1)
        self.y_train = self.y_train2.reshape((-1, 1))
        self.y_test = self.y_test2.reshape((-1, 1))


    """
    Function description: Initialize model parameters
    Parameters:
        feature_num 
    Returns:
        y=wx+b
        w &&b
    """


    def initialize_params(self, feature_num):
        w = np.random.rand(feature_num, 1)
        b = 0
        return w, b


    """
    Function description: Pre mathematical preparation for linear regression
    Parameters:
        X, y, w, b 
    Returns:
        y_hat,loss,dw,db
    """


    def forward(self, X, y, w, b):
        num_train = X.shape[0]
        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train
        dw = np.dot(X.T, (y_hat - y)) / num_train
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db


    # Define the training process of linear regression model
    """
    Function description:Define the training process of linear regression model
    Parameters:
        X, y, learning_rate, epochs
    Returns:
        loss_his,w,b
    """


    def my_linear_regression(self, X, y, learning_rate, epochs):
        loss_his = []
        w, b = self.initialize_params(X.shape[1])
        for i in range(epochs):
            y_hat, loss, dw, db = self.forward(X, y, w, b)
            w += -learning_rate * dw
            b += -learning_rate * db
            loss_his.append(loss)
            if i % 100 == 0:
                print("epochs %d loss %f" % (i, loss))
        return loss_his, w, b


    def linearR(self):
        # Linear regression model training
        loss_his, w, b = self.my_linear_regression(self.X_train, self.y_train, 0.01, 5000)

        # Print the model parameters after training
        print("w:", w, "\nb", b)

        y_pred = np.dot(self.X_test, w) + b
        evaluation = Evaluation()
        # Calculate the MSE of the test set
        print("the MSE of the dataset: {:.2f}".format(evaluation.MSE(self.y_test, y_pred)))
        # Calculate the R-factor of the test set
        print("the R2 of the dataset: {:.2f}".format(evaluation.r2_score(self.y_test, y_pred)))

        print("the MAE of the dataset: {:.2f}".format(evaluation.MAE(self.y_test, y_pred)))

        print("the RMSE of the dataset: {:.2f}".format(evaluation.RMSE(self.y_test, y_pred)))


