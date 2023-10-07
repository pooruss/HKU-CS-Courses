import numpy as np

from Readdata import Readdata
from sklearn.model_selection import train_test_split


"""
read the dataset

"""

X_train = np.array(Readdata.train_data)  # 切片得到训练集
X_test = np.array(Readdata.test_data)  # 切片得到测试集
y_train = np.array(Readdata.train_label)  # 切片得到训练集
y_test = np.array(Readdata.test_label)  # 切片得到测试集

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Initialize model parameters"
"""
Function description: Initialize model parameters
Parameters:
    feature_num 
Returns:
    y=wx+b
    w &&b
"""


def initialize_params(feature_num):
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


def forward(X, y, w, b):
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


def my_linear_regression(X, y, learning_rate, epochs):
    loss_his = []
    w, b = initialize_params(X.shape[1])
    for i in range(epochs):
        y_hat, loss, dw, db = forward(X, y, w, b)
        w += -learning_rate * dw
        b += -learning_rate * db
        loss_his.append(loss)
        if i % 100 == 0:
            print("epochs %d loss %f" % (i, loss))
    return loss_his, w, b


# Linear regression model training
loss_his, w, b = my_linear_regression(X_train, y_train, 0.01, 5000)

# Print the model parameters after training
print("w:", w, "\nb", b)

# Define MSE functions
"""
Function description:Define MSE functions
Parameters:
    y_test, y_pred
Returns:
    MSE
"""


def MSE(y_test, y_pred):
    return np.sum(np.square(y_pred - y_test)) / y_pred.shape[0]


# Define R coefficient function
"""
Function description:Define R coefficient function
Parameters:
    y_test, y_pred
Returns:
    R2
"""


def r2_score(y_test, y_pred):
    # Test set label mean
    y_avg = np.mean(y_test)
    # Sum of squares of total dispersion
    ss_tot = np.sum((y_test - y_avg) ** 2)
    # Sum of squares of residuals
    ss_res = np.sum((y_test - y_pred) ** 2)
    # R calculation
    r2 = 1 - (ss_res / ss_tot)
    return r2


def MAE(y_test, y_pred):
    return np.sum(abs(y_pred - y_test)) / y_pred.shape[0]


def RMSE(y_test, y_pred):
    return pow(np.sum(np.square(y_pred - y_test)) / y_pred.shape[0], 0.5)


# Forecast on test set
y_pred = np.dot(X_test, w) + b
# Calculate the MSE of the test set
print("测试集的MSE: {:.2f}".format(MSE(y_test, y_pred)))
# Calculate the R-factor of the test set
print("测试集的R2: {:.2f}".format(r2_score(y_test, y_pred)))

print("测试集的MAE: {:.2f}".format(MAE(y_test, y_pred)))

print("测试集的RMSE: {:.2f}".format(RMSE(y_test, y_pred)))
