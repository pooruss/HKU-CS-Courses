import numpy as np
from Evaluation import Evaluation

class MLP:

    # suppose one hidden layer

    def __init__(self, numI, numH, numO, lr, epochs, train_X, train_y, test_X, test_y):
        self.numI = numI
        self.numH = numH
        self.numO = numO
        self.lr = lr
        self.epochs = epochs
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.b1 = np.zeros((numH,1))
        self.b2 = np.zeros((numO,1))
        self.W1 = np.random.normal(0.0, pow(numH, -0.5), size=(numH, numI))
        self.W2 = np.random.normal(0.0, pow(numO, -0.5), size=(numO, numH))


    
    def forward_propagation(self, X):
        # X = np.reshape(X, [X.shape[0], 1])      # transfer the input list to (numI, 1)

        hidden_layer = MLP.sigmoid(np.dot(self.W1, X) + self.b1)                # (numH, 1)
        output_layer = MLP.sigmoid(np.dot(self.W2, hidden_layer) + self.b2)     # (numO, 1)

        return output_layer


    # y_hat: predict value
    # y: real value
    def criterion(self, y_hat, y):
        loss = np.mean((y-y_hat)**2)

        return loss
    
    # y_hat: predict value
    # y: real value
    def backpropagation(self, y_hat, y, X):
        output_error = y - y_hat                                    # (numO, 1)
        hidden_error = np.dot(self.W2.T, output_error)              # W2.T(numH, numO), hidden_error(numH, 1)
        hidden_layer = MLP.sigmoid(np.dot(self.W1, X) + self.b1)    # (numH, 1)

        delta_W2 = np.dot(output_error * (y_hat * (1 - y_hat)), hidden_layer.T)
        delta_W1 = np.dot(hidden_error * (hidden_layer * (1 - hidden_layer)), X.T)

        return [delta_W1, delta_W2]
    
    def gradient_descent(self, delta_W1, delta_W2):
        self.W1 += self.lr * delta_W1
        self.W2 += self.lr * delta_W2

    def relu(x):
        return np.maximum(0,x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def train(self):
        for epoch in range(self.epochs):
            for X, y_index in zip(self.train_X, self.train_y):
                X = np.reshape(X, [X.shape[0], 1])
                y = np.zeros((self.numO, 1)) + 0.01
                y[int(y_index)] = 0.99
                result = self.forward_propagation(X)
                loss = self.criterion(result, y)
                gradient = self.backpropagation(result, y, X)
                self.gradient_descent(gradient[0], gradient[1])
        print("loss: ", loss)
        
    def test(self):
        result_mat = np.zeros((self.test_X.shape[0], 1))

        for X, y_index, i in zip(self.test_X, self.test_y, range(self.test_X.shape[0])):
            X = np.reshape(X, [X.shape[0], 1])
            y = np.zeros((self.numO, 1)) + 0.01
            y[int(y_index)] = 0.99
            result = self.forward_propagation(X)
            number = np.argmax(result)
            result_mat[i] = number

        eva = Evaluation()
        acc = eva.accuracy(self.test_y.astype(int), result_mat)
        print("Accuracy:", acc)
