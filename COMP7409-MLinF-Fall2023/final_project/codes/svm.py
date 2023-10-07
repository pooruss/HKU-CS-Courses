import numpy as np
import random

class SVM:

    def __init__(self, X, Y, C, e, max_iteration):
        self.X = X
        self.Y = Y
        self.C = C
        self.e = e
        self.max_iteration = max_iteration

        self.M = X.shape[0]
        self.N = X.shape[1]

        self.w = np.zeros(self.N)
        self.b = 0

        self.a = np.zeros(self.M)
    
    def SMO(self):
        iteration = 0 
        while iteration < self.max_iteration:
            counter = 0
            
            for i in range(self.M):
                E_xi = self.E_x(i)

                if self.Y[i] * self.E_x(i) < -self.e and self.a[i] < self.C or self.Y[i] * self.E_x(i) > self.e and self.a[i] > 0:
                    j = self.random_value(i)
                    E_xj = self.E_x(j)

                    a_i_old = self.a[i]
                    a_j_old = self.a[j]

                    L, H = self.get_boundary(i, j, a_i_old, a_j_old)
                    if L == H:
                        continue

                    n = self.Kernel(i, i) + self.Kernel(j, j) - 2 * self.Kernel(i, j)
                    if n <= 0:
                        continue

                    a_j_new_unclipped = a_j_old + self.Y[j] * (E_xi - E_xj) / n
                    self.a[j] = self.use_boundary(L, H, a_j_new_unclipped)

                    if abs(self.a[j] - a_j_old) < 0.00001:
                        continue

                    self.a[i] = a_i_old + self.Y[i] * self.Y[j] * (a_j_old - self.a[j])

                    b1 = self.b - E_xi - self.Y[i] * self.Kernel(i, i) * (self.a[i] - a_i_old) - self.Y[j] * self.Kernel(i, j) * (self.a[j] - a_j_old)
                    b2 = self.b - E_xj - self.Y[i] * self.Kernel(i, j) * (self.a[i] - a_i_old) - self.Y[j] * self.Kernel(j, j) * (self.a[j] - a_j_old)
                    if 0 < self.a[i] < self.C:
                        self.b = b1
                    elif 0 < self.a[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    counter += 1
            if counter == 0:
                iteration += 1
            else:
                iteration = 0
        for i in range(self.M):
            self.w = self.w + self.Y[i] * self.a[i] * self.X[i] 

        return self.w, self.b

    def g_x(self, i):
        g_xi = 0
        for n in range(self.M):
            g_xi += self.a[n] * self.Y[n] * self.Kernel(i, n)
        g_xi += self.b
        return g_xi
    
    def E_x(self, i):
        return self.g_x(i) - self.Y[i]

    def Kernel(self, i, j):
        result = np.matmul(self.X[i], self.X[j].T)
        return result
    
    def KKT(self, i):
        return self.Y[i] * self.E_x(i) < -self.e and self.a[i] < self.C or self.Y[i] * self.E_x(i) > self.e and self.a[i] > 0
    
    def random_value(self, i):
        numbers = list(range(self.M))
        numbers.remove(i)
        j = random.choice(numbers)
        return j

    def get_boundary(self, i, j, a_i_old, a_j_old):
        L = 0
        H = 0
        if self.Y[i] == self.Y[j]:
            L = max([0, a_j_old + a_i_old - self.C])
            H = min([self.C, a_j_old + a_i_old])
        else:
            L = max([0, a_j_old - a_i_old])
            H = min([self.C, self.C + a_j_old - a_i_old])

        return L, H
    
    def use_boundary(self, L, H, a_j_new_unclipped):
        a_j_new = a_j_new_unclipped
        if a_j_new_unclipped < L:
            a_j_new = L
        if a_j_new_unclipped > H:
            a_j_new = H
        return a_j_new
    
    def preprocess(self):
        for i in range(len(self.Y)):
            if self.Y[i] == 0:
                self.Y[i] = -1
            else:
                self.Y[i] = 1
        