import numpy as np
from Readdata import Readdata
from Evaluation import Evaluation

class PCA:
    X =np.array(Readdata.dataset)

    k=2
    def PCA(X, k):
        X = X - np.mean(X, axis=0)
        # De-averaging
        cov_X = np.cov(X, rowvar=False)
        # calculate the covariance
        eig_vals, eig_vecs = np.linalg.eig(cov_X)
        # The eigenvalue decomposition of covariance matrix
        eigenvalue_sum = np.sum(eig_vals)
        # calculate the variance ratio in each principal components
        temp = np.argsort(-eig_vals)
        # sort eigenvalues from biggest to smallest
        top_k_eig_vecs = eig_vecs[:, temp[:k]]
        # get the first k eigenvectors
        Y = np.dot(X, top_k_eig_vecs)
        # map the data to low dimensional space

        return Y,eig_vals,eigenvalue_sum

    # Use PCA to get the result
    Y,eig_vals,eigenvalue_sum = PCA(X, k)
    # Evaluate the result
    eva = Evaluation()
    variance_ratio=eva.variance_ratio(eig_vals,eigenvalue_sum,k)
    print('variance ratio:',variance_ratio)
    eva.plotscatter2d(np.array(Readdata.label),Y)