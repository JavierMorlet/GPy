# Create new kernel 
# # Define K_discrete(y_i, y_j) using an indicatorbased similarity metric k(y_i, y_j) = sigma^2/m * sum_{k=1}^{m}I(y_i^k = y_j^k)
# I is the indicator function: I(y_i^k = y_j^k) = 1 if y_i^k = y_j^k, 0 otherwise
# sigma^2 is the variance of the kernel
# m is the number of discrete variables

from .kern import Kern
import numpy as np

class Discrete(Kern):
    from .core.parameterization import Param
    def __init__(self, input_dim, variance=1., active_dims=None, name='Discrete'):
        super(Discrete, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance)
        self.link_parameters(self.variance)
    # Define K_discrete(y_i, y_j) using an indicatorbased similarity metric k(y_i, y_j) = sigma^2/m * sum_{k=1}^{m}I(y_i^k = y_j^k)
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance* * np.sum(X == X2, axis=1)[:, None]
    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])
    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        self.variance.gradient = np.sum(dL_dK * np.sum(X == X2, axis=1)[:, None])
    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        return np.zeros_like(X)
    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros_like(X)
    