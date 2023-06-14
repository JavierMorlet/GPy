# Create new kernel
# ConstantKernel is a kernel that returns a constant value for any two points X and X' given by the variance parameter sigma^2

from .kern import Kern
import numpy as np

class ConstantKernel(Kern):
    def __init__(self, input_dim, variance=1., active_dims=None, name='Constant'):
        super(ConstantKernel, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance)
        self.link_parameters(self.variance)
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance*np.ones((X.shape[0], X2.shape[0]))
    def Kdiag(self, X):
        return self.variance*np.ones(X.shape[0])
    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        self.variance.gradient = np.sum(dL_dK)
    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        return np.zeros_like(X)
    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros_like(X)