import numpy as np
from scipy.linalg import cholesky, solve_triangular # type: ignore

class LeastSquares(object):
    """
    Least squares
    
    min || Ax - b ||_2^2
    
    X in R^(m x n)
    y in R^m
    beta in R^n
    """
    
    def __init__(self, A, y):
        
        # data
        self.A = A
        self.y = y
        self.B = self.A.T @ self.A
        self.C = 2*self.y.T @ self.A
        self.d = self.y.T @ self.y
        
        self.n = A.shape[1]
        self.m = A.shape[0]
        
        # Lipshitz constant
        self.L_F = np.linalg.norm(A)**2
        self.mu_F = 0
        
    def F(self, x):
        return (x.T*self.B*x - self.C.T*x + self.d)
    
    def grad_F(self, x):
        return (2*self.B*x - self.C)
    
    def hess_F(self):
        return (2*self.B)
    
    def get_solution(self):
        R = cholesky(self.B)
        w = solve_triangular(R, self.A.T @ self.y, trans='T')
        return solve_triangular(R, w)