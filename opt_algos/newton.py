import numpy as np

def BFGS(model, alpha=None, max_iteration=1e4, epsilon=1e-5,
        x_start=None):

    Bf = np.eye(len())