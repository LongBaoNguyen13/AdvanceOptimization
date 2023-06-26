import numpy as np

def backtracking_line_search(model, x_k, alpha_bar=1, ro=0.5, c=1e-4):
    # intial step length
    alpha = alpha_bar

    # vector gradient at x_k
    grad_k = model.grad_F(x_k)
    p_k = -grad_k

    # value f(x) at x_k
    f_k = model.F(x_k)

    # gradient at x_k*p_k
    grad_p_k = c*np.dot(grad_k, p_k)

    # update f(x) at x_k
    f_new = model.F(x_k + alpha*p_k)

    while f_new > f_k + alpha*grad_p_k:
        alpha *= ro
        f_new = model.F(x_k + alpha*p_k)
    
    # return step lengthsss
    return alpha