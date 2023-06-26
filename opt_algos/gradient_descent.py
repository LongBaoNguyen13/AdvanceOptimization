import numpy as np


def gradient_descent(model, alpha, max_iteration=1e4, epsilon=1e-5,
                     x_start=None):
    """
    Gradient descent

    Parameters
    ----------
    model: optimization model object
    alpha: step length
    max_iterations: maximum number of gradient iterations
    epsilon: tolerance for stopping condition
    x_start: where to start (otherwise random)

    Output
    ------
    solution: final x* value
    f_value: f(x*)
    x_history: beta values from each iteration
    """
     
    # data from model
    grad_F = model.grad_F
    n = model.n

    # intialization x_start to start:
    if x_start:
        x_current = x_start
    else:
        x_current = np.random.normal(loc=0, scale=1, size=n)

    # keep track of interation
    x_history = []

    for k in range(int(max_iteration)):

        x_history.append(x_current)

        # gradient update
        # alpha = backtracking_line_search(model, x_current)
        x_next = x_current - alpha * grad_F(x_current)

        # error stoping condition based on Princenton slide
        if np.linalg.norm(x_next - x_current) <= epsilon*np.linalg.norm(x_current):
            break

        # update x
        x_current = x_next
        #alpha = backtracking_line_search(model, x_current)

    print('GD finished after ' + str(k) + ' iterations')

    return {'solution': x_current,
            'f_value': model.F(x_current),
            'x_history': x_history}
