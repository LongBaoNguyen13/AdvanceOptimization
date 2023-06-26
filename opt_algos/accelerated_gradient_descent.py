import numpy as np

#from opt_algos.backtracking import backtracking_line_search

def accelerated_gradient_descent(model, alpha, max_iterartion=1e4, epsilon=1e-5,
                                 x_start=None):
    """
    Nesterov's accelerated gradient descent

    Parameters
    ----------
    model: optimization model object
    alpha: step length
    max_iterations: maximum number of gradient iterations
    epsilon: tolerance for stopping condition
    x_start: where to start (otherwise random)

    Output
    ------
    solution: final x value
    f_value: f(solution)
    beta_history: beta values from each iteration
    """

    # data from model
    grad_F = model.grad_F
    n = model.n


    # intialization x_start to start:
    if x_start:
        x_current = x_start
    else:
        x_current = np.random.normal(loc=0, scale=1, size=n)

    y_current = x_current
    t_current = 1.0

    # keep track of interation
    x_history = []

    for k in range(int(max_iterartion)):

        x_history.append(x_current)

        # gradient update
        t_next = .5*(1 + np.sqrt(1 + 4*t_current**2))
        x_next = y_current - alpha * grad_F(y_current)
        y_next = x_next + (t_current - 1.0)/(t_next)*(x_next - x_current)

        # error stoping condition
        if np.linalg.norm(x_next - x_current) <= epsilon * np.linalg.norm(x_current):
            break

        # restarting strategies
        #if np.dot(y_current - x_next, x_next - x_current) > 0:
            y_next = x_next
            t_next = 1

        x_current = x_next
        y_current = y_next
        t_current = t_next

    print('accelerated GD finished after ' + str(k) + ' iterations')

    return {'solution': x_current,
            'f_value': model.F(x_current),
            'x_history': x_history}