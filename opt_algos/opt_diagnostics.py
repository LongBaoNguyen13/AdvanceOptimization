import numpy as np
import matplotlib.pyplot as plt


def opt_history(model, x_history, x_solution):
    """
    Computes optimization history

    Parameters
    ----------
    model: optimization model
    x_history: history of x values

    Returns
    -------
    fun_history: function values
    grad_history: 2-norm of gradient
    x_error: 2-norm error of x from solution where solution is either
    provided or computed based on the min function value

    fun_history, grad_history, x_error = opt_history(model, x_history)
    """

    fun_history = [model.F(x) for x in x_history]
    grad_history = [np.linalg.norm(model.grad_F(x)) for x in x_history]

    # beta_solution = model.get_solution()
    # if not beta_solution:
    # beta giving min function value
    # beta_solution = beta_history[np.argmin(fun_history)]

    x_error = [np.linalg.norm(x_solution - x) for x in x_history]

    return fun_history, grad_history, x_error

def plot_opt_path(x_history, model, x_solution):

    # model history
    fun_history, grad_history, x_error = opt_history(model, x_history, x_solution)

    plt.figure(figsize=[15, 5])

    # absolutes
    plt.subplot(2, 3, 1)
    plt.plot(fun_history)
    plt.ylabel('function value')
    plt.xlabel('iteration')
    plt.title(opt_algo + ' for ' + model.name)

    plt.subplot(2, 3, 2)
    plt.plot(x_error)
    plt.ylabel('||beta - beta*||')
    plt.xlabel('iteration')

    plt.subplot(2, 3, 3)
    plt.plot(grad_history)
    plt.ylabel('grad F')
    plt.xlabel('iteration')