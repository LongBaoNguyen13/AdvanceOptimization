# AdvanceOptimization
This is my implementation of Convex optimization algos from Adavance Optimization class in Ms. Data Science of Hanoi University of Science - VNU.

While taking a convex optimization class this semester I have implemented a few basic algorithms for unconstrained optimization (e.g. Steepest GD,  Nesterov’s accelerated gradient descent, Newton methods) in Python.

The purpose of this repo is for me to learn and to have bare bones implementations of these algorithms sitting around. I tried to make the code modular and simple as possible so that you (or a future me) can modify it for other purposes (e.g. add bells and whistles, implement other algorithms, etc). While off the shelf solvers such as sklean or cvxopt are preferable for many applications there are times when you want full control over the solver.

The important folders are:

    . models: This folder contains a few simple optimization models (e.g. least squares). Each model is an object and has functions such as F or grad_F (the function value and the gradient) that return data used by various optimization algorithms.

    . opt_algos: Various optimization algorithms (line search, back tracking, gradient descent, SGD, etc) are implemented in python. Each optimization function takes a model object (see above) as an argument plus various optimization hyperparameters (e.g. step length, max iterations, etc).

