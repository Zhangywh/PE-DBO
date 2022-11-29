###################################################
# This file declares the acquisition functions, such as:
# UCB, RandomGetAPoint, soft-Copeland Score, Max GP Variance et al.
###################################################

import torch
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.sampling import IIDNormalSampler
from cmaes import CMA
import numpy as np
from time import time
import warnings

warnings.filterwarnings("ignore")

def logistic(x):
    return 1 / (1 + torch.exp((-1) * x))


def upper_confidence_bound(objective_function, model, beta):
    """
    Basic UCB acquisition function
    :param objective_function: the objective function containing the bounds and dimension information
    :param model: the Gaussian Process model, find next point according to this model and UCB
    :param beta: UCB parameter
    :return: next point found by UCB and cost time
    """
    time_begin = time()
    bounds = objective_function['bounds']
    dim = objective_function['dim']
    bound_lows = bounds[..., 0].reshape(1, dim)
    bound_highs = bounds[..., 1].reshape(1, dim)
    bounds_new = torch.cat([bound_lows, bound_highs], 0)
    next_point, acq_value_list = optimize_acqf(
        acq_function=UpperConfidenceBound(model=model, beta=beta),
        bounds=bounds_new,
        q=1,
        num_restarts=20,
        raw_samples=10000
    )
    time_end = time()
    return next_point, round(time_end - time_begin, 3)


def get_random_point(objective_function):
    """
    get a point randomly, constrained by the objective function bounds
    :param objective_function: the information of objective function
    :return: a random point tensor, shape is (1, dim)
    """
    bounds = objective_function['bounds']
    return torch.rand(1, objective_function['dim']) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]


def soft_copeland_score_next_x(model, bounds_info, seed, points_number, optimization_type='cmaes'):
    """
    use soft-Copeland score to find next x from duels dataset like {([x, x'], f_preference([x, x']))}
    this process use basic UCB Bayesian Optimization to find next point
    :param model: the GP model fit by {([x, x'], y)}
    :param bounds_info: the dimension and the bound of the acquisition function
    :param seed: random seed of sampling from GP
    :param beta: UCB parameter
    :param points_number: points number of random grid
    :return: next x point
    """
    time_begin = time()
    dim = bounds_info['dim']
    bounds = bounds_info['bounds']

    # define objective function i.e. soft-Copeland score
    def scs_objective_function(x):
        """
        build the objective of the optimization through monte-carlo integral
        :param x: the point x
        :return: soft-Copeland score of x
        """
        # this enables multiple points to be sampled at the same time
        # build the grid
        x = torch.from_numpy(x)
        grid = torch.rand(points_number, dim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]

        # build points set which need to be sampled
        x_ex = x * torch.ones([grid.shape[0], grid.shape[1]])
        sample_points = torch.cat([x_ex, grid], -1)
        sample_points = torch.as_tensor(sample_points, dtype=torch.float32)

        # sample and compute
        sampler = IIDNormalSampler(1, seed=seed)
        posterior = model.posterior(sample_points)
        values = logistic(torch.tensor(sampler(posterior)))
        integral_value = torch.mean(values)

        return integral_value.numpy()

    if optimization_type == 'cmaes':
        optimizer = CMA(mean=np.zeros(dim), sigma=1.3, bounds=bounds.numpy(), population_size=2)
        score = 1e10
        optimal_point = 'NULL'
        for generation in range(20):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = -scs_objective_function(x)
                solutions.append((x, value))
                if value <= score:
                    score = value
                    optimal_point = x
            optimizer.tell(solutions)
    else:
        score = 'NULL'
        optimal_point = 'NULL'
        print('Please input correct optimization type, such as \'direct\' or \'cmaes\'')
    print(f'Soft-copeland score = {score}')
    print(f'x = {optimal_point}')
    print('\n', end='')
    optimal_point = torch.from_numpy(optimal_point).unsqueeze(0)
    optimal_point = torch.as_tensor(optimal_point, dtype=torch.float32)
    time_end = time()
    return optimal_point, round(time_end - time_begin, 3)


def max_variance_next_xx(next_x, model, dataset_info, optimization_type = 'cmaes'):
    """
    use max variance to find next x' in PBO
    :param next_x: the next_x found
    :param model: the GP model
    :param dataset_info: the dimension and the bound of the acquisition function
    :return: next x' point
    """
    time_begin = time()
    dim = dataset_info['dim']
    bounds = dataset_info['bounds']
    def variance(xx):
        xx = torch.from_numpy(xx)
        points = torch.cat((next_x.squeeze(0), xx), -1).unsqueeze(0)
        points = torch.as_tensor(points, dtype=torch.float32)
        y = torch.tensor(model(points).stddev)
        return y.numpy()

    optimization_type = 'cmaes'
    if optimization_type == 'cmaes':
        optimizer = CMA(mean=np.zeros(dim), sigma=1.3, bounds=bounds.numpy(), population_size=2)
        score = 1e10
        next_xx = 'NULL'
        for generation in range(20):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = -variance(x)
                solutions.append((x, value))
                if value <= score:
                    score = value
                    next_xx = x
            optimizer.tell(solutions)
    else:
        score = 'NULL'
        next_xx = 'NULL'
        print('Please input correct optimization type, such as \'direct\' or \'cmaes\'')
    print(f'xx = {next_xx}')
    print('\n', end='')

    next_xx = torch.from_numpy(next_xx).unsqueeze(0)
    next_xx = torch.as_tensor(next_xx, dtype=torch.float32)
    time_end = time()
    return next_xx, round(time_end - time_begin, 3)


def find_max_from_sampling(points_number, model, objective_function, seed):
    """
    find x via find max point of a sample from GP
    :param points_number: points number of the grid
    :param model: the GP model
    :return: next point and cost time
    """
    time_begin = time()
    dim = objective_function['dim']
    bounds = objective_function['bounds']
    # build the grid
    grid = torch.rand(points_number, dim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]

    sampler = IIDNormalSampler(1, seed=seed)
    posterior = model.posterior(grid)
    values = logistic(torch.tensor(sampler(posterior)))

    max_point = torch.tensor(grid[torch.argmax(values)], dtype=torch.float32).reshape(1, dim)
    time_end = time()
    return max_point, round(time_begin - time_end, 3)

