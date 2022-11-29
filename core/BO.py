###################################################
# This file includes modules in BO expect AF, such as:
# FitGP, Observation, initPoints, initDuels, getPiF, updateDataset et al.
###################################################

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from time import time
from core import RE


def init_points_dataset_bo(n, objective_function):
    """
    random init n points for basic BO
    :param n: init points num
    :param objective_function: the objective function containing the bounds and dimension information
    :return: the original dataset (a dict have 2 elements 'x' and 'y')
    """
    dim = objective_function['dim']
    bounds = objective_function['bounds']
    dataset = {'x': torch.rand(n, dim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]}
    dataset['y'] = objective_function['form'](dataset['x']).reshape(n, 1)
    return dataset


def init_points_dataset_compucb(n, objective_function):
    """
    random init n points for COMP-UCB
    :param n: init points num
    :param objective_function: the objective function containing the bounds and dimension information
    :return: the original dataset (a dict have 2 elements 'x' and 'y')
    """
    dim = objective_function['dim']
    bounds = objective_function['bounds']
    x = torch.rand(n, dim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
    xx = torch.rand(n, dim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
    y, cost_time, f = get_preference_information(x, xx, objective_function)
    y = y.reshape(n, 1)
    dataset = {'x': x, 'y': y, 'f': f[:, 0]}
    return dataset


def init_duels_dataset_pbo(n, objective_function):
    """
    random init n duels for PBO or KSS(kernel self sparring)
    :param n: number of duels
    :param objective_function: the objective function containing the bounds and dimension information
    :return: the original dataset (a dict have 2 elements 'x' and 'y')
            the dimension of 'x' is 2 * obj_fun['dim'], because it's a set of duels
    """
    dim = objective_function['dim']
    bounds = objective_function['bounds']
    a = torch.rand(n, dim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
    aa = torch.rand(n, dim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
    x = torch.cat([a, aa], 0)
    xx = torch.cat([aa, a], 0)
    y, r1, f = get_preference_information(x, xx, objective_function)
    y = y.reshape(2 * n, 1)
    dataset = {'x': torch.cat([x, xx], -1), 'y': y, 'f': f}
    return dataset


def fit_model_gp(dataset):
    """
    Use training dataset to fit the Gaussian Process Model
    :param dataset: a dict have 2 elements 'x' and 'y', each of them is a tensor shaped (n, dim) and (n, 1)
    :return: the GP model, the marginal log likelihood and cost time
    """
    time_begin = time()
    dataset_x = dataset['x']
    dataset_y = dataset['y']
    model = SingleTaskGP(dataset_x, dataset_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    time_end = time()
    return model, mll, round(time_end - time_begin, 3)


def test_function_observation(observation_point, objective_function):
    """
    observe function value of the observation_point via objective_function
    :param observation_point: shape is (1, dim)
    :param objective_function: this is a dict like
        objective_function = {
            'form': ObjFunc.test_function_2d,
            'dim': 2,
            'bounds': torch.tensor([[0, 1], [0, 1]])
        }
        and the 'form' is a func which can return the value of x
    :return: the observation value
    """
    ans = objective_function['form'](observation_point)
    return ans, torch.abs(objective_function['optimal'] - ans)


def update_dataset(new_point, value, f, dataset):
    """
    add (new_point, value) to the normal dataset (NOT COMP DATASET)
    :param new_point: shape is (1, dim) or (dim)
    :param value: observed value of new point, shape is (1) or (1, 1)
    :param dataset: the dataset as {(x, f(x))}, which is a dict
    :return: the new dataset
    """
    new_point = new_point.reshape(1, new_point.shape[-1])
    value = value.reshape(1, 1)
    dataset['x'] = torch.cat([dataset['x'], new_point], 0)
    dataset['y'] = torch.cat([dataset['y'], value], 0)
    dataset['f'] = torch.cat([dataset['f'], f], 0)
    return dataset


def update_dataset_ucb(new_point, value, dataset):
    """
    add (new_point, value) to the normal dataset (NOT COMP DATASET)
    :param new_point: shape is (1, dim) or (dim)
    :param value: observed value of new point, shape is (1) or (1, 1)
    :param dataset: the dataset as {(x, f(x))}, which is a dict
    :return: the new dataset
    """
    new_point = new_point.reshape(1, new_point.shape[-1])
    value = value.reshape(1, 1)
    dataset['x'] = torch.cat([dataset['x'], new_point], 0)
    dataset['y'] = torch.cat([dataset['y'], value], 0)
    return dataset



def update_dataset_pbo(x_next, xx_next, objective_function, dataset):
    """
    use (x,x') and (x',x) to update dataset in PBO
    :param x_next: first element of duels
    :param xx_next: second element of duels
    :param objective_function: the attribute dict of objective function
    :param dataset: the original dataset
    :return: new dataset
    """
    duel1 = torch.cat([x_next, xx_next], 1)
    duel2 = torch.cat([xx_next, x_next], 1)
    preference_info1, r1, f1 = get_preference_information(x_next, xx_next, objective_function)
    preference_info2, r2, f2 = get_preference_information(xx_next, x_next, objective_function)
    update_dataset(duel1, preference_info1, f1, dataset)
    update_dataset(duel2, preference_info2, f2, dataset)

    if r1 <= r2:
        r = r1
        better = 'x'
    else:
        r = r2
        better = 'xx'
    return dataset, r, better


def logistic(x):
    return 1 / (1 + torch.exp((-1) * x))


def get_preference_information(point1, point2, objective_function):
    """
    observe f(x) and f(x'), do logistic(f(x) - f(x'))
    :param point1: the x
    :param point2: the x'
    :param objective_function: the attribute dict of objective function
    :return: the preference information
    """
    fx1, regret1 = test_function_observation(point1, objective_function)
    fx2, regret2 = test_function_observation(point2, objective_function)
    n = regret1.shape[0]
    regret = torch.min(torch.cat([regret1.reshape(1, n), regret2.reshape(1, n)], 0), 0).values
    preference_info = logistic(fx1 - fx2) - torch.rand(fx1.shape[0])
    for i, cc in enumerate(preference_info):
        if cc >= 0:
            preference_info[i] = 1.
        else:
            preference_info[i] = 0.
    return preference_info, regret, torch.cat([fx1.resize(fx1.shape[-1], 1), fx2.resize(fx2.shape[-1], 1)], 1)


def init_points_dataset_kss(init_points_number, objective_function):
    """
    init original dataset for kss
    :param init_points_number: number of init points, x2
    :param objective_function: the attribute dict of objective function
    :return: the original dataset
    """
    dataset = init_duels_dataset_pbo(init_points_number, objective_function)
    x = dataset['x'][..., 0:objective_function['dim']]
    dataset['x'] = x
    return dataset


def update_dataset_kss(x_next, xx_next, objective_function, dataset):
    """
    use (x,x') and (x',x) to update dataset in KSS
    :param x_next: first element of duels
    :param xx_next: second element of duels
    :param objective_function: the attribute dict of objective function
    :param dataset: the original dataset
    :return: new dataset
    """
    preference_info1, r1, f1 = get_preference_information(x_next, xx_next, objective_function)
    preference_info2, r2, f2 = get_preference_information(xx_next, x_next, objective_function)
    update_dataset(x_next, preference_info1, f1, dataset)
    update_dataset(xx_next, preference_info2, f2, dataset)
    if r1 <= r2:
        r = r1
        better = 'x'
    else:
        r = r2
        better = 'xx'
    return dataset, r, better


def init_duels_dataset_pedbo(n, dataset_info, objective_function, rem):
    """
    init duels, embedding them to high-dimension space, and generate original dataset
    :param n: duels number = 2 * n
    :param dataset_info: dataset dimension and bounds infomation
    :param objective_function: the ['dim'] is dim_high
    :param rem: random embedding matrix
    :return:
    """
    # generate points in low dimension
    dim_low = dataset_info['dim']
    bounds = dataset_info['bounds']
    a = torch.rand(n, dim_low) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
    aa = torch.rand(n, dim_low) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]
    a = torch.as_tensor(a, dtype=torch.float32)
    aa = torch.as_tensor(aa, dtype=torch.float32)
    # embedding to high dimension
    b = RE.random_embedding(a, rem)
    bb = RE.random_embedding(aa, rem)
    # get preference information
    y, r, f = get_preference_information(torch.cat([b, bb], 0), torch.cat([bb, b], 0), objective_function)
    y = y.reshape(2 * n, 1)
    # generate original duels dataset
    x = torch.cat([a, aa], 0)
    xx = torch.cat([aa, a], 0)
    dataset = {'x': torch.cat([x, xx], -1), 'y': y, 'f': f}
    return dataset


def update_dataset_pedbo(x_next, xx_next, rem, dataset_info, objective_function, dataset):
    """
    use (x,x') and (x',x) to update dataset in PE-DBO
    :param x_next: the first point of duel
    :param xx_next: the second point of duel
    :param rem: random embedding matrix (random matrix)
    :param dataset_info: dataset dimension and bounds infomation
    :param objective_function: the information of objective function
    :param dataset: low dimension optimization dataset
    :return: updated dataset
    """
    # embedding and get preference information
    x_next_high = RE.random_embedding(x_next, rem)
    xx_next_high = RE.random_embedding(xx_next, rem)
    y, r1, f1 = get_preference_information(
        torch.cat([x_next_high], 0),
        torch.cat([xx_next_high], 0),
        objective_function)
    yy, r2, f2 = get_preference_information(
        torch.cat([xx_next_high], 0),
        torch.cat([x_next_high], 0),
        objective_function)
    y = y.reshape(1, 1)
    yy = yy.reshape(1, 1)

    # generate original duels dataset
    x = torch.cat([x_next, xx_next], 0)
    xx = torch.cat([xx_next, x_next], 0)
    dataset['x'] = torch.cat([dataset['x'], torch.cat([x, xx], -1)], 0)
    dataset['y'] = torch.cat([dataset['y'], y], 0)
    dataset['y'] = torch.cat([dataset['y'], yy], 0)
    dataset['f'] = torch.cat([dataset['f'], f1], 0)
    dataset['f'] = torch.cat([dataset['f'], f2], 0)
    if r1 <= r2:
        r = r1
        better = 'x'
    else:
        r = r2
        better = 'xx'
    return dataset, r, better


def bounds11(dim):
    a = torch.tensor([[-1., 1.]])
    b = a
    for i in range(dim - 1):
        b = torch.cat([b, a], 0)
    return b
