###################################################
# This file is to test the effective dimension of ChEMBL dataset
###################################################

from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset as CEMBLD
from design_bench.oracles.sklearn import RandomForestOracle as RFO
import numpy as np
from cmaes import CMA
import random

np.set_printoptions(suppress=False, threshold=100)


# find the maximum and minimum of the function
def find_max_min(obj, dim_chosen, dim_ori, seed, iter_num, bound, fill_num):
    print('finding min...')
    dim = len(dim_chosen)
    optimizer = CMA(mean=np.zeros(dim), sigma=1.3, bounds=np.array([[-1, 1]] * dim), seed=seed)
    global_min = 1e10
    for generation in range(iter_num):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            point = np.ones(dim_ori) * fill_num
            for i, idx in enumerate(dim_chosen):
                point[idx] = x[i]
            for i in range(dim_ori):
                point[i] = (point[i] + 1) / 2 * (bound[0][i] - bound[1][i]) + bound[1][i]
            point = np.array(point, dtype=np.int32)
            value = obj.predict(point[np.newaxis, :])[0]
            solutions.append((x, value))
            if value <= global_min:
                global_min = value
                arg_min = x
        optimizer.tell(solutions)
    print('finding max...')
    optimizer = CMA(mean=np.zeros(dim), sigma=1.3, bounds=np.array([[-1, 1]] * dim), seed=seed)
    global_max = 1e10
    for generation in range(iter_num):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            point = np.ones(dim_ori) * fill_num
            for i, idx in enumerate(dim_chosen):
                point[idx] = x[i]
            for i in range(dim_ori):
                point[i] = (point[i] + 1) / 2 * (bound[0][i] - bound[1][i]) + bound[1][i]
            point = np.array(point, dtype=np.int32)
            value = -obj.predict(point[np.newaxis, :])[0]
            solutions.append((x, value))
            if value <= global_max:
                global_max = value
                arg_max = x
        optimizer.tell(solutions)
    return global_min, -global_max

# create a dataset and a noisy oracle
dataset = CEMBLD()
oracle = RFO(dataset, noise_std=0.0)

seed = 0
dim_ori = 31
iter_num = 100
fill_num = -1

# the bounds of each dimension in ChEMBL dataset
bound = [[12, 161, 184, 184, 45, 119, 60, 60, 184, 119, 184, 184, 60, 184, 184, 401, 115, 84, 84, 84, 85, 85, 85, 84,
          60, 60, 60, 60, 48, 48, 13],
         [12, 15, 15, 15, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

repeat_time = 50
global_max = 24906.674
global_min = 763.47363

# test the affect of each dimension
file = open('chembl_eff_dim.txt', 'w')
file.write(f'=====================the effect of every dimension=============================\n')
points = np.ones([10000, dim_ori]) * fill_num
diff = np.zeros(dim_ori)
for i in range(dim_ori):
    change = np.linspace(bound[1][i], bound[0][i], 10000)
    temp = np.array(points)
    temp[:, i] = change
    temp = np.array(temp, dtype=np.int32)
    y = oracle.predict(temp)
    temp_max = y.max()
    temp_min = y.min()
    diff[i] = temp_max - temp_min
file.write('the function value change in different dimension :\n')
file.write(str(diff.tolist()))
file.write('\nthe percentage of function value change in different dimension :\n')
diff = diff / (global_max - global_min)
file.write(str(diff.tolist()) + '\n')


# test the affect of different subspaces, repeat repeat_time times
dim = [i for i in range(2, dim_ori, 2)]
diff_mean = []
diff_max = []
for i in dim:
    file.write(f'\n=====================number of chosen dimensions {i}=============================\n')
    diff = np.zeros(repeat_time)
    for j in range(repeat_time):
        dim_chosen = random.sample(range(dim_ori), i)
        dim_chosen.sort()
        file.write(f'in iteration {j}, the chosen dimensions are :{dim_chosen}\n')
        temp_min, temp_max = find_max_min(oracle, dim_chosen, dim_ori, seed, iter_num, bound, fill_num)
        diff[j] = temp_max - temp_min
    file.write(f'the function value change when {i} dimensions can change :\n')
    file.write(str(diff.tolist()))
    file.write('\nthe percentage of function value change in different dimension :\n')
    diff = diff / (global_max - global_min)
    file.write(str(diff.tolist()))
    file.write(f'\nthe mean of the percentage of function value change in different dimension :{diff.mean()}\n')
    diff_mean.append(diff.mean())
    diff_max.append(diff.max())
file.write('\n=====================summary=============================\n')
file.write(f'global max = {global_max}, global min = {global_min}\n')
file.write('\nthe mean of the percentage of function value change when different dimensions can change :\n')
file.write(str(diff_mean))
file.write('\nthe max of the percentage of function value change when different dimensions can change :\n')
file.write(str(diff_max))

