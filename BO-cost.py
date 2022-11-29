###################################################
# Code for BO
###################################################
from bayes_opt import BayesianOptimization
from core import OF_Datasets
from core.OF_Datasets import Net
import torch
import os
import datetime


function_name = "cost/MSLR"
obj_func = OF_Datasets.MSLR
random_seed = 0
init_points = 13
n_eval = 20
n_iter = 80

regret_all = torch.ones(n_eval, n_iter + init_points)

pbounds = {}
for i in range(136):
    pbounds['x' + str(i + 1)] = (-1, 1)

def black_box_function(**x):
    inputs = torch.tensor([list(x.values())], dtype=torch.float32)
    result = float(obj_func(inputs))
    return result

for e in range(n_eval):
    print("evaluation: " + str(e))
    optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=e+random_seed)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    max_value = -1e20
    for i, res in enumerate(optimizer.res):
        if res['target'] >= 0:
            a = res['target']
        else:
            a = - res['target']
        if a <= max_value:
            regret_all[e][i] = max_value
        else:
            regret_all[e][i] = a
            max_value = a

mean = torch.mean(regret_all, dim=0)
std = torch.sqrt(torch.var(regret_all, dim=0))

folder = os.path.exists("results")
if not folder:
    os.makedirs("results")
path = "./results/" + function_name + "/BASIC_UCB" + "_D136_exp" + str(n_eval) + "_loop" + str(n_iter) + "_" + \
       datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)
file = open(str(path + '/experiment_result.txt'), 'w')

file.write(f"\n\nThe mean: \n")
file.write(str(mean))
file.write(f"\n\nThe standard deviation: \n")
file.write(str(std))
file.write(f"\n\nAll the regret: \n")
file.write(str(regret_all))

