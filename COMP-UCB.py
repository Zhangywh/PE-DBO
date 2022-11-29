###################################################
# Code for comp-GP-UCB
###################################################

import torch
import numpy as np
from core import BO, AF, OF_TFunctions
import os
import datetime

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

def no_growth(x):
    x_no = [x[0]]
    for i in range(len(x)):
        if i != 0:
            if x[i] <= x_no[i - 1]:
                x_no.append(x[i])
            else:
                x_no.append(x_no[i - 1])
    return torch.tensor(x_no)


# Step 1: set the objective function / dataset
# And set iteration times
seed_start = 0
dim_high = 200
dim_eff = 10
init_points = 30
exp_times = 20
iteration_times = 50
function_name = "ackley"  # choose the testing function
beta = 0.5

objective_function = {
    'form': OF_TFunctions.ackley_function,  # choose the testing function
    'dim': dim_high,
    'bounds': BO.bounds11(dim_high),
    'optimal': 0
}


# file save path
folder = os.path.exists("results")
if not folder:
    os.makedirs("results")
path = "./results/" + function_name + "/COMP-UCB_" + datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__()
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

regret_all_exp = torch.zeros([exp_times, iteration_times])

for exp, random_seed in enumerate(range(seed_start, seed_start + exp_times)):
    print("Start experiment: %3d" % exp)
    print('\n', end='')
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Step 2: init original dataset and Gaussian Process
    init_duels_number = init_points
    dataset = BO.init_points_dataset_compucb(init_duels_number, objective_function)
    gp_model, gp_mll, time_fit_gp = BO.fit_model_gp(dataset)

    # Optional: Save the training data
    regret_all = torch.ones(iteration_times)
    time_fit_gp_all = torch.ones(iteration_times)
    time_find_next_all = torch.ones(iteration_times)

    # best point initialization
    min_regret = 1e20
    regret = 0
    optimal_point = dataset['x'][0]

    # Step 3-7: the iteration loop
    for i in range(iteration_times):
        # Step 3: use UCB to find next point x
        next_point, time_find_next = AF.upper_confidence_bound(objective_function, gp_model, beta)

        # Step 4: random choose a point x'
        random_point = AF.get_random_point(objective_function)

        # Seep 5: compare points (x and x') and get the preference information
        preference_info, r, f = BO.get_preference_information(next_point, random_point, objective_function)

        # Step 6: update dataset
        dataset = BO.update_dataset(next_point, preference_info, f[:, 0], dataset)

        # Step 7: fit GP
        gp_model, gp_mll, time_fit_gp = BO.fit_model_gp(dataset)

        # iteration log
        print("iteration: %3d, next point: %s" % (i + 1, next_point))
        print("                regret now: %s" % r)
        print('\n', end='')
        regret_all[i] = r
        time_fit_gp_all[i] = time_fit_gp
        time_find_next_all[i] = time_find_next
        if r <= min_regret:
            min_regret = r
            optimal_point = next_point

    print('Best point is:')
    print(optimal_point)
    print('The best regret is:')
    print(min_regret)


    # Save data to file
    file_path = path + '/' + "seed" + str(random_seed) + "_" + \
                datetime.datetime.now().strftime('%m%d-%H-%M-%S').__str__() + ".txt"

    file = open(str(file_path), 'w')
    file.write("=============================== \n")
    file.write("EX: comp-GPUCB \n")
    file.write("Datetime: " + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').__str__()) + " \n")
    file.write("=============================== \n\n\n")
    file.write("=============================== \n")
    file.write("          BASIC INFOS           \n")
    file.write("=============================== \n")
    file.write("D: " + str(dim_high) + " \n")
    file.write("d_e: " + str(dim_eff) + " \n")
    file.write("Objective Function: \n" + str(objective_function) + " \n")
    file.write("Experiment times: " + str(exp_times) + " \n")
    file.write("iteration times: " + str(iteration_times) + " \n")
    file.write("init points number: " + str(init_duels_number) + " \n")
    file.write("random seed: " + str(random_seed) + " \n")
    file.write("=============================== \n\n\n")

    file.write("=============================== \n")
    file.write("        All Loop Results        \n")
    file.write("=============================== \n")
    file.write("\nThe regret: \n")
    file.write(str(regret_all))
    file.write("\n\nThe Sampling duels cost time: \n")
    file.write(str(time_find_next_all))
    file.write("\n\nThe Fit GP cost time: \n")
    file.write(str(time_fit_gp_all))
    file.write("\n\nThe optimal point: \n")
    file.write(str(optimal_point))
    file.write("\n\nThe regret of optimal point: \n")
    file.write(str(regret))
    file.write("\n\nThe total dataset of x is: \n")
    file.write(str(dataset['x']))
    file.write("\n\nThe total dataset of preference imformation is: \n")
    file.write(str(dataset['y']))
    file.write("\n\nThe total dataset of function value is: \n")
    file.write(str(dataset['f']))
    file.write("\n\n=============================== \n\n\n")

    file.close()
    regret_all_exp[exp] = no_growth(regret_all)

best_regret = torch.zeros(iteration_times)
for i in range(iteration_times):
    best_regret[i] = regret_all_exp[:, i].min()
mean = torch.mean(regret_all_exp, dim=0)
std = torch.sqrt(torch.var(regret_all_exp, dim=0))
median = np.median(regret_all_exp.numpy(), axis=0)
file = open(str(path + '/experiment_result.txt'), 'w')
file.write(f"The best regret across all the {exp_times} experiments: \n")
file.write(str(best_regret))
file.write(f"\n\nThe mean of the regret across all the {exp_times} experiments: \n")
file.write(str(mean))
file.write(f"\n\nThe standard deviation of the regret across all the {exp_times} experiments: \n")
file.write(str(std))
file.write(f"\n\nThe median of the regret across all the {exp_times} experiments: \n")
file.write(str(median))

