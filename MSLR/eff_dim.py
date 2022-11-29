###################################################
# This file is to test the effective dimension of MSLR dataset
###################################################

import torch
import numpy as np
from cmaes import CMA
import random
from Fit import Net


# find the maximum and minimum of the function
def find_max_min(model, dim_chosen, x_mean):
    dim = len(dim_chosen)
    print(dim)
    print("finding min")
    min_optimizer = CMA(mean=np.zeros(dim), sigma=1.3, bounds=np.array([[-1, 1]] * dim))
    global_min = 1e10
    for generation in range(50):
        solutions = []
        for _ in range(min_optimizer.population_size):
            x = min_optimizer.ask()
            x_model = x_mean
            for idx, dim_i in enumerate(dim_chosen):
                x_model[dim_i] = x[idx]
            x_model = torch.from_numpy(x_model)
            x_model = torch.as_tensor(x_model, dtype=torch.float32).to(device)
            value = model(x_model).cpu().detach().numpy()
            value = 4.0 * value
            solutions.append((x, value))
            if value <= global_min:
                global_min = value
                x_opt = x
        min_optimizer.tell(solutions)
    print("finding max")
    max_optimizer = CMA(mean=np.zeros(dim), sigma=1.3, bounds=np.array([[-1, 1]] * dim))
    global_max = 1e10
    for generation in range(50):
        solutions = []
        for _ in range(max_optimizer.population_size):
            x = max_optimizer.ask()
            x_model = x_mean
            for idx, dim_i in enumerate(dim_chosen):
                x_model[dim_i] = x[idx]
            x_model = torch.from_numpy(x_model)
            x_model = torch.as_tensor(x_model, dtype=torch.float32).to(device)
            value = -model(x_model).cpu().detach().numpy()
            value = 4.0 * value
            solutions.append((x, value))
            if value <= global_max:
                global_max = value
                x_opt = x
        max_optimizer.tell(solutions)
    global_max = -global_max
    return global_min, global_max


device = torch.device("cuda:0")
model_path = 'model.pt'
model = torch.load(model_path)

# device = torch.device('cpu')
# model_path = 'model.pt'
# model = torch.load(model_path, map_location=torch.device('cpu'))

# MSLR mean datas
x_mean = np.array([-0.9495505830261964, -0.9770971829683402, -0.9134581676746579, -0.9261938089913366,
                   -0.9482613892976484, 0.6040201325053502, -0.8021616541219451, 0.08011218012616877,
                   -0.42498195715370024, 0.6670944234536929, -0.9020095599326677, -0.9990498257064565,
                   -0.9975265663282166, -0.9697920966896619, -0.8997809112101561, -0.8729183253038606,
                   -0.9341863103052187, -0.9333953301912015, -0.9275087019600515, -0.8727762966198558,
                   -0.9998605320061679, -0.9976992128922081, -0.9976314618374403, -0.9771054259487332,
                   -0.9998463154231735, -0.9975666540211799, -0.9977648149943444, -0.9971208786466262,
                   -0.9903730482525744, -0.997202656988022, -0.9935292216086944, -0.9952358154586683,
                   -0.9980979025245318, -0.9791334180253317, -0.9928641214924782, -0.9957631731230566,
                   -0.9963859178854532, -0.9956466512364345, -0.9853161748601489, -0.995253566252839,
                   -0.9999254733502664, -0.9999128479414915, -0.999986215693827, -0.9989792215216418,
                   -0.9999193112031527, -0.9983324276186766, -0.9540881365566223, -0.904296598093236,
                   -0.9131593105385585, -0.9981587219422159, -0.967434126333235, -0.9547368324317334,
                   -0.8381826233911442, -0.9131860112611055, -0.9531971721482307, -0.9329053282875062,
                   -0.8960056396423085, -0.6994050316085086, -0.8226605210582053, -0.9074150044722361,
                   -0.9513965765378891, -0.9282195037992725, -0.7723438939977124, -0.8726408665120559,
                   -0.9318354803619987, -0.9977785816680251, -0.9714974423558572, -0.9648191211677718,
                   -0.9793974604981637, -0.9916407354277638, -0.9632819757554524, -0.9976939139311306,
                   -0.9968198871493543, -0.9632037941028344, -0.9614256056438941, -0.4323192759448323,
                   -0.9977264049860414, -0.9966919440143636, -0.9718629185710204, -0.42283948570258967,
                   -0.9953780148978585, -0.9949439987306903, -0.9977895440431332, -0.9642345586755506,
                   -0.9948603600793513, -0.5814418180230199, -0.9962843208051017, -0.9958658767660298,
                   -0.9641889054530266, -0.5740025438299596, -0.9999755512168406, -0.9999140100423989,
                   -0.999987939542051, -0.9980565307876915, -0.999973247197116, 0.40003932704100675,
                   -0.8729569935476991, -0.22385918253079506, -0.614819962139391, 0.46659367834479815,
                   0.4824773196506431, -0.7749152730563075, 0.1702556642703592, -0.3444819298028742,
                   0.5450842706800176, -0.905120358165528, -0.9757372601846693, -0.9295416347013745,
                   -0.8531310653768752, -0.9018049937756292, 0.8732993347890201, 0.9170256395607282,
                   0.9304499120336222, 0.8098721717384064, 0.8775587895179736, 0.842479974917766,
                   0.9011215864159964, 0.9014214423616568, 0.7603089770001206, 0.846163998092285,
                   0.8620280161280893, 0.9178199527298004, 0.9329665760812776, 0.8287099121961946,
                   0.8668511182910867, -0.9625215000314864, -0.9493075401368025, 0.7380788550726509,
                   -0.9513180887317586, -0.4012152895708278, 0.09935417555902051, -0.8696142543407639,
                   -0.8089992477784796, -0.9999680914993644, -0.9996748394774676, -0.9999629607214002])

file = open('result.txt', 'w')
global_min, global_max = find_max_min(model, [i for i in range(136)], x_mean)
print(global_max, global_min)

# test the affect of each dimension
points = torch.ones(10000)
torch_mean = torch.from_numpy(x_mean)
torch_mean = torch.as_tensor(torch_mean, dtype=torch.float32)
points = torch_mean.resize(136, 1) * points
points = points.resize(10000, 136)
change = torch.linspace(-1, 1, 10000)
diff = torch.zeros(136)
file.write(f'=====================the effect of every dimension=============================\n')
for i in range(136):
    points_now = points.clone()
    points_now[:, i] = change
    result = model(points_now.to(device)).cpu()
    resul = 4.0 * result
    dim_max = torch.max(result)
    dim_min = torch.min(result)
    diff[i] = dim_max - dim_min
    diff[i] = dim_max - dim_min
print(f'the function value change in different dimension :\n{diff}')
file.write(f'the function value change in different dimension :{diff.tolist()}\n')
print(f'global max = {global_max}, global min = {global_min}')
file.write(f'global max = {global_max}, global min = {global_min}\n')
diff = diff / (float(global_max) - float(global_min))
print(f"the percentage of function value change in different dimension :\n{diff}")
file.write(f"the percentage of function value change in different dimension :{diff.tolist()}\n")


# test the affect of different subspaces, repeat iter_nums times
dim_num = [i for i in range(5, 136, 5)]
mean_eff = []
iter_nums = 50
for chosen_dim_num in dim_num:
    file.write(f'=====================number of chosen dimensions {chosen_dim_num}=============================\n')
    diff = np.zeros(iter_nums)
    for i in range(iter_nums):
        chosen_dim = random.sample([i for i in range(136)], chosen_dim_num)
        chosen_dim.sort()
        dim_min, dim_max = find_max_min(model, chosen_dim, x_mean)
        diff[i] = dim_max - dim_min
        file.write(f'in iteration {i}, the chosen dimensions are :{chosen_dim}\n')

    print(f'the function value change in different dimension :{diff}')
    file.write(f'the function value change in different dimension :\n{diff.tolist()}\n')
    print(f'global max = {global_max}, global min = {global_min}')
    file.write(f'global max = {global_max}, global min = {global_min}\n')
    diff = diff / (float(global_max) - float(global_min))
    print(f"the percentage of function value change in different dimension :{diff}")
    file.write(f"the percentage of function value change in different dimension :\n{diff.tolist()}\n")
    print(f"the mean of the percentage of function value change in different dimension :{diff.mean()}")
    file.write(f"the mean of the percentage of function value change in different dimension :{diff.mean()}\n")
    mean_eff.append(diff.mean())
file.write('=====================summary=============================\n')
file.write(f"the mean of the percentage of function value change in different dimension :\n{mean_eff}\n")

