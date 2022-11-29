###################################################
# The Datasets APIs
###################################################
import torch
import numpy as np
from torch import nn
from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset as CEMBLD
from design_bench.oracles.sklearn import RandomForestOracle as RFO


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        for i in range(1):
            x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x


def MSLR(x):
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    x = x.clone()
    model = Net(136, 1)
    model = torch.load('../MSLR/model.pt')
    dim = x.shape[1]
    num = x.shape[0]
    ans = 4 * model(x.to(device)).cpu().detach()
    return ans.reshape(1, num)[0]


dataset = CEMBLD()
oracle = RFO(dataset, noise_std=0.0)
bound = [[12, 161, 184, 184, 45, 119, 60, 60, 184, 119, 184, 184, 60, 184, 184, 401, 115, 84, 84, 84, 85, 85, 85, 84,
          60, 60, 60, 60, 48, 48, 13],
         [12, 15, 15, 15, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def ChEMBL(x):
    x = x.clone().numpy()
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    for i in range(31):
        x[:, i] = (x[:, i] + 1) / 2 * (bound[0][i] - bound[1][i]) + bound[1][i]
    x = np.array(x, dtype=np.int32)
    y = oracle.predict(x)
    y = torch.from_numpy(y)
    y = torch.as_tensor(y, dtype=torch.float32)
    n = y.shape[0]
    y = y.resize(1, n)
    return y[0]
