###################################################
# The Testing Functions APIs
# Domain of Testing Functions: [-1, 1]
###################################################

import torch
import numpy as np

# The dimension of x (D) must >= de (effective dimension)
# Set the effective dimension
de = 10

def levy_function(x):
    # [-1, 1] --> [-10, 10]
    x = x.clone() * 10
    dim = x.shape[1]
    num = x.shape[0]

    # paras
    c = 0.1
    K = 1000
    # shift
    x = x - c

    # the levy function
    w = 1 + (x - 1) / 4
    a = torch.sin(np.pi * w[..., 0]) ** 2
    b = torch.zeros(num)
    for i in range(de-1):
        b += ((w[..., i] - 1) ** 2) * (1 + 10 * torch.sin(np.pi * w[..., i] + 1) ** 2)
    c = ((w[..., de-1] - 1) ** 2) * (1 + (torch.sin(2 * np.pi * w[..., de-1])) ** 2)
    # min
    ans = a + b + c

    # the disturbance
    dis = torch.zeros(num)
    for i in range(de, dim):
        dis += x[..., i] ** 2
    ans += dis / K

    return - ans


def dixonPrice_function(x):
    # [-1, 1] --> [-10, 10]
    x = x.clone() * 10
    dim = x.shape[1]
    num = x.shape[0]

    # paras
    c = 0.2
    K = 1000
    # shift
    x = x - c

    # the dixon function
    a = (x[..., 0] - 1) ** 2
    b = torch.zeros(num)
    for i in range(1, de):
        b += (i + 1) * (2 * x[..., i] ** 2 - x[..., i-1]) ** 2
    # min
    ans = a + b

    # the disturbance
    dis = torch.zeros(num)
    for i in range(de, dim):
        dis += x[..., i] ** 2
    ans += dis / K

    # max
    return - ans


def sphere_function(x):
    # [-1, 1] --> [-5.12, 5.12]
    x = x.clone() * 5.12
    dim = x.shape[1]
    num = x.shape[0]

    # paras
    c = 0.2
    K = 1000
    # shift
    x = x - c

    # the sphere function
    a = torch.zeros(num)
    for i in range(de):
        a += x[..., i] ** 2
    # min
    ans = a

    # the disturbance
    dis = torch.zeros(num)
    for i in range(de, dim):
        dis += x[..., i] ** 2
    ans += dis / K

    # max
    return - ans


def ackley_function(x):
    # [-1, 1] --> [-32.768, 32.768]
    x = x.clone() * 32.768
    dim = x.shape[1]
    num = x.shape[0]

    # paras
    c = 0.2
    K = 1000
    # shift
    x = x - c

    # the ackley function
    a = torch.zeros(num)
    b = torch.zeros(num)
    for i in range(de):
        a += x[..., i] ** 2
        b += np.cos(2 * np.pi * x[..., i])
    a = a / de
    b = b / de
    # min
    ans = -20 * np.exp(-0.2 * a) - np.exp(b) + 20 + np.exp(1)

    # the disturbance
    dis = torch.zeros(num)
    for i in range(de, dim):
        dis += x[..., i] ** 2
    ans += dis / K

    # max
    return - ans



