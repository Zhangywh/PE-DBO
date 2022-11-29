###################################################
# This file describes the process of the Random Embedding, including:
# initRandomMatrix, randomEmbedding,
###################################################

import torch


def generate_random_matrix(low_dim, high_dim, exp_times, seed):
    """
    generate random matrix shape as (low_dim, high_dim)
    :param low_dim: dimension of low dim space
    :param high_dim: dimension of high dim space
    :param exp_times: exp times
    :param seed: random seed
    :return: random matrix
    """
    torch.manual_seed(seed)
    M = torch.randn([exp_times, low_dim, high_dim])
    for k in range(exp_times):
        for i in range(high_dim):
            for j in range(low_dim):
                M[k, j, i] = M[k, j, i] / torch.sqrt(torch.tensor([low_dim]))
    return M


def random_embedding(low_x, random_matrix):
    high_x = torch.mm(low_x, random_matrix)
    for i, elem in enumerate(high_x):
        for j, data in enumerate(elem):
            if high_x[i][j] > 1:
                high_x[i][j] = 1
            elif high_x[i][j] < -1:
                high_x[i][j] = -1
    return high_x

