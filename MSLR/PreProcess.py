###################################################
# This file is the pre-process of the MSLR dataset
###################################################

import pandas as pd

# generate labels
label = ['rank']
for i in range(136):
    label.append(str(i + 1))
print(label)

# read the dataset
train = pd.read_csv('train.txt', sep=' ', index_col=False, header=None, names=label)
test = pd.read_csv('test.txt', sep=' ', index_col=False, header=None, names=label)
valid = pd.read_csv('valid.txt', sep=' ', index_col=False, header=None, names=label)

# extract ID
def del_idx(item):
    item = item.split(':')
    return item[-1]


# extract features
def del_idx_f(item):
    item = item.split(':')
    return float(item[-1])


# concatenate data and process
data = pd.concat([train, test, valid], ignore_index=True)
print('id')
data['id'] = data['id'].apply(del_idx)
for idx in label[2:]:
    print(idx)
    data[idx] = data[idx].apply(del_idx_f)
print(data)
data.describe()
# keep original datas
data.to_csv('data.csv', index=False)

# scale the original data
data = pd.read_csv('data.csv')
for idx in label[2:]:
    print(idx)
    min = data[idx].min()
    data[idx] = data[idx] - min
    max = data[idx].max()
    print(max)
    data[idx] = data[idx] / max
    data[idx] = 2 * data[idx] - 1
    print('\n')

# rank normalization
data['rank'] = data['rank'] / 4.0
# print(data)
data.to_csv('data_normalized.csv', index=False)
data.describe()

# rank=4
data = data[(data['rank'] == 4)]
print(data)

# describe the mean
result = []
for i in label:
    print(i)
    result.append(data[i].mean())
print(result)
