###################################################
# This file is to a NN model
###################################################

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


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


class MSLR_Dataset(Dataset):
    def __init__(self, data):
        self.data_num = data.shape[0]
        self.y = data['rank'].to_numpy().reshape([self.data_num, 1])
        self.x = data.drop(['rank', 'id'], axis=1).to_numpy()
        self.y = torch.from_numpy(self.y)
        self.x = torch.from_numpy(self.x)
        self.y = torch.as_tensor(self.y, dtype=torch.float32)
        self.x = torch.as_tensor(self.x, dtype=torch.float32)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.data_num


def predict_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = map(lambda x: x.to(device), batch)
            predict = model(x)
            mse += F.mse_loss(predict, y, reduction='sum').item()
            sample_count += len(y)
    return mse / sample_count


def train(train_dataloader, valid_dataloader, model, learning_rate, num_epochs, device, model_path, l2_regularization):
    print('#### Start training!')
    opt = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=l2_regularization)

    best_loss = 1e5
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            x, y = map(lambda x: x.to(device), batch)
            predict = model(x)
            loss = F.mse_loss(predict, y, reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_samples += len(predict)

        model.eval()
        valid_mse = predict_mse(model, valid_dataloader, device)
        train_loss = total_loss / total_samples
        print(f"#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")
        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)


if __name__ == '__main__':
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    num_epochs = 1000
    learning_rate = 1e-4
    batch_size = 256
    l2_regularization = 1e-5
    model_path = './model2.pt'

    model = Net(136, 1).to(device)
    print('#### Preparing dataset...')
    data = pd.read_csv('data_normalized.csv')
    train_data, valid_data = train_test_split(data, test_size=0.1, random_state=0)
    train_dataset = MSLR_Dataset(train_data)
    valid_dataset = MSLR_Dataset(valid_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    print('#### Prepare dataset done!')

    train(train_dataloader, valid_dataloader, model, learning_rate, num_epochs, device, model_path, l2_regularization)

