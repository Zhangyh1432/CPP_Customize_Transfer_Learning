import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, label_path = None):
        self.data_path = data_path
        self.label_path = label_path
        self.data = pd.read_csv(data_path).values
        if label_path!=None:
            self.labels = pd.read_csv(label_path).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label_path!=None:
            data = torch.tensor(self.data[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return data, label
        else:
            data = torch.tensor(self.data[idx], dtype=torch.float32)
            return data

# 编解码器网络
class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True), 
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, output_dim), 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batchsize = 256
    epochs = 1800
    interval = 50

    model = Autoencoder(input_dim=125, output_dim=121).to(device)  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    dataset = CustomDataset('/root/CPP_Customize_Transfer_Learning/Pretrain/data/augmented_data.csv', '/root/CPP_Customize_Transfer_Learning/Pretrain/data/augmented_labels.csv')
    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d loss: %.6f' % (epoch + 1, running_loss / len(data_loader)))
        if(epoch%interval == 0):
            torch.save(model, "/root/CPP_Customize_Transfer_Learning/Pretrain/ckpt/" + str(epoch) + ".pth")

    test_dataset = CustomDataset('/root/CPP_Customize_Transfer_Learning/Pretrain/data/test_data.csv', '/root/CPP_Customize_Transfer_Learning/Pretrain/data/test_labels.csv')
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    # 测试模型
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) 
            output = model(data)
            test_loss += criterion(output, target).item()
            

    print('Test set loss: %.3f' % (test_loss / len(test_loader)))