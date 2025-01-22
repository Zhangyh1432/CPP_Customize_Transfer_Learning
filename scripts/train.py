import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import random
from utils import get_loss_function , calculate_metrics_all, load_data, plot_scatter, Autoencoder, FlexibleMLP

# Hyperparameters and configuration
input_dim = 125
output_dim_autoencoder = 121

config = {"mol": 'all', "P_spectra_dim": 101, 
     "input_csv_path": '/root/cpl_prediction/data/all/combined_x.csv',
     "label_csv_path": '/root/cpl_prediction/data/all/combined_y.csv',
     "output_plot_dir": '/root/cpl_prediction/pred/all'}

file_paths = ['/root/cpl_prediction/data/P_mol_spectra/mol1-p.xlsx', 
              '/root/cpl_prediction/data/P_mol_spectra/mol2-p.xlsx', 
              '/root/cpl_prediction/data/P_mol_spectra/mol3-p.xlsx']

loss_function_name = 'mae' 
sampling_frequency = 15
freeze = False
output_dim_combined = 501
onehot_dim = 7  
batch_size = 128
learning_rate = 5e-5
lr_decay_step = 200  
lr_decay_gamma = 0.7 
augment_times = 10
noise_std = 1e-7
test_size = 0.1
k_folds = 10
autoencoder_checkpoint = '/root/cpl_prediction/ckpt/autoencoder/1500.pth'
ckpt_dir = "/root/cpl_prediction/ckpt/model"
results_file_path = '/root/cpl_prediction/result/base_results_summary.csv'

num_epochs = 300
random_seed = 6547


class CombinedModel(nn.Module):
    def __init__(self, autoencoder, P_spectra_dim, output_dim, mlp_output_dim, embedding_dim):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.P_spectra_dim = P_spectra_dim
        self.output_dim = output_dim
        
        self.base_mlp = FlexibleMLP([256 + P_spectra_dim, 1024, 2048, 1024, output_dim])
        self.weight_mlp = FlexibleMLP([3, 5, 1])
        self.offset_mlp = FlexibleMLP([embedding_dim, 64, 128, mlp_output_dim])
        self.embedding = nn.Parameter(torch.randn(mlp_output_dim))
    
    def forward(self, input):
        
        x = input[:, :125]
        offset_feat = input[:, 125:132]
        weight_feat = input[:, 132:135]
        P_spectra_tensor = input[:, 135:]  #Phosphorescent molecular luminescence spectra
        
        features = self.autoencoder.encoder(x)
        combined = torch.cat((features, P_spectra_tensor), dim=1)
        base_signal = self.base_mlp(combined)
        
        weight_factor = self.weight_mlp(weight_feat)
        weight_factor = torch.clamp(weight_factor, min=-1.5, max=1.5)
        
        mlp_output = self.offset_mlp(offset_feat)
        offset_factor = mlp_output * self.embedding
        
        final_output = base_signal + offset_factor
        final_output = final_output * weight_factor
        
        return final_output


class CustomDataset(Dataset):
    def __init__(self, inputs, labels, file_paths, sampling_frequency, augment_times=augment_times, noise_std=noise_std, is_train=True):
        self.is_train = is_train
        self.file_paths = file_paths
        self.P_spectra_tensors = []
        self.sampling_frequency = sampling_frequency

        for file_path in file_paths:
            df = pd.read_excel(file_path)
            list2 = df.iloc[:, 1].tolist()
            self.P_spectra_tensors.append(torch.tensor(list2, dtype=torch.float32))
        
        if is_train:
            augmented_inputs = []
            augmented_labels = []
            for i in range(len(inputs)):
                for _ in range(augment_times):
                    noise = np.random.normal(0, noise_std, inputs[i, :-6].shape).astype(np.float32)
                    augmented_inputs.append(np.concatenate((inputs[i, :-6] + noise, inputs[i, -6:])))
                    augmented_labels.append(labels[i])
            self.inputs = np.array(augmented_inputs)
            self.labels = np.array(augmented_labels)
        else:
            self.inputs = inputs
            self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        P_spectra_tensor_index = int(self.inputs[idx, -1]) - 1
        P_spectra_tensor = self.P_spectra_tensors[P_spectra_tensor_index]
        
        # Sample the P_spectra_tensor
        P_spectra_tensor = P_spectra_tensor[::self.sampling_frequency]
        
        extra_features = self.one_hot_encode(self.inputs[idx, -5]) + self.one_hot_encode(self.inputs[idx, -2]) + [self.inputs[idx, -4]]
        weight_features = self.one_hot_encode(self.inputs[idx, -5])
        
        combined_inputs = np.concatenate((self.inputs[idx, :-2], extra_features, weight_features, P_spectra_tensor.numpy()))
        
        return torch.tensor(combined_inputs, dtype=torch.float32), torch.tensor(self.labels[idx])

    @staticmethod
    def one_hot_encode(value):
        if value in [0, 1, 2]: 
            one_hot = [0] * 3
            one_hot[int(value)] = 1
        elif value in [30, 48, 80]: 
            one_hot = [0] * 3
            if value == 30:
                one_hot[0] = 1
            elif value == 48:
                one_hot[1] = 1
            elif value == 80:
                one_hot[2] = 1
        else:
            raise ValueError("Unexpected value for one-hot encoding")
        return one_hot


def main():
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w') as f:
            f.write("mol,epochs,random_seed,mae,mse,r2,pearson\n")
    print(f"Running for mol={config['mol']}, epochs={num_epochs}, random_seed={random_seed}")
    
    inputs, labels = load_data(config["input_csv_path"], config["label_csv_path"])
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    all_predictions = []
    all_labels = []

    for fold, (train_index, test_index) in enumerate(kf.split(inputs)):
        print(f'Fold {fold+1}')
        
        autoencoder = torch.load(autoencoder_checkpoint)
        autoencoder = autoencoder.to(device)
        
        if freeze:
            for param in autoencoder.parameters():
                param.requires_grad = False
                
        combined_model = CombinedModel(autoencoder, P_spectra_dim=config['P_spectra_dim'], output_dim=output_dim_combined, mlp_output_dim=output_dim_combined, embedding_dim=onehot_dim).to(device)
        
        criterion = get_loss_function(loss_function_name)
        optimizer = optim.Adam(
            [param for param in combined_model.parameters() if param.requires_grad],
            lr=learning_rate
        )
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
        
        train_inputs, test_inputs = inputs[train_index], inputs[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        
        train_dataset = CustomDataset(train_inputs, train_labels, file_paths, sampling_frequency, is_train=True)
        test_dataset = CustomDataset(test_inputs, test_labels, file_paths, sampling_frequency, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        for epoch in range(num_epochs):
            combined_model.train()
            total_loss = 0
            for inputs_tr, labels_tr in train_loader:
                inputs_tr, labels_tr = inputs_tr.to(device), labels_tr.to(device)
                
                optimizer.zero_grad()
                
                outputs = combined_model(inputs_tr)
                loss = criterion(outputs, labels_tr)
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            print(f'Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}')
        
        combined_model.eval()
        test_loss = 0
        fold_predictions = []
        fold_labels = []
        
        for sample_idx, (inputs_te, labels_te) in enumerate(test_loader):
            inputs_te, labels_te = inputs_te.to(device), labels_te.to(device)
            with torch.no_grad():
                outputs = combined_model(inputs_te)
                loss = criterion(outputs, labels_te)
                test_loss += loss.item()
                fold_predictions.append(outputs.cpu().numpy().flatten())
                fold_labels.append(labels_te.cpu().numpy().flatten())

        
        all_predictions.extend(np.concatenate(fold_predictions, axis=0))
        all_labels.extend(np.concatenate(fold_labels, axis=0))
        
        print(f'Test Loss for Fold {fold+1}: {test_loss/len(test_loader)}')

        # Save the model weights after each fold
        model_save_path = os.path.join(ckpt_dir, f"model_{config['mol']}_epochs_{num_epochs}_seed_{random_seed}_fold_{fold+1}.pth")
        torch.save(combined_model, model_save_path)
        print(f"Model weights saved to {model_save_path}")
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    mae, mse, rmse, r2, pearson_corr = calculate_metrics_all(all_labels, all_predictions)
    
    print(f'Average MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}, Pearson: {pearson_corr} for mol={config["mol"]}, epochs={num_epochs}, random_seed={random_seed}')


    with open(results_file_path, 'a') as f:
        f.write(f"{config['mol']},{num_epochs},{random_seed},{mae},{mse},{rmse},{r2},{pearson_corr}\n")
    
    plot_scatter(all_labels, all_predictions, config["mol"], num_epochs, random_seed, config['output_plot_dir'])

if __name__ == "__main__":
    main()