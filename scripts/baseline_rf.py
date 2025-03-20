import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

from utils import get_loss_function, calculate_metrics_all, load_data, plot_scatter, Autoencoder, FlexibleMLP, set_seed

    
# Hyperparameters and configuration
input_dim = 125  # Update this if needed
output_dim_autoencoder = 121

config = {"mol": 'all', "P_spectra_dim": 101, 
     "input_csv_path": '/root/CPP_Customize_Transfer_Learning/data/all/combined_x.csv',
     "label_csv_path": '/root/CPP_Customize_Transfer_Learning/data/all/combined_y.csv',
     "output_plot_dir": '/root/CPP_Customize_Transfer_Learning/pred/all'} #optional

file_paths = ['/root/CPP_Customize_Transfer_Learning/data/P_mol_spectra/mol1-p.xlsx', 
              '/root/CPP_Customize_Transfer_Learning/data/P_mol_spectra/mol2-p.xlsx', 
              '/root/CPP_Customize_Transfer_Learning/data/P_mol_spectra/mol3-p.xlsx']

loss_function_name = 'mse'  # Change this to 'mae' or 'huber' as needed
sampling_frequency = 15
freeze = False
output_dim_combined = 501
onehot_dim = 7   # Adjust as needed
batch_size = 128
learning_rate = 5e-5
lr_decay_step = 200  # Step size for learning rate decay
lr_decay_gamma = 0.7  # Decay factor
augment_times = 10
noise_std = 1e-7
test_size = 0.1
k_folds = 10
autoencoder_checkpoint = '/root/CPP_Customize_Transfer_Learning/ckpt/autoencoder/1500.pth'
ckpt_dir = "/root/CPP_Customize_Transfer_Learning/ckpt/model"
results_file_path = '/root/CPP_Customize_Transfer_Learning/result/baseline_rf_results_summary.csv'

# Different configurations for epochs and random seeds
random_seeds = [2024]



class CombinedModel(nn.Module):
    def __init__(self, autoencoder, P_spectra_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.P_spectra_dim = P_spectra_dim
        self.output_dim = output_dim
    
    def forward(self, x, P_spectra):
        features = self.autoencoder.encoder(x)
        combined = torch.cat((features, P_spectra), dim=1)
        return combined


class CustomDataset(Dataset):
    def __init__(self, inputs, labels, file_paths, sampling_frequency, random_seed, augment_times=augment_times, noise_std=noise_std, is_train=True):
        self.is_train = is_train
        self.file_paths = file_paths  # List of file paths for head tensors
        self.P_spectra_tensors = []
        self.sampling_frequency = sampling_frequency
        
        # Load all head tensors
        for file_path in file_paths:
            df = pd.read_excel(file_path)
            list2 = df.iloc[:, 1].tolist()
            self.P_spectra_tensors.append(torch.tensor(list2, dtype=torch.float32))
        
        if is_train:
            augmented_inputs = []
            augmented_labels = []
            np.random.seed(random_seed)
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
        
        # Sample the head tensor
        P_spectra_tensor = P_spectra_tensor[::self.sampling_frequency]
        
        return torch.tensor(self.inputs[idx, :-2]), torch.tensor(self.labels[idx]), P_spectra_tensor


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   

    for random_seed in random_seeds:
        results = []
        set_seed(random_seed)
        print(f"Running for mol={config['mol']}, random_seed={random_seed}")
        
        if not os.path.exists(results_file_path):
            with open(results_file_path, 'w') as f:
                f.write("mol,random_seed,mae,mse,rmse,r2,pearson\n")

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
            
            combined_model = CombinedModel(autoencoder, P_spectra_dim=config['P_spectra_dim'], output_dim=output_dim_combined).to(device)
            
            train_inputs, test_inputs = inputs[train_index], inputs[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            
            train_dataset = CustomDataset(train_inputs, train_labels, file_paths, sampling_frequency, random_seed = random_seed, is_train=True)
            test_dataset = CustomDataset(test_inputs, test_labels, file_paths, sampling_frequency, random_seed = random_seed, is_train=False)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=set_seed(random_seed))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, worker_init_fn=set_seed(random_seed))
      
            
            # Extract features for training set
            combined_model.eval()
            train_features = []
            train_targets = []
            for inputs_tr, labels_tr, P_spectra_tensor in train_loader:
                inputs_tr, labels_tr, P_spectra_tensor = inputs_tr.to(device), labels_tr.to(device), P_spectra_tensor.to(device)
                with torch.no_grad():
                    combined_features = combined_model(inputs_tr, P_spectra_tensor)
                train_features.append(combined_features.cpu().numpy())
                train_targets.append(labels_tr.cpu().numpy())
            
            train_features = np.concatenate(train_features, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            
            # Extract features for test set
            test_features = []
            test_targets = []
            for inputs_te, labels_te, P_spectra_tensor in test_loader:
                inputs_te, labels_te, P_spectra_tensor = inputs_te.to(device), labels_te.to(device), P_spectra_tensor.to(device)
                with torch.no_grad():
                    combined_features = combined_model(inputs_te, P_spectra_tensor)
                test_features.append(combined_features.cpu().numpy())
                test_targets.append(labels_te.cpu().numpy())
            
            test_features = np.concatenate(test_features, axis=0)
            test_targets = np.concatenate(test_targets, axis=0)
            
            # Train RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
            rf_model.fit(train_features, train_targets)
            
            # Predict on test set
            predictions = rf_model.predict(test_features)
            
            all_predictions.extend(predictions.flatten())
            all_labels.extend(test_targets.flatten())
            
            print(f'Fold {fold+1} completed')
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        mae, mse, rmse, r2, pearson_corr = calculate_metrics_all(all_labels, all_predictions)
        print(f'Average MAE: {mae}, MSE: {mse}, RMSE:{rmse}, R2: {r2}, Pearson: {pearson_corr} for mol={config["mol"]}, random_seed={random_seed}')
        
        results.append({
            "mol": config["mol"],
            "random_seed": random_seed,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "pearson": pearson_corr
        })

        with open(results_file_path, 'a') as f:
            f.write(f"{config['mol']},{random_seed},{mae},{mse},{rmse},{r2},{pearson_corr}\n")

if __name__ == "__main__":
    main()
