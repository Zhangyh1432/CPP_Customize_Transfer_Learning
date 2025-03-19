import os
import random 
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

def get_loss_function(loss_name):
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'huber':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def calculate_metrics_all(y_true, y_pred, vector_length=501):

    num_samples = y_true.shape[0] // vector_length
    
    y_true = np.reshape(y_true, (num_samples, vector_length))
    y_pred = np.reshape(y_pred, (num_samples, vector_length))

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    pearson_corr_values = []
    for i in range(vector_length):
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        pearson_corr_values.append(corr)
    average_pearson_corr = np.mean(pearson_corr_values)
    
    return mae, mse, rmse, r2, average_pearson_corr


def load_data(input_csv, label_csv):
    inputs = pd.read_csv(input_csv, header=None).values.astype(np.float32)
    labels = pd.read_csv(label_csv, header=None).values.astype(np.float32)
    return inputs, labels

    
    
def plot_scatter(true_labels, predicted_labels, mol, epochs, seed, output_dir, sample_step=100):
    sampled_true_labels = true_labels[::sample_step]
    sampled_predicted_labels = predicted_labels[::sample_step]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(sampled_true_labels, sampled_predicted_labels, alpha=0.5, label='Predictions')
    plt.plot([min(sampled_true_labels), max(sampled_true_labels)], [min(sampled_true_labels), max(sampled_true_labels)], 'r--', label='y=x')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot of True vs Predicted Values for mol={mol}, epochs={epochs}, seed={seed}')
    plt.legend()
    scatter_plot_path = os.path.join(output_dir, f'scatter_{mol}_epochs_{epochs}_seed_{seed}.png')
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f'Scatter plot saved to {scatter_plot_path}')
    
    
    
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


class FlexibleMLP(nn.Module):
    def __init__(self, layer_dims):
        super(FlexibleMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Build MLP layers
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # Don't add activation after the last layer
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 如果使用多个 worker，需要在 DataLoader 里设置 worker_init_fn
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return worker_init_fn
