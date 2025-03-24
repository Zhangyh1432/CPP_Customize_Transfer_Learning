# CPP_Customize_Transfer_Learning

This is the official implementation of: Customization Design of Circularized Polarized Phosphorescent Materials Guided by Transfer Learning.

## Directory Structure

```bash
.
├── ckpt
│   ├── autoencoder       # Pretrained models
│   ├── demo              # Forward Prediction Model weights for demo
│   └── model             # Forward Prediction Model weights
├── data                  # Dataset directory
│   ├── all               # Complete dataset
│   ├── demo              # Random-split dataset for demo
│   └── P_mol_spectra     # Phosphorescent molecular emission spectra
├── Pretrain              # Pretrain
│   ├── ckpt              # autoencoder model weight
│   ├── data              # Pretrain dataset  
│   └── train.py          # Pretrain Model training and validation script    
├── README.md             # Documentation
├── result                # Results
├── scripts               # Code directory
│   ├── baseline_lgb.py  
│   ├── baseline_rf.py   
│   ├── baseline_svr.py  
│   ├── baseline_xgb.py   
│   ├── demo.ipynb        # Demo script
│   ├── train.py          # Forward Prediction Model training and validation script  
│   ├── reverse.py        # Inverse design Screening script  
│   └── utils.py          # Utility functions
└── virtual_database      # Virtual database
```


## Environment

```bash
CUDA                           11.8
torch                          2.0.0
pandas                         2.0.3
numpy                          1.24.2
scikit-learn                   1.3.2
matplotlib                     3.7.1
xgboost                        2.0.3
```

## Usage

### Demo

We provide demo.ipynb, a Jupyter Notebook that allows researchers to:

- View key hyperparameter settings.
- Perform model training and validation, fully reproducing the experimental process described in the paper.
- Obtain key results to better understand the model’s predictive performance.

Note: In the demo, the model uses a randomly split training/test set, enabling researchers to obtain experimental results more quickly. However, the paper employs 10-fold cross-validation and reports the average prediction performance across all data. If you wish to strictly follow the methodology described in the paper, please refer to the next section.

### Pretraining 

The dataset in the `Pretrain/data` directory is compressed as `data.zip`. Ensure it is properly extracted before running the training script.

Execute the following command to train and validate the pretrain model:

```bash
python Pretrain/train.py
```

Once training is complete, the model weights will be saved in the `Pretrain/ckpt` directory. If needed for transfer learning, manually move the pre-trained autoencoder weights to `ckpt/autoencoder` for further use.

### Training and Evaluating the Forward Prediction Model

To train and evaluate the complete model, run:

```bash
python scripts/train.py
```

- **The CombinedModel class in the code is the core module of our proposed method. Its forward method integrates transfer learning mechanisms and two parameter encoding factors, significantly enhancing model performance.**
- **The model architecture, experimental hyperparameters, and other configurations used in the paper are pre-configured in the source code.**
- Additionally, you can run the corresponding baseline scripts (e.g., `/scripts/baseline_rf.py`) to obtain baseline results.



### Inverse Design and Screening

The complete virtual database is provided under `/virtual_database`. You can obtain the virtual screening results by running:

```bash
python scripts/reverse.py
```

For custom screening requirements, you need to configure them in `scripts/reverse.py`, including:

- Max/Min
- Target wavelength range (converted to indices)
- g_lum range, e.g., 1.4–1.6 (around 1.5)

The results of inverse design screening are saved in `/result/reverse_design`. Each row in the result file represents a set of candidate parameters, including phosphorescent molecules, dyes, thickness, etc.

## Results

All experimental results are stored in the `/result/` directory.
The forward prediction results include multiple evaluation metrics: MAE, MSE, RMSE, R², and Pearson correlation coefficient.
You can find the reproduced results from the paper in `/result/base_results_summary.csv`.
