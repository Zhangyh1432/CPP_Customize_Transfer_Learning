# CPP_Customize_Transfer_Learning

本仓库是论文《Customization Design of Circularized Polarized Phosphorescent Materials Guided by Transfer Learning》的官方实现。


## Directory Structure

```bash
.
├── ckpt
│   ├── autoencoder       # 预训练模型
│   ├── demo              # 正向预测demo模型权重
│   └── model             # 正向预测模型权重
├── data                  # 数据集目录
│   ├── all               # 所有数据
│   ├── demo              # 随机拆分数据
│   └── P_mol_spectra     # 磷光分子发光光谱
├── Pretrain              # 预训练
│   ├── ckpt              # 自编码器权重
│   ├── data              # 预训练数据集
│   └── train.py          # 预训练和验证
├── README.md             # 说明文档 / Documentation
├── result
├── scripts               # 代码目录
│   ├── baseline_rf.py   
│   ├── baseline_svr.py  
│   ├── baseline_xgb.py   
│   ├── demo.ipynb        # 演示脚本
│   ├── train.py          # 训练和验证脚本 
│   ├── reverse.py        # 逆向设计筛选脚本
│   └── utils.py          # 工具函数
└── virtual_database      # 虚拟数据库
```



## 环境依赖

```bash
CUDA                           11.8
torch                          2.0.0
pandas                         2.0.3
numpy                          1.24.2
scikit-learn                   1.3.2
matplotlib                     3.7.1
xgboost                        2.0.3
```

## 运行方法 / Usage

### demo

我们提供了 demo.ipynb，该 Jupyter Notebook 允许研究者方便地：

- 查看关键超参数设置；
- 执行模型训练与验证，完整复现论文中的实验过程；
- 获取关键结果，帮助理解模型的预测性能。

值得注意的是：在demo中，模型使用了一组随机拆分的训练集/测试集，这使得研究者可以更快速地获取实验结果。不过，文中使用了10折交叉验证，并给出了所有数据的平均预测性能。如果您需要完全遵循论文过程，请查看下一个内容。

### 预训练 

`Pretrain/data`目录中的数据集已压缩为压缩文件`data.zip`，请确保在运行训练脚本前已正确解压。

执行以下命令进行模型训练和验证：

```bash
python Pretrain/train.py
```

训练完成后，模型权重将保存在`Pretrain/ckpt`目录。如果需要用于迁移学习，请手动将预训练的 Autoencoder 权重移动到 `ckpt/autoencoder`目录，以便后续使用。

### 训练并评估正向预测模型 
您可以通过执行以下命令完成完整的模型的训练和评估：
```bash
python scripts/train.py
```
- 代码中的CombinedModel 类是实现我们提出方法的核心模块，其 forward 方法融合了迁移学习机制与两种参数编码因子，从而实现了模型性能的有效提升。
- **论文中使用的模型架构，实验超参数等均已在源码中配置完成。**
- 此外，您可以通过直接执行对应代码，如`scripts/baseline_rf.py`，获取相应的baseline的结果。

### 逆向设计筛选
我们提供了完整的虚拟数据库在`/virtual_database`下。你可以通过执行以下命令，获得虚拟筛选结果
```bash
python scripts/reverse.py
```
如果有自定义定制需求，需先在`scripts/reverse.py`中配置需求，包含：
- 最大/最小值
- 目标波段(需先转化为索引)
- g_lum范围，例如1.4-1.6(1.5左右)

逆向设计筛选的结果保存在`/result/reverse_design`下，结果文件中的每一行是一组候选参数，包含磷光分子、染料、厚度等。

## 结果 / Results
- 所有实验结果存储在 `/result/` 目录下，正向预测结果包含多个度量指标，包含mae，mse，rmse，r2和pearsonr。您可以在`/result/base_results_summary.csv`中获取论文的复现结果。

