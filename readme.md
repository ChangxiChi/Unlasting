##  Unlasting: Unpaired Single-Cell Multi-Perturbation Estimation by Dual Conditional Diffusion Implicit Bridges
## 🏁 Getting Started

### Dataset Preparation
The dataset can be downloaded from the following URL:
```bash
# Genetic Perturbation Dataset:
Adamson: https://dataverse.harvard.edu/api/access/datafile/6154417
Norman: https://dataverse.harvard.edu/api/access/datafile/6154020
```
```bash
# Molecular Perturbation Dataset:
The processed dataset will be uploaded in future releases.
```

### Environment Requirements
1. This project requires an environment compatible with **[scGPT](https://github.com/bowang-lab/scGPT)**.  
👉 It is strongly recommended to first set up the environment according to scGPT’s requirements.  
2. The pretrained scGPT model parameters are required for training and fine-tuning.  
Please download them from the following link and place them into the `Unlasting/Dataset/scGPT-main/save/` directory:

🔗 [Google Drive – scGPT pretrained models](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)

3. Additionally, please install **[Uni-Mol](https://github.com/dptech-corp/Uni-Mol)** for molecular representation extraction.  

 Recommended Setup
- Python 3.10+
- CUDA 11.8 (for GPU acceleration)
- PyTorch 2.1.2 (with GPU support)
- scGPT 0.2.1
- Uni-Mol 0.1.2.post2
- ...

## 📁 Repository Structure  
```plaintext
Unlasting/
├── Dataset/                    
│   ├── gene/                   # Dictionary, create it and download genetic perturbation dataset to here
│   ├── molecular/              # Dictionary, create it and download molecular perturbation dataset to here
│   ├── scGPT-main/
│   │   ├── save/               # contain the pretrained scGPT parameters
│   │   │   ├── args.json
│   │   │   ├── best_model.pt
│   │   │   ├── vocab.json
│   ├── Datasets.py
│   ├── GRN.py
│   ├── MoleEmb.py
│   ├── Preprocess.py
├── model.py
├── train_model.py
├── train_mask_model.py
├── test.py
└── ...          
```

## 🏋️ Training

After completing dataset preparation and environment setup, you can start training the models.

1. Train the main model:

```bash
python train_model.py
```

2. Train the mask model:

```bash
python train_mask_model.py
```

## Evaluation
See 📁 [`test.py`](./test.py) for details.