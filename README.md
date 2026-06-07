# FedMalDet: Hardened Federated Malware Detection Framework

This repository contains the official implementation and reproducibility suite for the paper **"FedMalDet"**.

## 🛠️ Environment Setup
We recommend using an isolated Anaconda virtual environment running Python 3.9:

```bash
# Create and activate environment
conda create -n fedmaldet python=3.9 -y
conda activate fedmaldet

# Install absolute core requirements
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm
