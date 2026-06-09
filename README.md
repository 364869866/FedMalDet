# FedMalDet: An Efficient and Privacy-Preserving Multi-Modal Federated Learning Framework for Malware Detection

This repository contains the official implementation of the FedMalDet framework, a decentralized and communication-efficient malware classification system under non-IID data skew.

## Environment Setup
Recommended environment: Python 3.9, PyTorch >= 2.1.0, CUDA >= 12.1.

```bash
conda create -n fedmaldet python=3.9 -y
conda activate fedmaldet
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm

Dataset Preparation
MalImg Dataset: Download the original image repository from Kaggle and position it within the designated local directory.

VirusShare Dataset: Extract the 295-dimensional PE structural features as specified in our unified multi-modal telemetry configuration.

Running Experiments
To replicate all empirical evaluations, convergence trajectories, and privacy-utility ablation benchmarks reported in the manuscript, execute the central tracking controller:
python main_reproduce.py
All empirical tracking results, stability charts, and telemetry log records will be securely cached inside the ./experiment_results directory.
