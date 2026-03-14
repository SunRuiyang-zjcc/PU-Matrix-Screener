# RL-Matrix-Screener: Reinforcement Learning Driven Dual-Mode Matrix Design

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**RL-Matrix-Screener** is a cutting-edge computational framework that integrates Graph Neural Networks (GNEprop), Surrogate QSAR Modeling, and Reinforcement Learning (RL) to discover novel dual-mode organic matrices for Single-Cell MSI.


<p align="center"><b>Figure 1: RL-Matrix-Screener Architecture.</b> A Reinforcement Learning agent explores the continuous GNEprop latent space to maximize Ionization Efficiency and minimize Crystal Size, followed by KNN mapping to the MCE commercial library.</p>

---

## 💻 System Requirements
* **Hardware:** CPU is sufficient for the RL agent, but a GPU (NVIDIA RTX series) is highly recommended for GNEprop embedding.
* **OS:** Linux/WSL2 or Windows 10/11.

## 🛠️ Installation

```bash
git clone [https://github.com/YourName/RL-Matrix-Screener.git](https://github.com/YourName/RL-Matrix-Screener.git)
cd RL-Matrix-Screener
conda env create -f environment.yml
conda activate rl_matrix_env