# PU-Matrix-Screener: Physics-Constrained Positive-Unlabeled Deep Learning for "Ionless" MALDI Matrix Discovery

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-blue.svg)](https://www.rdkit.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-ChemBERTa-yellow.svg)](https://huggingface.co/)

**PU-Matrix-Screener** is an advanced computational framework designed to discover novel, high-performance, and "Ionless" (background-free) matrices for high-resolution Spatial Metabolomics and Single-Cell Mass Spectrometry Imaging (MSI). 

Addressing the critical challenge of absent "negative" matrix samples, this pipeline employs a **Physics-Constrained Positive-Unlabeled (PU) Learning strategy**, combining Density Functional Theory (DFT) attributes with Pre-trained Language Models (ChemBERTa) to navigate massive uncharted chemical spaces.

<p align="center">
  <img src="images/pu_architecture.png" alt="PU-Matrix-Screener Architecture" width="800"/>
</p>
<p align="center"><b>Figure 1: The PU-Matrix-Screener Framework.</b> Integration of DFT physics (Ei, HOMO-LUMO) and ChemBERTa embeddings, followed by physics-constrained Manifold Clustering in a PU-Learning latent space, terminating in a funnel screening for SA score and MW constraints.</p>

---

## 💻 System Requirements
* **Hardware:** NVIDIA GPU (RTX 3090/4090 or A100) highly recommended for ChemBERTa embedding and Neural Network training.
* **Dependencies:** `pytorch`, `transformers`, `rdkit`, `umap-learn`, `scikit-learn`.

## 🛠️ Installation

```bash
git clone [https://github.com/YourName/PU-Matrix-Screener.git](https://github.com/YourName/PU-Matrix-Screener.git)
cd PU-Matrix-Screener
conda env create -f environment.yml
conda activate pu_matrix_env
