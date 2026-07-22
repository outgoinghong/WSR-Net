# WSR-Net

**Wavelet-Driven Spectral-Spatial Reciprocal Network for Hyperspectral Unmixing With Spectral Variability**

This repository provides the official implementation of WSR-Net, including dataset preparation, demo usage, training, and evaluation scripts. 

## 📁 Dataset & Preparation

To ensure reproducibility, we provide the prepared real-world hyperspectral datasets directly in this repository. 
The datasets (`houston.mat` and `moffett.mat`) are located in the root directory. 

*(No extra download or path configuration is required. The scripts are hard-coded to load them automatically.)*

## 🔧 Model Weights 

Since WSR-Net is an **unsupervised** blind unmixing autoencoder, it performs end-to-end training directly on the input hyperspectral image. 
Therefore, **no large-scale pre-trained foundation model weights are required**. The physical base endmembers are dynamically initialized via Vertex Component Analysis (VCA) during runtime.

## 🛠 Installation

The code was tested in the environment of **Python 3.8**, **PyTorch 1.9.1**, and **CUDA 11.3** (Windows 10 / Linux).

You can easily install the required dependencies using pip:
```bash
# Install PyTorch
pip install torch==1.9.1+cu113 torchvision torchaudio -f [https://download.pytorch.org/whl/cu113/torch_stable.html](https://download.pytorch.org/whl/cu113/torch_stable.html)

# Install other required dependencies
pip install numpy scipy scikit-learn matplotlib PyWavelets

🚀 Demo & Usage
Once the dataset is ready and dependencies are installed, you can run the unmixing process directly. We provide ready-to-use scripts for different test cases.

To run WSR-Net on the Houston dataset:

