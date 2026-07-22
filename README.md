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

You can easily install all required dependencies using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## 🚀 Demo & Usage
Once the dataset is ready and dependencies are installed, you can run the unmixing process directly. We provide ready-to-use scripts for different test cases.

To run WSR-Net on the Houston dataset:
```bash
python WSR-Net_houston.py
```
To run WSR-Net on the Moffett dataset:
```bash
python WSR-Net_moffett.py
```
## 🏋️‍♂️ Training & 📊 Evaluation
Unlike supervised segmentation tasks, hyperspectral unmixing is an unsupervised optimization process. The WSR-Net_*.py scripts will perform training and evaluation concurrently.
### Expected Outputs

After running the scripts, the following results will be generated:

1. **Console Metrics**: The terminal will display the training progress and print the final quantitative evaluation metrics: Mean Spectral Angle Distance (SAD) and Mean Root Mean Square Error (RMSE).
2. **Visualizations**: The code will automatically save the visual results in your directory, including:
   * 📈 Endmember curve comparisons (Extracted vs. Ground Truth).
   * 🗺️ Estimated spatial fractional abundance maps.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you find this code or our research helpful, please consider citing our paper:

```text
H. Qu, J. Jia, J. Zhang, S. Liu, L. Wang, M. Le, Y. Li, B. Jiang, and X. Chen, "Wavelet-Driven Spectral-Spatial Reciprocal Network for Hyperspectral Unmixing With Spectral Variability".
