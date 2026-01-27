# MSSA-Net: Mamba-Driven Spatial-Frequency Synergy for Medical Image Segmentation**
[[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]][![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%2Bcu117-orange)] [[![CUDA](https://img.shields.io/badge/CUDA-11.6%2B-red)]]

***Abstract***: Precise segmentation of medical images (abdominal CT, cardiac MRI) is critical for clinical diagnosis. Convolutional neural networks (CNNs) struggle with long-range dependencies, while traditional Transformers suffer from high computational costs. To address these issues, we propose **MSSA-Net**, a novel architecture characterized by three key innovations:
**(i)** It introduces a **Spatial-Frequency Fusion Block (SMFB)** at the encoder stage, which combines Mamba's SS2D module (for efficient long-range modeling) and a Dual-Domain Enhancement Module (DDEM) (for spatial detail and frequency semantic reinforcement). This not only expands the receptive field but also preserves fine-grained boundary information.
**(ii)** A **Hierarchical Semantic Link (HSL)** module is integrated into the bottleneck layer, consisting of a Reconstruction of Global-Local Features (RUG) unit and a Cross-Source Alignment Attention (CAA) module. This harmonizes cross-scale semantic information and avoids feature drift.
**(iii)** The framework achieves linear computational complexity (O(n log n)) via Mamba's selective scanning mechanism, outperforming 14 state-of-the-art methods on the Synapse and ACDC datasets.

>** Key Reminder**: Mamba module is a mandatory dependency — code will fail to run if Mamba is not properly configured.


***1. Dependencies and Installation (Full Details)***
**1.1 Clone this repo:**
```bash
git clone https://github.com/你的仓库地址/MSSA-Net.git
cd MSSA-Net
# Step 1: Create and activate environment (Python 3.8 is mandatory)
conda create -n mssanet python=3.8 -y
conda activate mssanet

# Step 2: Install PyTorch (match CUDA version strictly)
# For CUDA 11.7 (recommended)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# For CUDA 11.6 (alternative)
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Step 3: Install core medical imaging dependencies (fixed versions to avoid conflicts)
pip install monai==1.1.0 numpy==1.23.5 scipy==1.10.1 scikit-image==0.20.0 tqdm==4.64.1
pip install opencv-python==4.7.0.72 tensorboard==2.11.2 scikit-learn==1.2.2 matplotlib==3.7.1
pip install thop==0.1.1.post2209072238 h5py==3.8.0 SimpleITK==2.2.1 medpy==0.4.0 yacs==0.1.8

# Step 4: Install Mamba core dependencies (REQUIRED, no version substitution)
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1

# If Mamba installation fails (common fixes):
# Fix 1: Build from source (bypass PyPI restrictions)
# pip install --no-build-isolation mamba_ssm
# Fix 2: Update setuptools first
# pip install --upgrade setuptools wheel
# pip install mamba_ssm==1.0.1
