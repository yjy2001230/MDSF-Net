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
