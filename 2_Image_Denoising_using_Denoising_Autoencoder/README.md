# Image Denoising using Denoising Autoencoder

A comprehensive implementation of a Convolutional Denoising Autoencoder (DAE) for reconstructing clean images from corrupted noisy inputs. This project demonstrates fundamental concepts in unsupervised learning, autoencoders, and image restoration using deep neural networks.

## 📋 Project Overview

This mini-project implements a complete image denoising pipeline using convolutional autoencoders. The system is trained to remove two types of noise (Gaussian and Salt-and-Pepper) from images using the CIFAR-10 dataset. The project explores the trade-offs between compression, reconstruction quality, and model complexity through systematic experimental analysis.

### Key Objectives

- Design and implement a convolutional denoising autoencoder architecture
- Implement two types of noise injection (Gaussian and Salt-and-Pepper)
- Train specialized models for different noise types
- Evaluate reconstruction quality using multiple metrics (MSE, PSNR, SSIM)
- Conduct comprehensive experimental studies on noise levels and bottleneck sizes
- Analyze compression-quality trade-offs in autoencoder architectures

## 🏗️ Architecture

### Encoder-Decoder Model

The denoising autoencoder comprises symmetric encoder and decoder networks:

```
Noisy Image (32x32x3)
      ↓
[Conv 3→32, ReLU, MaxPool]     (32x32x32 → 16x16x32)
      ↓
[Conv 32→64, ReLU, MaxPool]    (16x16x64 → 8x8x64)
      ↓
[Conv 64→128, ReLU, MaxPool]   (8x8x128 → 4x4x128)
      ↓
[Bottleneck Layer]              (4x4x128 → Compressed Rep)
      ↓
[TransposeConv 128→64, ReLU]   (4x4x64 → 8x8x64)
      ↓
[TransposeConv 64→32, ReLU]    (8x8x32 → 16x16x32)
      ↓
[TransposeConv 32→3, Sigmoid]  (16x16x3 → 32x32x3)
      ↓
Denoised Image (32x32x3)
```

**Encoder Specification:**

- Layer 1: Conv(3→32, kernel=3) + ReLU + MaxPool(2×2)
- Layer 2: Conv(32→64, kernel=3) + ReLU + MaxPool(2×2)
- Layer 3: Conv(64→128, kernel=3) + ReLU + MaxPool(2×2)
- Output: 128×4×4 feature map

**Bottleneck:**

- Fully connected compression layer
- Configurable size: 32, 64, 128, 256, 512 dimensions
- Represents the learned latent space

**Decoder Specification:**

- Layer 1: TransposeConv(128→64, stride=2) + ReLU
- Layer 2: TransposeConv(64→32, stride=2) + ReLU
- Layer 3: TransposeConv(32→3, stride=2) + Sigmoid
- Output: 3×32×32 image in range [0, 1]

## 📊 Dataset

**CIFAR-10 Dataset**

- Total images: 60,000 (50,000 training + 10,000 test)
- Image resolution: 32×32 pixels
- Channels: 3 (RGB color images)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Data Split:**

- Training: 45,000 images (90% of original training set)
- Validation: 5,000 images (10% of original training set)
- Test: 10,000 images
- Normalization: Pixel values scaled to [0, 1]

**Noise Types Implemented:**

### Gaussian Noise

```
I_noisy = I_clean + N(0, σ²)
```

- Added from normal distribution with mean=0
- Standard deviation (σ): {0.1, 0.2, 0.3, 0.4, 0.5}
- Simulates sensor/channel noise

### Salt-and-Pepper Noise

```
I_noisy(x,y) = {
    0 with probability p_pepper
    1 with probability p_salt
    I_clean(x,y) otherwise
}
```

- Randomly sets pixels to black (0) or white (1)
- Combined probability (p): {0.02, 0.05, 0.08, 0.10, 0.15}
- Simulates impulse noise (transmission errors)

## 🎯 Results and Evaluation

### Performance Metrics

Three complementary metrics evaluate reconstruction quality:

**1. Mean Squared Error (MSE)**

```
MSE = (1/N) Σ ||I_reconstructed - I_clean||²
```

- Pixel-wise reconstruction error
- Lower values indicate better quality

**2. Peak Signal-to-Noise Ratio (PSNR)**

```
PSNR = 10 log₁₀(MAX²/MSE) = 20 log₁₀(MAX/√MSE)
```

- Measured in decibels (dB)
- MAX = 1 for normalized images
- Higher values indicate better quality
- Typical range: 18-30 dB for noisy→clean reconstruction

**3. Structural Similarity Index (SSIM)**

```
SSIM(x,y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / (μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)
```

- Range: 0 to 1 (1 = identical)
- Captures perceptual quality better than MSE
- Accounts for luminance, contrast, and structure

### Experimental Results

#### Experiment 1: Gaussian Noise Levels

| Noise Level (σ) | MSE    | PSNR (dB) | SSIM   |
| --------------- | ------ | --------- | ------ |
| 0.1             | 0.0119 | 19.72     | 0.5470 |
| 0.2             | 0.0121 | 19.64     | 0.5358 |
| 0.3             | 0.0132 | 19.20     | 0.5018 |
| 0.4             | 0.0142 | 18.85     | 0.4797 |
| 0.5             | 0.0160 | 18.29     | 0.4549 |

**Key Findings:**

- Progressive degradation as noise increases
- PSNR remains stable around 18-20 dB
- SSIM shows moderate structural preservation (0.45-0.55)
- Model trained on σ=0.3 generalizes to nearby noise levels

#### Experiment 2: Salt-and-Pepper Noise Levels

| Noise Level (p) | MSE    | PSNR (dB) | SSIM   |
| --------------- | ------ | --------- | ------ |
| 0.02            | 0.0124 | 19.53     | 0.5342 |
| 0.05            | 0.0120 | 19.68     | 0.5400 |
| 0.08            | 0.0129 | 19.34     | 0.5145 |
| 0.10            | 0.0127 | 19.40     | 0.5142 |
| 0.15            | 0.0139 | 18.99     | 0.4864 |

**Key Findings:**

- Best performance at p=0.05 (training noise level)
- Consistent PSNR around 19-20 dB
- Model shows robustness to corruption levels
- Performance more stable than Gaussian noise

#### Experiment 3: Bottleneck Size Impact

| Bottleneck | MSE    | PSNR (dB) | SSIM   | Parameters |
| ---------- | ------ | --------- | ------ | ---------- |
| 32         | 0.0146 | 18.78     | 0.4643 | 156,547    |
| 64         | 0.0137 | 19.05     | 0.4952 | 297,987    |
| 128        | 0.0133 | 19.18     | 0.4993 | 580,867    |
| 256        | 0.0134 | 19.15     | 0.5085 | 1,146,627  |
| 512        | 0.0128 | 19.33     | 0.5148 | 2,278,147  |

**Key Findings:**

- Bottleneck=512: Best quality (PSNR: 19.33 dB, SSIM: 0.5148)
- Diminishing returns: 128→512 only improves PSNR by 0.15 dB
- Parameter explosion: Parameters quadruple from 128 to 512
- **Optimal choice**: Bottleneck=128 for best compression-quality balance
- Compression ratio: 1.5:1 (bottleneck=512) to 20:1 (bottleneck=32)

## 🔧 Training Configuration

| Parameter              | Value                    | Notes                             |
| ---------------------- | ------------------------ | --------------------------------- |
| Loss Function          | Mean Squared Error (MSE) | Pixel-wise reconstruction         |
| Optimizer              | Adam                     | Adaptive learning rate            |
| Learning Rate          | 0.001                    | Good balance of convergence speed |
| Batch Size             | 64                       | Memory/stability trade-off        |
| Epochs                 | 30                       | Early stopping if needed          |
| Learning Rate Schedule | ReduceLROnPlateau        | Factor=0.5, patience=3            |
| Separate Models        | Gaussian & S-P           | Specialized denoising             |

**Training Results:**

- Gaussian model: Final validation MSE ≈ 0.0082
- Salt-Pepper model: Final validation MSE ≈ 0.0095
- Convergence: Both models converged within 30 epochs
- Generalization: Good performance on held-out test set

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AhmadKhan010/Generative-AI-core-implementations.git
cd 2_Image_Denoising_using_Denoising_Autoencoder

# Install dependencies
pip install torch torchvision torchvision transforms scikit-learn scikit-image matplotlib numpy
```

### Usage

```python
import torch
from Image_Denoising_using_Denoising_Autoencoder import DenoisingAutoencoder, add_gaussian_noise

# Load pre-trained model
model = DenoisingAutoencoder(bottleneck_size=128)
# Load checkpoint if available

# Add noise to an image
noisy_image = add_gaussian_noise(clean_image, noise_factor=0.3)

# Denoise
denoised_image = model(noisy_image)

# Visualize
plt.imshow(denoised_image.permute(1, 2, 0).detach().cpu().numpy())
plt.show()
```

### Training from Scratch

```python
# See Image_Denoising_using_Denoising_Autoencoder.ipynb for complete training pipeline
# The notebook includes:
# - CIFAR-10 dataset loading
# - Noise injection implementations
# - Model architecture definition
# - Training loop with validation
# - Comprehensive evaluation and visualization
# - Experimental studies
```

## 📁 Project Structure

```
2_Image_Denoising_using_Denoising_Autoencoder/
├── README.md                                           # This file
├── Image_Denoising_using_Denoising_Autoencoder.ipynb  # Complete implementation
├── data/                                               # CIFAR-10 dataset (auto-downloaded)
└── models/                                             # (Optional) Model checkpoints
    ├── gaussian_denoiser.pth                          # Gaussian noise model
    └── salt_pepper_denoiser.pth                       # S-P noise model
```

## 🔍 Key Insights and Limitations

### Strengths

1. **Noise-Type Specialization**: Separate models for different noise types yield better performance
2. **Effective Compression**: Achieves 20:1 compression with acceptable reconstruction quality
3. **Efficient Architecture**: Convolutional layers capture spatial hierarchies effectively
4. **Robust Performance**: Moderate generalization across nearby noise levels
5. **Clear Trade-offs**: Demonstrates compression vs. quality spectrum

### Limitations

1. **Detail Smoothing**: Fine textures can be over-smoothed, especially at higher noise levels
2. **Limited Generalization**: Models don't transfer well to unseen noise types
3. **Perceptual vs. MSE**: MSE loss doesn't always align with human perception
4. **Blurriness**: Reconstruction inherently smoother than original clean images
5. **Fixed Architecture**: Requires retraining for different image sizes

## 🔮 Future Improvements

### Short-term Enhancements

1. **Skip Connections (U-Net)**: Better detail preservation during reconstruction
2. **Residual Learning**: Easier gradient flow for deeper networks
3. **Attention Mechanisms**: Focus on important image regions
4. **Mixed Noise Training**: Single model for multiple noise types

### Medium-term Improvements

1. **Generative Adversarial Networks (GANs)**: Sharper, more realistic reconstructions
2. **Perceptual Loss**: VGG-based features for better visual quality
3. **Multi-scale Denoising**: Handle different frequency components separately
4. **Domain-Specific Models**: Specialized for medical/satellite/surveillance imagery

### Long-term Research Directions

1. **Diffusion Models**: State-of-the-art generative denoising
2. **Unsupervised Learning**: Noise2Noise, Blind Spot methods
3. **Real-World Datasets**: Medical imaging, night photography, satellite imagery
4. **Adaptive Denoising**: Content-aware noise level estimation and removal

## 📊 Experimental Design

### Methodology

1. **Baseline Training**: Train on medium noise levels (σ=0.3, p=0.05)
2. **Systematic Evaluation**: Test across full range of noise levels
3. **Architecture Ablation**: Vary bottleneck size keeping other parameters constant
4. **Metric Triangulation**: Use MSE, PSNR, and SSIM for comprehensive evaluation
5. **Generalization Testing**: Evaluate on held-out test set with unseen data

### Design Rationale

- **Two Specialized Models**: Noise types have different characteristics
- **Fixed Architecture**: Isolates the effect of bottleneck size
- **Multiple Metrics**: No single metric fully captures reconstruction quality
- **Test Set Validation**: Ensures results generalize to new images

## 📚 Related Work

- **Autoencoders**: Hinton & Zemel (2013) - Learning to Discover Representations
- **Denoising Autoencoders**: Vincent et al. (2008) - Extracting and Composing Robust Features
- **Convolutional Networks**: LeCun et al. (1998) - Gradient-based learning for image analysis
- **U-Net**: Ronneberger et al. (2015) - Convolutional Networks for Biomedical Image Segmentation
- **PSNR & SSIM**: Wang et al. (2004) - Image Quality Assessment metrics

## 📝 Evaluation Metrics Deep Dive

### Why Multiple Metrics?

- **MSE**: Mathematically simple but doesn't capture perceptual quality
- **PSNR**: Commonly used benchmark but has limitations with compression artifacts
- **SSIM**: Better correlation with human perception, captures structural similarity
- **Combined**: Three metrics provide complementary views of reconstruction quality

### Interpreting Results

- PSNR > 30 dB: Excellent quality
- PSNR 20-30 dB: Good quality
- PSNR 15-20 dB: Acceptable quality (our range)
- PSNR < 15 dB: Poor quality

## 🎓 Learning Outcomes

This project demonstrates:

1. **Autoencoder Design**: Symmetric encoder-decoder architectures
2. **Convolutional Networks**: Feature learning from spatial data
3. **Unsupervised Learning**: Training without paired labeled data
4. **Noise Injection**: Simulating real-world corruption
5. **Experimental Methodology**: Systematic parameter search and evaluation
6. **Trade-off Analysis**: Compression vs. quality considerations

## 🤝 Contributing

Potential contributions:

- Implement alternative architectures (U-Net, ResNet-DAE)
- Add new noise types (Poisson, uniform, motion blur)
- Extend to other datasets (ImageNet, medical imaging)
- Optimize inference speed
- Create interactive visualization tools

## 📄 License

Copyright © 2026 Ahmad Khan. All rights reserved.

## 👨‍💻 Author

**Ahmad Khan**

- Email: ahmadkhanmarwat8@gmail.com
- Institution: FAST National University of Computer and Emerging Sciences, Islamabad

## 🙏 Acknowledgments

- **Dataset**: CIFAR-10 from Alex Krizhevsky
- **Framework**: PyTorch
- **Metrics**: scikit-image (PSNR, SSIM)
- **Course**: Generative AI, FAST-NUCES

## 📚 References

[1] Hinton, G. E., & Zemel, R. S. (2013). Autoencoders, Minimum Description Length and Helmholtz Free Energy. _NIPS 1993_.

[2] Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). Extracting and Composing Robust Features with Denoising Autoencoders. _ICML 2008_.

[3] LeCun, Y., Bottou, L., Bengio, Y., & LeCun, Y. (1998). Gradient-based Learning Applied to Document Recognition. _Proceedings of the IEEE_.

[4] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. _MICCAI 2015_.

[5] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. _IEEE TIP_, 13(4), 600-612.

---

**Last Updated:** April 2026  
**Status:** Complete - Fully Functional Image Denoising System
