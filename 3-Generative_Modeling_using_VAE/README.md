# Generative Modeling using Variational Autoencoder (VAE)

A comprehensive implementation of a Variational Autoencoder (VAE) for learning latent representations and generating synthetic images. This project demonstrates fundamental concepts in generative modeling, probabilistic inference, and deep learning using the Fashion-MNIST dataset.

## 📋 Project Overview

This mini-project implements a complete VAE pipeline for generative modeling. The system learns compressed latent representations of fashion item images and can generate new, realistic synthetic samples by sampling from the learned latent distribution. The project explores the impact of latent dimensionality on reconstruction quality, generation capability, and latent space organization through systematic experimental analysis.

### Key Objectives

- Design and implement a Variational Autoencoder with reparameterization trick
- Implement the VAE loss function (reconstruction + KL divergence)
- Train and evaluate on Fashion-MNIST dataset
- Conduct systematic experiments on latent dimensions
- Analyze reconstruction vs. generation trade-offs
- Visualize learned latent space representations
- Perform latent space interpolation for smooth image transitions

## 🏗️ Architecture

### Variational Autoencoder (VAE)

The VAE comprises an encoder, reparameterization layer, and decoder:

```
Fashion-MNIST Image (1x28x28 = 784 dims)
      ↓
[Embedding Layer]
      ↓
[Dense 784 → 400, ReLU]     (Encoder hidden representation)
      ↓
[Split into μ and log(σ²)]   (Latent parameters)
      ↓
[Reparameterization Trick]   z = μ + σ ⊙ ε
      ↓
[Dense dz → 400, ReLU]       (Decoder hidden representation)
      ↓
[Dense 400 → 784, Sigmoid]   (Reconstructed image)
      ↓
Fashion-MNIST Image (1x28x28 = 784 dims)
```

**Encoder Specification:**

- Input: Flattened image (784 dimensions)
- Hidden layer: 400 units with ReLU activation
- Output layer:
  - μ (mean): dz dimensions
  - log(σ²) (log-variance): dz dimensions
- Parameterization: Latent dimension (dz) ∈ {2, 5, 10, 20, 50}

**Reparameterization Trick:**

```
z = μ + σ ⊙ ε,  where ε ~ N(0, I)
σ = exp(0.5 * log(σ²))
```

- Enables differentiable sampling
- Allows backpropagation through stochastic sampling
- Critical for end-to-end training

**Decoder Specification:**

- Input: Latent vector (dz dimensions)
- Hidden layer: 400 units with ReLU activation
- Output layer: 784 units with Sigmoid activation
- Output range: [0, 1] (pixel values)

## 📊 Dataset

**Fashion-MNIST Dataset**

- Total images: 70,000 (60,000 training + 10,000 test)
- Image resolution: 28×28 pixels
- Channels: 1 (grayscale)
- Classes: 10 (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)

**Data Split:**

- Training: 48,000 images (80% of 60,000)
- Validation: 12,000 images (20% of 60,000)
- Test: 10,000 images
- Normalization: Pixel values scaled to [0, 1]
- Flattening: 28×28 → 784-dimensional vectors

## 🎯 VAE Loss Function

### Variational Lower Bound (ELBO)

```
L_VAE = L_recon + β * L_KL
```

Where:

**Reconstruction Loss (Binary Cross-Entropy):**

```
L_recon = -Σ[x_i * log(x̂_i) + (1-x_i) * log(1-x̂_i)]
```

- Pixel-wise reconstruction accuracy
- Encourages accurate image reconstruction

**KL Divergence Loss:**

```
L_KL = -0.5 * Σ[1 + log(σ_j²) - μ_j² - σ_j²]
```

- Regularizes latent distribution
- Encourages match with standard normal prior N(0,I)
- Prevents posterior collapse

**Hyperparameter:**

- β = 1.0 (standard VAE balance)
- Equal weight to reconstruction and regularization

### Interpretation

- **High Reconstruction Loss**: Poor image reconstruction
- **High KL Divergence**: Latent space far from prior, underutilized dimensions
- **Balance**: Trade-off between reconstruction quality and generation capability

## 📈 Results and Evaluation

### Training Dynamics Across Latent Dimensions

| Latent Dim (dz) | Recon Loss | KL Div | Total Loss |
| --------------- | ---------- | ------ | ---------- |
| 2               | 256.80     | 6.41   | 263.21     |
| 5               | 233.21     | 11.55  | 244.77     |
| 10              | 226.74     | 15.42  | 242.17     |
| 20              | 225.99     | 16.42  | 242.40     |
| 50              | 226.45     | 15.87  | 242.32     |

**Key Findings:**

1. **Reconstruction Loss:**
   - Significant decrease from dz=2 (256.80) to dz=5 (233.21)
   - Plateaus around 226 for higher dimensions
   - Additional dimensions provide marginal gains (~1% improvement)

2. **KL Divergence:**
   - Increases from 6.41 (dz=2) to peak of 16.42 (dz=20)
   - Indicates better utilization of latent space with more dimensions

3. **Optimal Trade-off:**
   - **dz=10**: Minimizes total loss (242.17)
   - **dz=20**: Better reconstruction (225.99) but higher KL divergence
   - **Recommendation**: Use dz=10 for efficiency, dz=20 for quality

### Generation Quality Analysis

| Latent Dim | Reconstruction | Generation          | Interpretability             |
| ---------- | -------------- | ------------------- | ---------------------------- |
| dz=2       | Blurry         | Limited diversity   | Excellent (2D visualization) |
| dz=5, 10   | Improved       | Good diversity      | Good (t-SNE visualization)   |
| dz=20, 50  | Sharp          | Excellent diversity | Difficult (high-dimensional) |

**Quality Progression:**

- **dz=2**: Limited capacity, blurry reconstructions, but fully interpretable
- **dz=5, 10**: Good balance of quality and interpretability
- **dz=20**: Best overall quality with sharp edges and fine details
- **dz=50**: Marginal improvement, risk of overfitting and posterior collapse

## 🔧 Training Configuration

| Parameter        | Value             | Notes                             |
| ---------------- | ----------------- | --------------------------------- |
| Optimizer        | Adam              | Adaptive learning rates           |
| Learning Rate    | 0.001             | Good convergence speed            |
| Batch Size       | 128               | Balances memory and stability     |
| Epochs           | 20                | Effective training epochs: ~15    |
| Hidden Dimension | 400               | Sufficient for feature extraction |
| Weight Decay     | 1e-5              | L2 regularization                 |
| Loss Function    | ELBO (Recon + KL) | Balanced VAE training             |

**Training Results:**

- All five models converged successfully within 20 epochs
- Effective training: ~15 epochs before plateau
- No severe overfitting observed on validation set
- Clean loss trajectories for all latent dimensions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AhmadKhan010/Generative-AI-core-implementations.git
cd 3-Generative_Modeling_using_VAE

# Install dependencies
pip install torch torchvision torchvision transforms matplotlib numpy sklearn
```

### Usage

```python
import torch
from Generative_Modeling_using_VAE import VAE

# Create model
model = VAE(latent_dim=20)
model.load_state_dict(torch.load('vae_model.pth'))
model.eval()

# Generate new images
with torch.no_grad():
    z = torch.randn(16, 20)  # 16 random samples
    generated_images = model.decode(z)

# Reconstruct images
reconstructed = model(images)

# Latent space interpolation
z1 = model.encode(img1)
z2 = model.encode(img2)
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = (1-t) * z1 + t * z2
    img_interp = model.decode(z_interp)

# Visualize results
plt.imshow(generated_images[0].squeeze().detach().numpy(), cmap='gray')
plt.show()
```

### Training from Scratch

```python
# See Generative_Modeling_using_VAE.ipynb for complete training pipeline
# The notebook includes:
# - Fashion-MNIST dataset loading and preprocessing
# - VAE architecture definition
# - Reparameterization trick implementation
# - Loss function computation
# - Training loop with validation
# - Inference and evaluation
# - Latent space visualization
# - Comprehensive experimental studies
```

## 📁 Project Structure

```
3-Generative_Modeling_using_VAE/
├── README.md                                    # This file
├── Generative_Modeling_using_VAE.ipynb         # Complete implementation
├── data/                                        # Fashion-MNIST dataset (auto-downloaded)
└── models/                                      # (Optional) Model checkpoints
    ├── vae_latent_2.pth                        # VAE with latent_dim=2
    ├── vae_latent_10.pth                       # VAE with latent_dim=10
    └── vae_latent_20.pth                       # VAE with latent_dim=20 (best)
```

## 🔍 Key Insights and Demonstrations

### 1. Latent Space Visualization (2D VAE)

For the 2D VAE model, the learned latent space reveals:

- **Clear clustering** by fashion categories
- **Continuous transitions** between similar items
- **Semantic organization**: Adjacent regions produce similar images
- **Example separations**:
  - Trousers and bags clearly separated
  - Shoes (sandals, sneakers, ankle boots) clustered together
  - Clothing items (t-shirt, shirt, coat, pullover) form a region

### 2. Image Generation

Sampling z ~ N(0,I) from the learned latent space produces:

- Realistic fashion item images (especially with dz≥20)
- Diverse clothing variations in each category
- Smooth transitions in clothing style and appearance
- High-quality reconstructions maintaining key features

### 3. Latent Space Interpolation

Linear interpolation between two encoded images z₁ and z₂:

```
z_t = (1-t) * z₁ + t * z₂, t ∈ [0, 1]
```

**Results:**

- Smooth visual transitions between clothing items
- Maintains semantic consistency during interpolation
- Demonstrates continuous, structured latent space
- Validates learned representations

### 4. Reconstruction Analysis

**Good Reconstructions (dz≥20):**

- Preserve clothing type and basic shape
- Maintain color/intensity patterns
- Sharp edges and clear details
- Minimal loss of information

**Blurry Reconstructions (dz=2, 5):**

- Generic clothing shapes
- Averaged intensities (due to MSE loss)
- Loss of fine details
- Still recognizable (fundamental features preserved)

## 🧠 Understanding VAEs

### Key Properties

1. **Probabilistic Generative Model:**
   - Models p(x|z) where z is latent
   - Samples from z produce diverse, realistic images
   - Can compute likelihood (approximately)

2. **Learned Latent Distribution:**
   - Not arbitrary but structured by training
   - Categories occupy meaningful regions
   - Similar items cluster together

3. **Reparameterization Trick:**
   - Enables gradient flow through sampling
   - z = μ + σ ⊙ ε is differentiable
   - Critical for end-to-end learning

4. **ELBO Objective:**
   - Balances reconstruction (fit to data) and regularization (prior matching)
   - Prevents degenerate solutions
   - Enables stable training

### Reconstruction vs. Generation Trade-off

- **Reconstruction:** High fidelity to input distribution
- **Generation:** High fidelity to prior distribution
- **KL divergence:** Quantifies this trade-off
- **Lower β:** Better reconstruction, worse generation
- **Higher β:** Better generation, worse reconstruction

## 🔮 Future Improvements

### Short-term Enhancements

1. **β-VAE**: Adjust β to explore reconstruction-generation trade-off
2. **Convolutional Architecture**: Better spatial feature extraction
3. **Hierarchical VAE**: Multiple latent variable levels
4. **Conditional VAE**: Generate specific clothing categories

### Medium-term Improvements

1. **Variational Dropout**: Improved regularization
2. **VampPrior**: Learnable mixture prior instead of Gaussian
3. **Adversarial Training**: GAN + VAE hybrid (VAE-GAN)
4. **Disentangled Representations**: β-VAE and factor-VAE

### Long-term Research Directions

1. **Diffusion Models**: Alternative generative approach
2. **Energy-Based Models**: Complementary perspective
3. **Flow-Based Generative Models**: More flexible posteriors
4. **Large-Scale Pre-training**: On diverse image datasets

## 📊 Experimental Design

### Latent Dimension Study

**Rationale:**

- Latent dimension dz critically affects model behavior
- Too small: Information bottleneck, poor generation
- Too large: Overfitting, inefficient representation
- Find sweet spot balancing quality and efficiency

**Methodology:**

1. Train separate models for dz ∈ {2, 5, 10, 20, 50}
2. Keep all other hyperparameters fixed
3. Evaluate on same test set
4. Compare reconstruction, generation, and latent organization

**Results Application:**

- dz=2: Educational visualization, clear structure
- dz=10: Practical efficiency, good quality
- dz=20: Best generation quality
- Guide practitioners in dimensionality selection

## 📚 Related Work

- **Autoencoders**: Rumelhart et al. (1986) - Learning representations by backpropagating errors
- **Variational Autoencoders**: Kingma & Welling (2014) - Auto-Encoding Variational Bayes
- **ELBO**: Jordan et al. (1999) - Introduction to Variational Methods for Graphical Models
- **Reparameterization Trick**: Rezende et al. (2014) - Stochastic Backpropagation
- **β-VAE**: Higgins et al. (2017) - Learning Basic Visual Concepts with a Constrained VAE
- **VampPrior**: Tomczak & Welling (2018) - VAE with a VampPrior

## 📝 Mathematical Details

### Variational Inference

**Goal:** Maximize p(x), intractable due to p(z|x)

**Solution:** Maximize ELBO (Evidence Lower Bound)

```
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = Reconstruction - KL Divergence
```

### Reparameterization Trick

**Problem:** z ~ q(z|x) is not differentiable
**Solution:** z = μ + σ ⊙ ε where ε ~ N(0,I)
**Result:** Gradient flows through deterministic transformation

### Posterior and Prior

- **Posterior q(z|x):** Encoder outputs μ and log(σ²)
- **Prior p(z):** Standard normal N(0,I)
- **KL divergence:** Closed-form for Gaussians

## 🎓 Learning Outcomes

This project demonstrates:

1. **Generative Models:** How to learn data distributions
2. **Probabilistic Inference:** Variational lower bounds, KL divergence
3. **Latent Variable Models:** Hidden variable representation
4. **Reparameterization Trick:** Critical technique in modern deep learning
5. **Loss Function Design:** Balancing competing objectives
6. **Experimental Methodology:** Parameter search and evaluation
7. **Visualization:** Understanding high-dimensional learned representations

## 🤝 Contributing

Potential contributions:

- Implement alternative architectures (Convolutional VAE, Hierarchical VAE)
- Extend to color images (natural images, medical imaging)
- Add conditional generation (category-specific generation)
- Implement β-VAE and other VAE variants
- Create interactive generation and interpolation tools

## 📄 License

This project is part of the Generative AI course at FAST-NUCES.

## 👨‍💻 Author

**Ahmad Khan**

- Email: i221288@nu.edu.pk
- Institution: FAST National University of Computer and Emerging Sciences, Islamabad

## 🙏 Acknowledgments

- **Dataset**: Fashion-MNIST by Zalando Research
- **Framework**: PyTorch
- **Course**: Generative AI, FAST-NUCES
- **References**: Kingma & Welling (2014), Rezende et al. (2014)

## 📚 References

[1] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. _ICLR 2014_.

[2] Rezende, D. J., Mohamed, S., & Welling, M. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. _ICML 2014_.

[3] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. _ICLR 2017_.

[4] Tomczak, J., & Welling, M. (2018). VAE with a VampPrior. _AISTATS 2018_.

[5] Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An Introduction to Variational Methods for Graphical Models. _Machine Learning_, 37(2), 183-233.

---

**Last Updated:** April 2026  
**Status:** Complete - Fully Functional Generative VAE System
