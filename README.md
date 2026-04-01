# Generative AI Core Implementations

A comprehensive collection of mini-projects covering fundamental concepts in Generative Artificial Intelligence. This repository contains deep learning implementations of key generative models, demonstrating core principles in neural networks, unsupervised learning, and probabilistic inference.

## 🎯 Repository Overview

This repository showcases practical implementations of essential generative AI techniques, from sequence-to-sequence models to autoencoders to variational inference. Each project is self-contained with complete implementations, documentation, and experimental analysis.

### What is in this Repository?

Each project folder contains:

- **Jupyter Notebook** (.ipynb): Complete implementation with explanations
- **README.md**: Comprehensive project documentation
- **Code**: Well-structured, commented Python code
- **Experiments**: Systematic evaluation and ablation studies
- **Results**: Quantitative metrics and visualizations

### Target Audience

- Students learning generative AI concepts
- Practitioners implementing foundational models
- Researchers exploring model architectures and trade-offs
- Educators teaching deep learning and AI courses

---

## 📚 Projects Overview

### ✨ Project 1: Neural Machine Translation (English → Urdu)

**Implementing sequence-to-sequence learning with vanilla RNNs**

A complete Neural Machine Translation system using vanilla RNN encoder-decoder architecture for translating English sentences to Urdu. The project demonstrates fundamental concepts in sequence modeling, attention mechanisms, and beam search decoding.

**Key Highlights:**

- Vanilla RNN encoder-decoder with attention mechanism
- 9,078 English-Urdu parallel sentence pairs
- Greedy decoding vs. Beam search (BLEU: 27.23 vs. 71.64)
- Comprehensive hyperparameter tuning and error analysis
- Training curves and convergence analysis

**Technologies:** PyTorch, torchtext, SacreBLEU

**[👉 Explore Project 1 →](./1-Neural-Machine-Translation/README.md)**

---

### 🖼️ Project 2: Image Denoising using Denoising Autoencoder

**Learning unsupervised image restoration with autoencoders**

A convolutional denoising autoencoder implementation for reconstructing clean images from corrupted noisy inputs. The project explores noise types (Gaussian and Salt-and-Pepper), compression-quality trade-offs, and bottleneck architecture effects on the CIFAR-10 dataset.

**Key Highlights:**

- Convolutional encoder-decoder architecture
- Two specialized models for different noise types
- Gaussian noise: PSNR 18-20 dB, SSIM 0.45-0.55
- Bottleneck size analysis: 32 to 512 dimensions
- Systematic experiments on noise levels and compression

**Metrics:** MSE, PSNR (dB), SSIM

**Technologies:** PyTorch, scikit-image, torchvision

**[👉 Explore Project 2 →](./2_Image_Denoising_using_Denoising_Autoencoder/README.md)**

---

### ✨ Project 3: Generative Modeling using Variational Autoencoder (VAE)

**Learning generative models through probabilistic inference**

A complete Variational Autoencoder implementation for learning latent representations and generating new synthetic images. The project demonstrates the reparameterization trick, ELBO optimization, and the impact of latent dimensionality on reconstruction vs. generation trade-offs using Fashion-MNIST.

**Key Highlights:**

- Variational Autoencoder with reparameterization trick
- ELBO loss combining reconstruction and KL divergence
- Latent dimension analysis: 2D to 50D exploration
- Latent space visualization and interpolation
- Smooth transitions between learned image categories

**Latent Dimensions Studied:** 2, 5, 10, 20, 50

**Technologies:** PyTorch, scikit-learn, matplotlib

**[👉 Explore Project 3 →](./3-Generative_Modeling_using_VAE/README.md)**

---

## 🗂️ Repository Structure

```
Generative-AI-core-implementations/
├── README.md                                           # This file
├── 1-Neural-Machine-Translation/
│   ├── README.md                                      # Project documentation
│   ├── Neural_Machine_Translation.ipynb               # Complete implementation
│   └── best_model.pt                                  # Pre-trained model (optional)
│
├── 2_Image_Denoising_using_Denoising_Autoencoder/
│   ├── README.md                                      # Project documentation
│   ├── Image_Denoising_using_Denoising_Autoencoder.ipynb  # Complete implementation
│   └── data/                                          # CIFAR-10 (auto-downloaded)
│
├── 3-Generative_Modeling_using_VAE/
│   ├── README.md                                      # Project documentation
│   ├── Generative_Modeling_using_VAE.ipynb           # Complete implementation
│   └── data/                                          # Fashion-MNIST (auto-downloaded)
│
└── Docs/                                              # (Optional) Additional documentation
    └── Concepts.md                                    # Theoretical background
```

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Python 3.8+
python --version

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation

```bash
# Clone repository
git clone https://github.com/AhmadKhan010/Generative-AI-core-implementations.git
cd Generative-AI-core-implementations

# Install dependencies
pip install torch torchvision torch-audio
pip install jupyter notebook
pip install pandas numpy matplotlib seaborn
pip install scikit-learn scikit-image
pip install sacrebleu
```

### Running Projects

Each project is a self-contained Jupyter notebook. Choose your project:

```bash
# Project 1: Neural Machine Translation
cd 1-Neural-Machine-Translation
jupyter notebook Neural_Machine_Translation.ipynb

# Project 2: Image Denoising
cd 2_Image_Denoising_using_Denoising_Autoencoder
jupyter notebook Image_Denoising_using_Denoising_Autoencoder.ipynb

# Project 3: VAE
cd 3-Generative_Modeling_using_VAE
jupyter notebook Generative_Modeling_using_VAE.ipynb
```

## 📊 Comparison Matrix

| Aspect               | Project 1: NMT          | Project 2: DAE      | Project 3: VAE           |
| -------------------- | ----------------------- | ------------------- | ------------------------ |
| **Model Type**       | Sequence-to-Sequence    | Unsupervised        | Generative               |
| **Architecture**     | RNN Encoder-Decoder     | Conv Autoencoder    | Variational Auto-Encoder |
| **Task**             | Translation             | Denoising           | Generation               |
| **Dataset**          | Custom (English-Urdu)   | CIFAR-10            | Fashion-MNIST            |
| **Input Size**       | Variable sequences      | 32×32×3 images      | 28×28×1 images           |
| **Training Samples** | 9,078 pairs             | 45,000              | 48,000                   |
| **Key Metric**       | BLEU Score              | PSNR/SSIM           | ELBO/Reconstruction      |
| **Main Challenge**   | Long-range dependencies | Detail preservation | Blurry generations       |
| **Decoding Methods** | Greedy, Beam Search     | N/A (Direct)        | Sampling, Interpolation  |

## 🎓 Learning Path

### Beginner Level

1. Start with **Project 3 (VAE)** to understand probabilistic models
2. Review the reparameterization trick and ELBO loss
3. Modify latent dimensions and observe generation changes

### Intermediate Level

1. Study **Project 2 (Denoising Autoencoder)** for unsupervised learning
2. Experiment with different bottleneck sizes
3. Analyze reconstruction metrics: MSE, PSNR, SSIM

### Advanced Level

1. Implement **Project 1 (NMT)** for sequence modeling
2. Understand attention mechanisms and beam search
3. Perform hyperparameter tuning and error analysis

## 🔑 Key Concepts Covered

### Foundational Concepts

- **Neural Networks**: Dense, convolutional, recurrent layers
- **Autoencoders**: Encoder-decoder paradigm, dimensionality reduction
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Optimization**: Adam, SGD, learning rate scheduling

### Generative Models

- **Sequence-to-Sequence Models**: Encoder-decoder for variable-length I/O
- **Unsupervised Learning**: Denoising autoencoders without labels
- **Variational Inference**: Probabilistic latent variable models
- **Reparameterization Trick**: Enabling gradient flow through sampling

### Advanced Topics

- **Attention Mechanisms**: Selective focus on input features
- **Beam Search**: Improved decoding for sequence models
- **KL Divergence**: Measuring distribution similarity
- **Trade-offs**: Compression vs. quality, reconstruction vs. generation

## 📈 Model Comparison

### Performance Overview

| Model      | Task         | Best Metric    | Generalization                     |
| ---------- | ------------ | -------------- | ---------------------------------- |
| NMT (RNN)  | English→Urdu | BLEU: 71.64    | Moderate (long sentences struggle) |
| DAE (Conv) | Denoising    | PSNR: 19.33 dB | Good (learned noise type)          |
| VAE        | Generation   | ELBO: 242.17   | Excellent (smooth latent space)    |

### Computational Requirements

| Project   | Model Size   | Training Time | Inference Speed |
| --------- | ------------ | ------------- | --------------- |
| Project 1 | 7.2M params  | ~2 hours      | Real-time       |
| Project 2 | 0.58M params | ~1 hour       | Real-time       |
| Project 3 | 0.42M params | ~30 min       | Real-time       |

## 🔍 Experimental Features

### Project 1: NMT

- Hyperparameter grid search (embedding, hidden, layers, lr, dropout, batch)
- Greedy vs. Beam search comparison
- Error analysis on 15+ test examples
- Training curve visualization

### Project 2: Denoising Autoencoder

- Gaussian noise level study: σ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
- Salt-and-Pepper noise level study: p ∈ {0.02, 0.05, 0.08, 0.10, 0.15}
- Bottleneck size ablation: {32, 64, 128, 256, 512}
- Compression vs. quality trade-off analysis

### Project 3: VAE

- Latent dimension analysis: dz ∈ {2, 5, 10, 20, 50}
- Reconstruction vs. generation trade-off
- 2D latent space visualization
- Latent space interpolation and smooth transitions

## 💡 Potential Extensions

### Short-term Projects (Intermediate)

- [ ] Implement LSTM/GRU variants for NMT
- [ ] Add U-Net architecture for denoising
- [ ] Explore β-VAE for disentangled representations
- [ ] Implement conditional VAE for class-specific generation

### Medium-term Projects (Advanced)

- [ ] Transformer-based neural machine translation
- [ ] Generative adversarial networks (GANs)
- [ ] Attention mechanisms for denoising
- [ ] Hierarchical VAE with multiple latent levels

### Long-term Research (Expert)

- [ ] Diffusion models for generative modeling
- [ ] Large-scale pre-training and transfer learning
- [ ] Multimodal generative models
- [ ] Energy-based generative models

## 📚 Educational Resources

### Theory

- **Deep Learning**: Goodfellow, Bengio, Courville (2016)
- **Attention is All You Need**: Vaswani et al. (2017)
- **Auto-Encoding Variational Bayes**: Kingma & Welling (2014)
- **Denoising Autoencoders**: Vincent et al. (2008)

### Implementations

- PyTorch Documentation: https://pytorch.org/
- TorchVision Datasets: https://pytorch.org/vision/
- HuggingFace Transformers: https://huggingface.co/transformers/

## 🤝 Contributing

Contributions are welcome! Potential areas:

### Code Improvements

- Implement alternative architectures
- Optimize training speed and memory usage
- Add more comprehensive documentation
- Improve visualization and analysis tools

### New Projects

- [ ] Object Detection
- [ ] Image Segmentation
- [ ] Reinforcement Learning
- [ ] Graph Neural Networks
- [ ] Recommendation Systems

### Documentation

- [ ] Theoretical explanations
- [ ] Mathematical derivations
- [ ] Hyperparameter tuning guides
- [ ] Troubleshooting guides

## 📋 Troubleshooting

### Common Issues

**CUDA/GPU Not Available:**

```python
import torch
print(torch.cuda.is_available())  # Should be True for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Dataset Download Issues:**

- Ensure stable internet connection
- Check disk space (entire datasets ~2GB)
- Manually download from official sources if needed

**Memory Issues:**

- Reduce batch size
- Reduce model size
- Use gradient checkpointing
- Clear unnecessary variables

**Training Divergence:**

- Reduce learning rate
- Add gradient clipping
- Check for NaN values
- Verify data preprocessing

## 📄 License

This repository is part of the **Generative AI Course** at **FAST-NUCES (National University of Computer and Emerging Sciences), Islamabad**.

## 👨‍💻 Author & Contact

**Ahmad Khan**

- **Email**: i221288@nu.edu.pk
- **Institution**: FAST-NUCES, Islamabad
- **Course**: Generative AI (Spring 2026)

## 🙏 Acknowledgments

- **Datasets**: Kaggle, CIFAR-10, Fashion-MNIST, PyTorch
- **Frameworks**: PyTorch, TorchVision, scikit-learn
- **References**: Excellent papers and tutorials from the deep learning community
- **Course Instructors**: FAST-NUCES Generative AI Course Faculty

## 📊 Repository Statistics

| Metric                     | Value                                     |
| -------------------------- | ----------------------------------------- |
| **Total Projects**         | 3                                         |
| **Total Code Lines**       | 2,000+                                    |
| **Jupyter Notebooks**      | 3                                         |
| **Documentation Pages**    | 4                                         |
| **Implemented Models**     | 5+ architectures                          |
| **Datasets Used**          | 3 (English-Urdu, CIFAR-10, Fashion-MNIST) |
| **Training Hours**         | ~3.5 hours (CPU)                          |
| **Total Model Parameters** | 8M+                                       |

## 🚀 Future Plans

- [ ] Add Project 4: Generative Adversarial Networks (GANs)
- [ ] Add Project 5: Transformers for NLP
- [ ] Add Project 6: Diffusion Models
- [ ] Add Project 7: Reinforcement Learning
- [ ] Create interactive visualization tools
- [ ] Publish research paper summarizing results
- [ ] Create video tutorials for each project

## 📞 Support

For questions or issues:

1. Check project-specific README
2. Review notebook comments and explanations
3. Search GitHub issues
4. Contact repository maintainer

---

## 🎯 Quick Navigation

| Project          | Folder                                                    | Notebook                                                                                                      | README                                                            |
| ---------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **1. NMT**       | [1-NMT](./1-Neural-Machine-Translation/)                  | [Notebook](./1-Neural-Machine-Translation/Neural_Machine_Translation.ipynb)                                   | [Docs](./1-Neural-Machine-Translation/README.md)                  |
| **2. Denoising** | [2-DAE](./2_Image_Denoising_using_Denoising_Autoencoder/) | [Notebook](./2_Image_Denoising_using_Denoising_Autoencoder/Image_Denoising_using_Denoising_Autoencoder.ipynb) | [Docs](./2_Image_Denoising_using_Denoising_Autoencoder/README.md) |
| **3. VAE**       | [3-VAE](./3-Generative_Modeling_using_VAE/)               | [Notebook](./3-Generative_Modeling_using_VAE/Generative_Modeling_using_VAE.ipynb)                             | [Docs](./3-Generative_Modeling_using_VAE/README.md)               |

---

**Last Updated:** April 2026  
**Status:** Active - New projects coming soon!  
**Version:** 1.0.0
