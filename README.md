# Dual Discriminator GAN with self attention

This project was implemented as part of the CS787(Generative AI) course at IIT Kanpur during my First Semester of M.Tech, 2025.

This project evaluates four generative models on the CIFAR-10 dataset:

1. **DCGAN**  
2. **DCGAN + Self-Attention (SAGAN-style)**
3. **Dual-Discriminator GAN (D2GAN)**
4. **Dual-Discriminator GAN + Self-Attention**

The final goal was to integrate self attention mechanism in a dual discriminator GAN, improving diversity as well as quality.

---

## Models

### 1. DCGAN  
Baseline convolutional GAN using transposed convolutions for upsampling and LeakyReLU in the discriminator.

### 2. DCGAN with Self-Attention  
Adds a self-attention block to capture long-range spatial dependencies and improve global consistency.

### 3. D2GAN  
Uses two discriminators:
- **D1** encourages diversity (forward KL)
- **D2** encourages realism (reverse KL)

This helps reduce mode collapse and encourages broader coverage of the data distribution.

### 4. D2GAN with Self-Attention  
Combines both ideas:
- dual-discriminator loss  
- attention-based feature modelling  
This is the most expressive model tested.

---

## Training
All models are trained on CIFAR-10 (32×32 RGB).  
Training scripts include:
- checkpointing
- deterministic mode (optional)
- fixed-noise visualisation
- periodic sample generation

---

## Evaluation
We compute:
- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**


Results:
## Final Model Performance on CIFAR-10 (FID ↓, IS ↑)

| Model            | FID ↓ | IS ↑ |
|------------------|-------|------|
| DCGAN (baseline) | 25.66 | 6.17 |
| D2GAN            | 21.40 | 6.43 |
| DCGAN-SA         | 23.62 | 6.34 |
| D2GAN-SA         | 27.55 | 6.09 |


D2GAN achieves the best overall generative performance on CIFAR-10, obtaining the lowest FID (21.40). The dual-discriminator setup effectively improves mode coverage and sample realism compared to single-discriminator GAN.

Integrating self-attention into DCGAN provided modest improvements (FID: 23.62); however, applying the same modification to D2GAN led to a decline in performance (FID: 27.55). This behavior can be attributed to two main factors: (i) attention mechanisms offer limited benefit at the low spatial resolution of CIFAR-10, and (ii) the KL-driven training dynamics of D2GAN may conflict with attention-based feature modeling, resulting in unstable optimization.

Additionally, our study underscores the critical role of spectral normalization, discriminator update frequency, and the strategic placement of refining layers in stabilizing training and enhancing generative quality.

Overall, the results suggest that self-attention is not particularly effective for low-resolution datasets like CIFAR-10, where long-range dependencies are limited. While attention improves feature coherence slightly in DCGAN variants, it does not translate into meaningful gains for D2GAN. In contrast, the standard D2GAN—without attention—remains the strongest model, offering the best balance between diversity and fidelity.








---
