# DeepLense-GSoC-Task1
Common Test I of the DeepLense project for GSoC 2026
# DeepLense GSoC 2026 — Common Test I & Specific Test VII

Evaluation test submissions for the **Physics Guided Machine Learning on Real Lensing Images** and **Physics-Informed Diffusion Models for Gravitational Lensing Simulation** projects under [ML4SCI / DeepLense](https://github.com/ML4SCI/DeepLense).

**Author:** Xinming (Tina) Shen · Johns Hopkins University · xshen43@jh.edu

---

## Task Descriptions

**Common Test I — Multi-Class Classification.**
Build a model to classify strong gravitational lensing images into three dark matter substructure classes (no substructure, subhalo, vortex) using PyTorch. Evaluation: ROC curves and AUC scores.

**Specific Test VII — Physics-Guided ML.**
Build a physics-informed neural network (PINN) that uses the gravitational lensing equation to improve classification over the Common Test baseline. Evaluation: ROC curves and AUC scores.

**Dataset:** ~30k training / ~7.5k validation images per class. Single-channel 150×150 simulated lensing images (`.npy`), min-max normalized.

---

## Results Summary

| Model | Val Accuracy | Macro AUC | no\_sub AUC | subhalo AUC | vortex AUC |
|-------|-------------|-----------|------------|------------|------------|
| **Test I** — EfficientNet-B0 | 96.2% | **0.9950** | 0.9951 | 0.9920 | 0.9980 |
| **Test VII V1** — Poisson (smoothing) | 96.5% | **0.9953** | 0.9955 | 0.9923 | 0.9981 |
| **Test VII V2** — Poisson (spectral) | 96.4% | 0.9949 | 0.9953 | 0.9914 | 0.9980 |

---

## Test I — EfficientNet-B0 Classifier

**Strategy:** Transfer learning from an ImageNet-pretrained EfficientNet-B0 with a replaced classification head (1280 → 3 classes). Training has two phases:

1. **Warmup (5 epochs):** backbone frozen, head-only training at lr=1e-3
2. **Fine-tuning (25 epochs):** full backbone unfrozen at lr=3e-4 with cosine annealing

Regularization: dropout (0.3), AdamW with weight decay (1e-4).

### Training curves

Loss drops sharply once the backbone is unfrozen at epoch 6. Train and val accuracy converge to ~96%.

![Training curves — Test I](figures/training_curves_test1.png)

### ROC curves

All three classes achieve AUC > 0.99. Subhalo is the hardest class to separate (0.9920), while vortex is easiest (0.9980).

![ROC curves — Test I](figures/roc_test1.png)

---

## Test VII — Physics-Guided Classifier

### Physics background

The lensing convergence κ(x,y) is the dimensionless projected surface mass density. It determines light deflection through the lensing potential ψ via the Poisson equation:

$$\nabla^2 \psi = 2\kappa$$

The three dark matter classes have different κ distributions: `no_sub` has a smooth profile, `subhalo` has localized mass clumps, and `vortex` has ring-like patterns. The idea is to make the network predict κ as an intermediate variable and enforce the Poisson equation as a physics constraint.

### V1: Smoothing-based Poisson constraint

The Test I backbone is extended with:
- A small conv head that predicts a 7×7 convergence map κ ≥ 0 (via Softplus) from the backbone spatial features
- A classification head that receives pooled backbone features (1280-d) + mean(κ) (1-d)
- The potential ψ is approximated by smoothing κ with a 3×3 uniform kernel
- Loss: cross-entropy + λ·‖∇²ψ − 2κ‖² − 0.05·Var(κ)
- λ ramps from 0 → 0.5 over the warmup phase
- Backbone is initialized from trained Test I weights

#### V1 Training curves

The physics loss (green dashed) drops to near-zero almost immediately, indicating the Poisson constraint is trivially satisfied.

![Training curves — V1](figures/training_curves_test7_v1.png)

#### ROC comparison: Test I vs V1

The improvement is marginal (0.9950 → 0.9953), consistent with the physics branch providing little additional signal.

![ROC comparison](figures/roc_comparison_v1.png)

#### V1 Convergence map samples

Row 1: input image. Row 2: predicted κ (7×7). Row 3: κ overlaid on image. The maps are near-zero with no visible class-discriminative structure.

![Convergence maps — V1](figures/kappa_overlay_v1.png)

#### V1 Per-class κ analysis

Mean and standard deviation of κ maps aggregated across the validation set by class. All three classes produce nearly identical near-zero maps, confirming the network finds a trivial solution (κ → 0) that satisfies Poisson without learning meaningful mass structure.

![Per-class kappa — V1](figures/kappa_per_class_v1.png)

### V2: Spectral Poisson solver (28×28)

V2 addresses V1's trivial-solution problem with three changes:

1. **Spectral Poisson solver:** ψ is computed from κ via FFT, then the residual is evaluated with a discrete Laplacian stencil. The spectral-vs-discrete mismatch prevents the trivial zero solution.
2. **28×28 κ maps** via learned upsampling with BatchNorm and GELU (vs 7×7 in V1).
3. **Deflection features:** |∇ψ| statistics fed to the classifier alongside κ statistics.

Training is stabilized with gradient clipping (max norm 1.0) and a differential learning rate (backbone 1e-5, heads 3e-4).

#### V2 Training curves

The physics loss is now non-trivial (stays non-zero), but the val accuracy does not improve over the baseline.

![Training curves — V2](figures/training_curves_v2.png)

#### Three-way ROC comparison

V2 achieves 0.9949 macro AUC — slightly below both V1 and the baseline.

![Three-way ROC](figures/roc_three_way.png)

#### V2 Per-class convergence, potential, and deflection

Row 1: mean κ. Row 2: mean ψ (lensing potential). Row 3: mean |∇ψ| (deflection magnitude). Despite higher resolution and a non-trivial Poisson constraint, per-class statistics remain nearly identical across all three classes.

![Per-class kappa/psi/deflection — V2](figures/kappa_psi_defl_v2.png)

### Confusion matrices

Side-by-side comparison of Test I and Test VII V1 predictions. Both show similar error patterns: subhalo is most commonly confused with no\_sub.

![Confusion matrices](figures/confusion_matrices.png)

---

## Key Takeaway

Both V1 and V2 demonstrate that **soft auxiliary losses on predicted convergence maps do not improve classification over a strong vision backbone.** The network can satisfy any κ-based Poisson constraint without learning physically meaningful mass distributions, because nothing forces the predicted κ to correspond to the actual convergence of the lensing system.

This motivates the **LensPINN architecture** ([Ojha et al., NeurIPS ML4PS 2024](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf)), which solves this problem by embedding the lens equation structurally: the encoder predicts the deflection field α(θ), a non-trainable layer performs lensing inversion (image plane → source plane), and the decoder classifies based on the difference between the reconstructed source and the observed image. The physics is enforced by information flow, not by a loss term.

---

## Repository Structure

```
├── deeplense_test1_and_7.ipynb       # Main notebook (Tests I + VII)
├── README.md
└── figures/
    ├── training_curves_test1.png
    ├── roc_test1.png
    ├── training_curves_test7_v1.png
    ├── roc_comparison_v1.png
    ├── kappa_overlay_v1.png
    ├── kappa_per_class_v1.png
    ├── training_curves_v2.png
    ├── roc_three_way.png
    ├── kappa_psi_defl_v2.png
    └── confusion_matrices.png
```

## How to Run

The notebook runs on Google Colab with a T4 GPU. Total runtime is approximately 1.5 hours.

1. Download the dataset from the link provided in the test specification
2. Upload `dataset.zip` to your Google Drive
3. Update `WEIGHTS_DIR` and `DRIVE_ZIP_PATH` in the first code cell to match your Drive path
4. Open the notebook in Colab, select a GPU runtime, and run all cells

**Dependencies:** PyTorch, torchvision, scikit-learn, matplotlib, numpy (all pre-installed in Colab).

## References

- [Deep Learning the Morphology of Dark Matter Substructure](https://arxiv.org/abs/2008.12731) (Alexander et al., 2020)
- [LensPINN](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf) (Ojha et al., NeurIPS ML4PS 2024)
- [DeepLense Repository](https://github.com/ML4SCI/DeepLense)
