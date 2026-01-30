#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Learned Reconstruction Methods

== Recall: Inverse Problems


Let $X = RR^n$ be the image space and $Y = RR^m$ be the measurement space. The inverse problem is defined as:
$ A x = y $
where $A in RR^(m times n)$ is the forward operator.

*Instances in Medical Imaging:*
- Computed Tomography (CT): 
  - $y$ is the sinogram data.
  - $A$ is the Radon transform.
- Reconstruction variants:
  - *Full-view CT*: Dense sampling of projections.
  - *Sparse-view CT*: Reduced number of projections (ill-posed problem).

#remark()[
  *Handwritten Flowchart:*
  CT Acquisition $arrow$ X-ray Projection Data ($y$) $arrow$ Filtered Backprojection ($A^(-1)$) $arrow$ Reconstructed Image ($x$).
]

== Deep Learning Approaches


There are three main paradigms for integrating Deep Learning into the reconstruction pipeline:

1. Post-processing:
   Applying a Neural Network (NN) to an initial reconstruction (e.g., FBP) to remove artifacts.
   $ y arrow.r "FBP" arrow.r x_"initial" arrow.r NN arrow.r x_"final" $

2. Pre-processing:
   Applying a NN to the raw data (sinogram/k-space) before reconstruction.
   $ y arrow.r NN arrow.r y_"full" arrow.r "FBP" arrow.r x $

3. Learned Inverse / Model-based Reconstruction:
   Replacing or augmenting the reconstruction operator itself.

=== Post-processing Approach: FBPConvNet


The FBPConvNet uses a U-Net architecture to refine sparse-view FBP reconstructions.
- Architecture: U-Net with skip connections and concatenation.
- Spatial Dimension: $512 times 512$.
- Operations: $3 times 3$ convolutions, Batch Normalization (BN), ReLU, and $2 times 2$ max pooling.

#example(title: "Performance Comparison")[
  Results for sparse-view CT reconstruction:
  - FBP: SNR 24.06
  - Total Variation (TV): SNR 29.64
  - FBPConvNet: SNR 35.38
]
*Reference*: Jin et al. (2017), "Deep convolutional neural network for inverse problems in imaging".

=== Pre-processing Approach: RAKI


RAKI (Scan-specific Robust Artificial-neural-networks for K-space Interpolation) is a database-free method for fast MRI imaging.
- It learns to interpolate missing k-space data from the auto-calibration signal (ACS) of the specific scan.
- Outperforms classical GRAPPA, especially at high acceleration rates (Rate 4 to 6).

*Reference*: Akçakaya et al. (2019), "Scan-specific robust artificial-neural-networks for k-space interpolation (RAKI) reconstruction".

== Model-based Reconstruction


In model-based approaches, we estimate the solution via a reconstruction operator $B(y)$ that approximates the inverse $A^(-1)$.

#definition(title: "Variational Formulation")[
  $ B(y) = arg min_x 1/2 norm(A x - y)_2^2 + R(x) $
  Where:
  - $1/2 norm(A x - y)_2^2$ is the data fidelity term.
  - $R(x)$ is the regularization term (prior knowledge).
]

=== Learned Inversion: AUTOMAP


AUTOMAP learns the entire mapping from sensor domain to image domain using a deep network.
*Reference*: Zhu et al. (2018), "Image reconstruction by domain-transform manifold learning," Nature.

== Learned Model-based Reconstruction


Modern methods focus on learning the regularization functional $R(x)$ or the optimization steps.

*Key Learning Principles:*
1. Bilevel Optimization: Learning parameters by solving an optimization problem within another.
2. Contrastive Learning: Learning representations by comparing positive and negative pairs.
3. Distribution Matching: Ensuring the reconstructed distribution matches the ground truth distribution.
4. Plug & Play (PnP): Using a pre-trained deep denoiser as a proximal operator in iterative algorithms.

#remark()[
  The evaluation of learned regularization is a critical area of current research (e.g., Hertrich et al., 2025).
]

=== Plug & Play Optimization


PnP replaces the traditional proximal operator with a deep denoiser $D_sigma$:
$ x^(k+1) = D_sigma (x^k - eta nabla f(x^k)) $
This allows the use of state-of-the-art denoisers without explicitly defining $R(x)$.

*References*: Venkatakrishnan et al. (2013); Zhang et al. (2021).
