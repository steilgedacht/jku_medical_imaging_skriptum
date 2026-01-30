
#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Image Registration

== What is Image Registration?


Image registration is the process of transforming different sets of data into one coordinate system.

#definition(title: "Fundamental Components")[
  - *Fixed image* $f(x)$: The reference image that remains stationary.
  - *Moving image* $m(x)$: The image that is deformed to match the fixed image.
  - *Transformation* $T$: A mapping $T: x arrow.r T(x)$ that defines how the moving image is warped.
  - *Warped image*: The result of applying the transformation to the moving image, denoted as $(m circle.small T)(x) = m(T(x))$.
]


== Variational Approach to Registration


Registration is typically formulated as an optimization problem where we seek the optimal transformation parameters $theta$:

#theorem(title: "Variational Formulation")[
  $ min_theta S(f, m circle.small T_theta) + R(T_theta) $
  Where:
  - $S(f, m circle.small T_theta)$ is the Similarity Metric (measures how well the images match).
  - $R(T_theta)$ is the Regularization term (ensures the transformation is physically plausible or smooth).
]

== Transformation Models


1. Global Linear Transformation Models:
   - *Rigid*: Rotation and translation (6 degrees of freedom in 3D).
   - *Affine*: Includes scaling and shearing.
   
2. Non-linear Transformation Models:
   - Allows for local deformations (e.g., organ movement, breathing).
   - Often parameterized by B-Splines or displacement fields.

#remark()[
  *Handwritten Note on Interpolation*: When warping an image, we often need to calculate values at non-integer coordinates. B-Splines (1D and 3D cases) are commonly used for smooth interpolation.
]


== Similarity Metrics


The choice of similarity metric depends on whether the images are from the same modality (intra-modal) or different modalities (inter-modal).

- Sum of Squared Differences (SSD): Best for intra-modal images with linear intensity relationships.
  $ S_"SSD" = integral (f(x) - m(T(x)))^2 d x $
- Normalized Cross Correlation (NCC): Robust to linear intensity changes.
- Normalized Gradient Field (NGF): Matches the edges/gradients of the images.
- Mutual Information (MI): The standard for multi-modal registration (e.g., MR to CT). It measures the statistical dependence between image intensities.

== Regularization


Regularization prevents "unrealistic" warping, such as folding the image onto itself.
- Implicit regularization: Built into the model architecture or transformation model (e.g., low-resolution B-spline grid).
- Explicit regularization: A penalty term added to the loss function (e.g., Diffusion, Elastic, or Total Variation regularizers).

== Optimization and Deep Learning


=== Optimization Tricks
- Coarse-to-fine strategy: Start by registering downsampled (low-res) versions of the images and gradually increase resolution to avoid local minima.
- Sequential complexity: Start with rigid/affine transforms before moving to non-linear deformations.

=== Deep Learning Approaches
Deep learning has shifted registration from iterative optimization to "one-shot" prediction.

- VoxelMorph: A CNN-based framework that learns to predict the displacement field between two images in a single forward pass.
- Implicit Neural Representations (INR): Representing the transformation as a continuous function $T(x)$ parameterized by a neural network (e.g., using periodic activation functions like SIREN).

#example(title: "Key References")[
  - Balakrishnan et al. (2019), *VoxelMorph: a learning framework for deformable medical image registration*.
  - Wolterink et al. (2022), *Implicit neural representations for deformable image registration*.
]
