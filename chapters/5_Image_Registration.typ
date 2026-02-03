
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

#figure(
  image("../assets/moving_image.png", height: 150pt),
)

#figure(
  image("../assets/registration_example.png", height: 150pt),
)
== Variational Approach to Registration


Registration is typically formulated as an optimization problem where we seek the optimal transformation parameters $theta$:

#definition(title: "Variational Formulation")[
  $ min_theta S(f, m circle.small T_theta) + R(T_theta) $
  where
  - $S(f, m circle.small T_theta)$ is the Similarity Metric which measures how well the fixed image $f$ matches the wraped image $m circle.small T_theta$.
  - $R(T_theta)$ is the Regularization term (ensures the transformation is physically plausible or smooth and not completely overfits local noise).
]
After we applyed the transform to the moved image, we need to wrap the image on the underying pixelgrid. There are different interpolation strategies for that:
#figure(
  image("../assets/warp.png", height: 150pt),
)

#figure(
  image("../assets/interpoltation.png", height: 150pt),
)
To choose a constant gradient is not a good idea as the function is not continuous and the gradient is 0. Most common is linear or spline interpolation, but spline interpolation is not currently implemented in pytorch.

== Interpolation by B-splines

$ hat(m)(x) = sum_(k=1)^(N_k) theta_k beta^n ( (x - mu_k) / sigma ) $

- $hat(m)(x)$: The interpolated value at position $x$. In image processing, this is often the estimated intensity of a pixel at a non-integer coordinate.
- $N_k$: The total number of basis functions (or control points) being used.
- $theta_k$: The interpolation coefficients (weights). These are the values we solve for to ensure the spline fits our data points.
- $beta^n$: The B-spline basis function of degree $n$. This determines the "shape" of the influence each control point has.
- $x$: The specific location where you want to calculate the value.
- $mu_k$: The center of the $k^"th"$ basis function. This is typically the pixel center.
- $sigma$: The spacing or scale between the basis functions. This controls how "stretched" or compressed the B-spline is.

For the 0-th order it looks like this: It is 1 inside a small window and 0 everywhere else. It represents the simplest form of interpolation where you just take the value of the closest pixel.

$ beta^0(x) = cases(
  1 & "if" |x| < 1/2,
  1/2 & "if" |x| = 1/2,
  0 & "else"
) $

$ beta^n (x) = [ underbrace(beta^0 * beta^0 * dots * beta^0, (n+1) " times") ] (x) $
- $*$: This symbol denotes convolution.
- $n$: The degree of the spline.
The Logic: A B-spline of degree $n$ is created by convolving the box function ($beta^0$) with itself $n$ times.

*Examples*:
- $beta^1$ (linear) results in a triangle shape.
- $beta^3$ (cubic) results in a smooth, bell-like curve. 

As $n$ increases, the function becomes smoother and converges toward a Gaussian distribution.

The regularizer helps, because when there are bumps, we dont want to fit them perfectly. This makes sense it we have a high resolution image with some noise. 


Thin plate splines:
$ min_theta 1/2 ||hat(m)_theta (x) - m(x)||^2_(Omega_M) + alpha integral_(Omega_M) |nabla^2 hat(m)(x)|^2 d x $

#figure(
  image("../assets/interpolation_bspline.png", height: 150pt),
)

== Transformation Models


Global Linear Transformation Models:
   - *Rigid*: Rotation and translation (6 degrees of freedom in 3D).
   - *Affine*: Includes scaling and shearing.

$ T(x) = A x + b $

$
"translation" &: T(x) = x + b \
"rotation" &: T(x) = R x + 0 \
"rigid / Euclidean" &: T(x) = R x + b \
"affine" &: T(x) = A x + b
$
#figure(
  image("../assets/linear_transformation.png", height: 80pt),
)


Non-linear Transformation Models:
   - Allows for local deformations (e.g., organ movement, breathing).
   - Often parameterized by B-Splines or displacement fields.

#figure(
  image("../assets/inr.png", height: 80pt),
)


== Similarity Metrics


The choice of similarity metric depends on whether the images are from the same modality (intra-modal, e.g. fit 2 MRI scans that were taken at 2 different time points) or different modalities (inter-modal, e.g. fit a MRI scan and a CT scan together).

- Sum of Squared Differences (SSD): Best for intra-modal images with linear intensity relationships.
  $ S_"SSD" (m compose T, f) = integral (f(x) - m(T(x)))^2 d x $
- Normalized Cross Correlation (NCC): Robust to linear intensity changes. Cross Correlation gives high values when both vectors point in similar directions and negative value when they point into the opposite direction.
  $ S(m compose T, f) = - frac(
  integral_(Omega_f) (f(x) - mu_f) (m(T(x)) - mu_m) d x,
  sqrt(integral_(Omega_f) (f(x) - mu_f)^2 d x) dot sqrt(integral_(Omega_f) (m(T(x)) - mu_m)^2 d x)
) $ So for different modalities it is good that it utilizes correlation instead of a perfect fit, but it is a bit more complicated to compute.
- Normalized Gradient Field (NGF): Matches the edges/gradients of the images.
  $ S(m compose T, f) = integral_Omega_f 1 - | (nabla f(x)^top) / (||nabla f(x)||_eta^2) dot (nabla m(T(x))) / (||nabla m(T(x))||_eta^2) | d x $

  $ ||x||_eta^2 = eta^2 + sum_i x_i^2 $ The advantage is that it focuses on edges not intensities and not directions. Multiple modalities are supported.
- Mutual Information (MI): The standard for multi-modal registration (e.g., MR to CT). It measures the statistical dependence between image intensities.
  $ S(m compose T, f) = D_"KL"  (p_(m, f) || underbrace(p_m times.o p_f, "outer product") ) $

  where $p_f$ is the  histogram of $f(x)$, $p_m$ is the histogram of $m(T(x))$ and $p_(m,f)$ is the joint histogram of $f$ and $m(T(x))$

  $ = - sum_(hat(m) in B_M) sum_(hat(f) in B_F) p_(m, f)(hat(m), hat(f)) dot log( frac(p_(m, f)(hat(m), hat(f)), p_m (hat(m)) dot p_f (hat(f))) ) $


  Here $B_M$ and $B_F$ denote the bins of the histograms of $m$ and $f$, respectively. The advantages are that they are suited for multiple modalities and they are very powerful (as they are most general). Their disadvantages are that they are very non-convex and have many hyperparameters.

== Regularization


Regularization prevents "unrealistic" warping, such as folding the image onto itself.
- Implicit regularization: Built into the model architecture or transformation model (e.g., low-resolution B-spline grid or only rigid transform).
- Explicit regularization: A penalty term added to the loss function (e.g., Diffusion, Elastic, or Total Variation regularizers).


=== Diffusion regularization:
$ R(theta) = integral_(Omega_m) |nabla T_theta (x)|^2 d x = ||nabla T||^2_(Omega_m) $

It is easy to compute but there are smooth edges in the transform $T$ as hard edges which mean big gradients are penalized.

=== Bending energy:
$ R(theta) = integral_(Omega_m) ||nabla^2 T_theta (x)||^2_F d x $

It is easy to compute and it penalizes the curvature.

=== Jacobian regularization:
$ R(theta) = integral_(Omega_m) |1 - det nabla T_theta (x)| d x $

It is more complex to compute and it penalizes local area changes.

== Optimization and Deep Learning

=== Optimization Tricks
- Coarse-to-fine strategy: Start by registering downsampled (low-res) versions of the images and gradually increase resolution to avoid local minima.
- Sequential complexity: Start with rigid/affine transforms before moving to non-linear deformations.

=== Deep Learning Approaches
Deep learning has shifted registration from iterative optimization to "one-shot" prediction.

- VoxelMorph: A Unet-based framework that learns to predict the displacement field between two images in a single forward pass.
#figure(
  image("../assets/voxelmorph.png", height: 190pt),
)


- Implicit Neural Representations (INR): Representing the transformation as a continuous function $T(x)$ parameterized by a neural network (e.g., using periodic activation functions like SIREN).
#figure(
  image("../assets/inr_paper.png", height: 150pt),
)


