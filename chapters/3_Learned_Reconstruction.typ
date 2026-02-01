#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Learned Reconstruction Methods

== Recall: Inverse Problems

Let $X = RR^n$ be the image space and $Y = RR^m$ be the measurement space. The inverse problem is defined as: $A x = y$ where $A in RR^(m times n)$ is the forward operator.

One instances of that in Medical Imaging is Computed Tomography (CT) where $y$ is the sinogram data $A$ is the Radon transform.

== Deep Learning Approaches


There are three main paradigms for integrating Deep Learning into the reconstruction pipeline:

1. Post-processing:
   Applying a Neural Network (NN) to an initial reconstruction (e.g., Filtered Backpropagation FBP) to remove artifacts.
   $ y arrow.r "FBP" arrow.r x_"initial" arrow.r "NN" arrow.r x_"final" $

2. Pre-processing:
   Applying a NN to the raw data (sinogram/k-space) before reconstruction.
   $ y arrow.r "NN" arrow.r y_"full" arrow.r "FBP" arrow.r x $

3. Learned Inverse / Model-based Reconstruction:
   Replacing or augmenting the reconstruction operator itself.
   $ y arrow.r "NN" arrow.r x $
#figure(
  image("../assets/DeepLearning_Approaches.png", height: 170pt),
)

=== Post-processing Approach: FBPConvNet


The FBPConvNet uses a U-Net architecture to refine sparse-view FBP reconstructions.
- Architecture: U-Net with skip connections and concatenation.
- Spatial Dimension: $512 times 512$.

#figure(
  image("../assets/FBPConvNet.png", height: 170pt),
)

#figure(
  image("../assets/learned_reconstruction.png", height: 170pt),
)
=== Pre-processing Approach: RAKI


RAKI (Scan-specific Robust Artificial-neural-networks for K-space Interpolation) is a database-free method for fast MRI imaging. Here they do not measure every single fourier coefficent, but uses CNN layers to learn to interpolate missing data from the auto-calibration signal (ACS) of the specific scan. It also outperforms classical GRAPPA, especially at high acceleration rates.



#figure(
  image("../assets/preprocessing_apporach.png", height: 170pt),
)

#figure(
  image("../assets/GRAPPA_vs_RAKI.png", height: 170pt),
)
== Model-based Reconstruction


In model-based approaches, we have 2 options for getting a solution: 
- via a reconstruction operator $B(y)$ that approximates the inverse $A^(-1)$ directly which is a robust approximation (Fourier scans are often noisy)
- via an iterative algorithms by solving $ B(y) = arg min_x 1/2 norm(A x - y)_2^2 + R(x) $

=== Learned Inversion: AUTOMAP


The most popular algorytm for learned inversion is AUTOMAP. You put in the fourier data and then they directly put it to a fully connected layer followed by some Convolution layers. That might be not a good idea as when the fourier signal is only shifted by one pixel, you have entirely different results. 

#figure(
  image("../assets/automap.png", height: 170pt),
)

== Learned Model-based Reconstruction

===  Recall: Lipschitz Continuous Gradients
Let $f: RR^N -> RR$ be differentiable and $0 < L < infinity$ such that:
$ ||nabla f(x) - nabla f(y)|| <= L ||x - y|| quad forall x, y in RR^N $

Then, we have the quadratic upper bound:
$ f(y) <= f(x) + chevron.l nabla f(x), y - x chevron.r + L/2 ||x - y||^2 quad forall x, y in RR^N $

===  Proximal Gradient Method (PGM)
To optimize a function $f: RR^N -> RR$ decomposable into $f(x) = g(x) + h(x)$, where $g(x) in C^(1,k)(RR^N)$ and $h$ is closed (l.s.c), convex, and proper (proximal mapping can be computed).

Iteration Rule:
$ x_(k+1) = "prox"_(1/L h) (x_k - 1/L nabla g(x_k)) $

Definition of Proximal Operator:
$ "prox"_(1/L h) (macron(x)) = arg min_x (1/L h(x) + 1/2 ||x - macron(x)||_2^2 ) $

Accelerated Version: FISTA (Fast Iterative Shrinkage & Thresholding Algorithm).


===  Derivation of PGM
Assuming $nabla g$ is Lipschitz continuous:
$ g(y) <= g(x) + chevron.l nabla g(x), y - x chevron.r + L/2 ||y - x||^2 $

This is $g(x)$, but we want also to have a $h(x)$ to have the same form as in the definition above. So we add $h(y)$ to both sides:
$ f(y) = g(y) + h(y) <= g(x) + h(y) + chevron.l nabla g(x), y - x chevron.r + L/2 ||y - x||^2 =: tilde(f)(y) $
We want to have a upper bound to our function and we want to optimize $f$. 
To minimize the upper bound $tilde(f)(y)$, we take the subgradient with respect to $y$:

$ partial / (partial y) tilde(f)(y) = partial / (partial y) h(y) + nabla g(x) + L(y - x) $
And then we can set it to $0$
$ partial  / (partial y) h(y) + L y =  -nabla g(x) + L x $
$ y + 1/L partial  / (partial y) h(y) = x - 1/L nabla g(x) $
$ underbrace( (I + 1/L partial / (partial y)  h )(y), "This is a function of y") = x - 1/L nabla g(x) $

Solving for $y$ using the resolvent (equivalent to proximal mapping):
$ y = underbrace((I + 1/L partial h)^(-1),"resolvent equivalent to prox") (x - 1/L nabla g(x)) $
$ y = "prox"_(1/L h) (x - 1/L nabla g(x)) $


#example(title:"Lasso")[
  $ min_x 1/2 ||A x - y||^2_2 + lambda ||x||_1 $
  Where $g(x) = 1/2 ||A x - y||^2_2$ and $h(x) = lambda ||x||_1$. Site note: The derivative of $lambda ||x||_1$ is not Lipschitz continuous 

  gradient: $ nabla g(x) = A^T (A x - y) $
  Proximal Operator (Soft Thresholding):
  $ "prox"_(1/L h)(y)_i = max(|y_i| - lambda/L, 0) dot "sign"(y_i) $

  Computing $L$:
  $ ||nabla g(x) - nabla g(z)||_2 <= L ||x-z||_2 $
  $ ||A^T A(x - y) -  A^T(A z-y)||_2 = ||A^T A(x-z)||_2 <=L ||x-z||_2 $
  $ ||A^T A||_2 ||x - z||_2 <=  L ||x-z||_2 $
  $ ||A^T A||_2 <= L $ (The largest singular value of A)
]


#example(title:"Fields of Experts regularization for MRI reconstruction")[
  $ A(x) = M dot.o F(x) = D_M F x  $
  where $D_M$ is a diagonal matrix

  Our goal is to solve

  $ min_x 1/2 ||A(x) - y||_2^2 + rho (W x) $

  Where $1/2 ||A(x) - y||_2^2$ is smooth and convex, and $rho$ is smooth (but not necessarily convex). The $1/2 ||A(x) - y||_2^2$  will be our $h(x)$ and the $rho$ is our $g(x)$

  $ nabla g(x) = W^T nabla_x rho(W x) $

  $ "prox"_(1/L l)(z) &= arg min_x 1/2 ||x - z||^2 + 1/L h(x) \
  &= arg min_x underbrace(1/2 ||x - z||^2 + 1/L 1/2 ||A(x) - y||_2^2, l(x)) $

  Now we can compute again the graident and set it to 0:

  $ (partial l) / (partial x) = (x - z) + 1/L A^T(x)(A(x) - y) = 0 \
  x + 1/L F^T M (M F x - y) = z \
  x + 1/L F^T M M F x = z + 1/L F^T M y $

  Assuming $M^2 = M$ (projection matrix):
  $ (I  + 1/L F^T M F) x &= z + 1/L F^T M y $

  Given $F^T F = I $:
  $ (F^T F + 1/L F^T M F) x &= z + 1/L F^T M y \
  F^T (I  + 1/L M) F x &= z + 1/L F^T M y \
  (I  + 1/L M) F x &= F z + 1/L M y \
  F x &= (F z + 1/L M y) / (I  + 1/L M) $

    $ x = F^T ((F z + 1/L M y) / (I  + 1/L M)) $
]

== Key Learning Principles:
1. Bilevel Optimization: Learning parameters by solving an optimization problem within another. (= supervised learning)
2. Contrastive Learning: Learning representations by comparing positive and negative pairs. (= semi-supervised)
3. Distribution Matching: Ensuring the reconstructed distribution matches the ground truth distribution. (= unsupervised)
4. Plug & Play (PnP): Using a pre-trained deep denoiser as a proximal operator in iterative algorithms. (= unsupervised)

== Bilevel Optimization

Given a set of paired training samples $D = (x_i, y_i)_{i=1}^n$, in bilevel optimization, we want to solve the following learning problem:

Upper level problem:
$ min_theta L(theta) = sum_(i=1)^n ||hat(x)_i (theta, y_i) - x_i||_2^2 "subject to" $

 Lower level problem
$ hat(x)_i (theta, y_i) = arg min_x { E_theta (x, y_i) = 1/2 ||A x - y_i||_2^2 + R_theta (x) } $

The challenge is to compute $ frac(partial L, partial theta) = sum_(i=1)^n (frac(partial hat(x)_i (theta, y_i), partial theta))) (hat(x)_i (theta, y_i) - x_i) $

Here it is difficult to compute $ frac(partial hat(x)_i (y_i, theta), partial theta) $

Let's do it step by step: 
1. Solve the lower level problem (with sufficient precision). $ nabla_x E_theta (hat(x)_i (y_i, theta), y_i) approx 0 $
2. Assume that $E_theta in C^2(RR^N)$ with invertible Hessian $H(theta) = nabla_x^2 E_theta$. Then, the Implicit Function Theorem (IFT) guarantees the existence of a continuously differentiable local solution map $theta arrow hat(x)_i (y_i, theta)$.
3. The first order optimality condition of $E_theta$ is:
   $ frac(partial, partial theta) (nabla_x E(hat(x)(theta), theta)) = frac(partial, partial theta) (0) = 0 $
   $ frac(partial hat(x)(theta), partial theta) nabla_x^2 E(hat(x)(theta), theta) + frac(partial, partial theta) nabla_x E(hat(x)(theta), theta) = 0 $
   $ frac(partial hat(x)(theta), partial theta) = - underbrace( frac(partial, partial theta) nabla_x E(hat(x)(theta), theta), "Jacobian o the" \ "lower level energy gradient" ) underbrace(( nabla_x^2 E(hat(x)(theta), theta) )^(-1), "inverse Hessian of" \ "lower level energy") $


=== Unrolling (truncated optimization):

Using the IFT requires that we have $nabla E(hat(x)) approx 0$. So, we need to approximate this using an optimization algorithm (e.g., PGD)
$ x_(k+1) = T(x_k; theta) quad "for" k = 0 dots K-1 $
E.g., if $T$ implements gradient descent, we have $T: x arrow x - 1/2 nabla E_theta (x)$.
If we use $K$-steps, we get a computational chain:
$ x_0(y) arrow^(T_theta) x_1(theta) arrow^(T_theta) x_2(theta) dots arrow^(T_theta) x_K (theta) $
In unrolling, we simply set $hat(x) = x_K (theta)$; $L(theta) = sum_(i=1)^n ||x_K^(i)(theta) - x_i||_2^2$. The gradient $frac(partial L(theta), partial x_K)$ can simply be computed by back-propagation. The advantage is that it is easy implementable, but you have large memory consumtion


=== Jacobian-free backpropagation (truncated backpropagation)

Assume $x_K$ approaches the optimum $nabla E(x_K) approx 0$. Instead of backprop through the entire sequence $(x_k)_{k=0}^K$, only the last $K_B$ steps are considered:
$ frac(partial, partial theta) x_K (theta) approx sum_(k=K-K_B)^K_B frac(partial, partial theta) T_theta(x_(k-1)) dot nabla_x T_theta(x_k) dots nabla_x T_theta(x_K) $
If the lower level problem is sufficiently regular, this approximation error decays exponentially with $K_B$. Interestingly, in practice often $K_B = 1$ works quite well. So this method is easy to implement and has a small memory requirement. However you only get a approximation of the real solution.

== Contrastive Learning
In contrast to bilevel-opt. that tries to learn a reconstruction scheme end-to-end, contrastive learning learns regularizers by contrasting "good" & "bad" images.

=== Adversarial Regularization (AR)

Let $\{x_i\}_(i=1)^n tilde p_X$ be samples from desired images (= no measurements required). Let $A^dagger$ be a simple reconstruction operator (e.g., FBP, regularized pseudo-inverse). Then $A^dagger$ yields a push-forward distribution $p_(A^dagger y)$ of denablaed images.

The key idea of AR is to train a model to be a discriminator (c.f. GAN), i.e.,

$ L(theta) = 1/n sum_(i=1)^n R_theta (x_i) - 1/m sum_(j=1)^m R_theta (A^dagger y_j) + underbrace(lambda E_x [ (||nabla_x R_theta (x)|| - 1)_+^2 ], "gradient penalty in Wasserstein GANs") $

where $R_theta: RR^N -> RR_+$ and $y_j = A x_j + n$ with $n tilde cal(N)(0, sigma^2 "I")$.

Gradient penalty in Wasserstein GANs: penalizes deviations from the 1-Lipschitz assumption in WGANs. The Lipschitz constant gives us the maximal gradient of a function.
An advantage is that it is an easy training problem (at least to code). A disadvantage is that the training could be unstable and the balancing regularization & data fidelity is hard during inference.

== Distribution Matching


Recall that regularizer $R(x)$ is associated with a Gibbs distribution:
$ p_theta (x) = 1/z exp(-R_theta (x)) $

The goal here is to align (match) $p_theta (x)$ and $p_x (x)$.


=== Maximum Likelihood Training

$ p_theta (x) = 1/z_theta exp(-R_theta (x)) "with" z_theta = integral_(bb(R)^n) exp(-R_theta (x)) d x $

$ theta &= arg max_theta EE_(x ~ p_x) [log p_theta] = arg min_theta EE_(x ~ p_x) [-log p_theta] = arg min_theta D_(K L)(p_x || p_theta) \
&= arg min_theta EE_(x ~ p_x) [R_theta (x)] + log z_theta $

$ nabla_theta D_(K L) (dot || dot) &= EE_(x ~ p_x) [nabla_theta R_theta (x)] + partial / (partial theta) log z_theta = EE_(x ~ p_x) [nabla_theta R_theta (x)] + 1/z_theta partial z_theta / (partial theta) \
&= EE_(x ~ p_x) [nabla_theta R_theta (x)] + 1/z_theta integral_(bb(R)^n) exp(-R_theta (x)) dot (-1) dot nabla_theta R_theta (x) d x \
&= EE_(x ~ p_x) [nabla_theta R_theta (x)] - integral_(bb(R)^n) exp(-R_theta (x)) / (integral_(bb(R)^n) exp(-R_theta (tilde(x))) d tilde(x)) dot nabla_theta R_theta (x) d x \
&= EE_(x ~ p_x) [nabla_theta R_theta (x)] - E_(x ~ p_theta) [nabla_theta R_theta (x)] $

Which is the contrastive divergence
An advantage is the easy training objective, but we need to sample from $p_theta$ which is not that easy. 

=== Score Matching

If we change the divergence from KL to the Fisher divergence, we get:
$ arg min_theta cal(L)_("ESM")(theta) = 1/2 EE_(x ~ p_x) [norm(nabla_x log p_x (x) - nabla_x log p_theta (x))_2^2] $

This aligns the Stein score $nabla log p_x$, but in practice, we cannot compute this. In reality, we approximate $p_x$ by Gaussian smoothing and get $p_sigma = p * G_sigma$. Then, we get the denoising score matching objective:

$ arg min_theta cal(L)_("DSM")(theta) = 1/2 EE_(x ~ p_x, n ~ G_sigma) [norm(nabla_x log p_theta (x+n) - (-n/sigma^2))_2^2] $
$ s_theta (x+n) approx y = x+n $

So one plus point here is a easy training and a problem is how to choose $sigma$ (training & inference schedule)

=== Proof of equivalence of ESM & DSM

$ cal(L)_("ESM")(theta) = EE_(y ~ p_sigma) [1/2 norm(s_theta (y) - nabla_y log p_sigma (y))_2^2] = EE_(y ~ p_sigma) [1/2 norm(s_theta (y))_2^2 - chevron.l s_theta (y), nabla_y log p_sigma (y) chevron.r + C] $

$ S(theta) &= integral_Y p_sigma (y) chevron.l s_theta (y), nabla_y log p_sigma (y) chevron.r d y = integral_Y chevron.l s_theta (y), nabla_y p_sigma (y) chevron.r d y \
&= integral_Y chevron.l s_theta (y), nabla_y integral_X p_x (x) p(y|x) d x chevron.r d y = integral_Y integral_X p_x (x) dot chevron.l s_theta (y), nabla_y p(y|x) chevron.r d x d y \
&= integral_Y integral_X p_x (x) dot p(y|x) dot chevron.l s_theta (y), nabla_y log p(y|x) chevron.r d x d y \
&= EE_(x,y ~ p_(x,y)) [chevron.l s_theta (y), nabla_y log p(y|x) chevron.r] $

$ => cal(L)_("ESM")(theta) = EE_(y ~ p_sigma) [1/2 norm(s_theta (y))_2^2] - EE_(x,y ~ p_(x,y)) [chevron.l s_theta (y), nabla_y log p(y|x) chevron.r] + C_1 $

$ <=> EE_(x,y ~ p_(x,y)) [1/2 norm(s_theta (y))_2^2 - chevron.l s_theta (y), nabla_y log p(y|x) chevron.r + C_1 + 1/2 norm(nabla_y log p(y|x))_2^2] $ 
$ &= EE_(x,y ~ p_(x,y)) [1/2 norm(s_theta (y) - nabla_y log p(y|x))_2^2] \
&= cal(L)_("DSM")(theta) + C_2 $

$ C_2 = - norm(nabla y log p(y|x))_2^2 + norm(nabla_y log p_sigma (y))_2^2 $

$ p(y|x) = |2 pi sigma^2 I |^(-1/2) exp(- 1 / (2 sigma^2) norm(x - y)_2^2) $

To summarize: The idea of score matiching is to match the log gradients of the distribution. This matching is hard, that's why we approximate it with smoothing, adding some noise. And we can show that by adding this noise, we can come up with a training objective which is $arg min_theta cal(L)_("DSM")(theta) = 1/2 EE_(x ~ p_x, n ~ G_sigma) [norm(nabla_x log p_theta (x+n) - (-n/sigma^2))_2^2] $ which is easy to compute. 


== Plug & Play Optimization

We would like to solve:
$ hat(x) = arg min_x 1/2 ||A x - y||^2 + lambda R(x) $

The Half Quadratic Splitting (HQS) algorithm decouples these terms:
$ hat(x) = arg min_{x, z} 1/2 ||A x - y||^2 + lambda R(z) quad "s.t." x = z $

This corresponds to the augmented Lagrangian (penalty method):
$ L_mu (x, z) = 1/2 ||A x - y||^2 + lambda R(z) + mu/2 ||x - z||^2 $

HQS Steps:
- $x_k = arg min_x 1/2 ||A x - y||^2 + mu/2 ||x - z_(k-1)||^2 = "prox"_(1/mu ||A x - y||^2)(z_(k-1))$
- $z_k = arg min_z mu/2 ||x_k - z||^2 + lambda R(z) = "prox"_(lambda/mu R)(x_k)$


Key Idea: ($z_k$) is an image denoising problem. We can replace this optimization sub-problem (the proximal operator) with a pre-trained deep image denoiser.

Pros (+): We only need to train a denoiser once. \
Cons (-): Have to tune hyperparameters (e.g., $mu$) during inference.


=== Algorithm: Plug-and-play image restoration with deep denoiser prior (DPIR)

#rect(width: 100%, stroke: 0.5pt, inset: 10pt)[
  *Input:* Deep denoiser prior model, denablaed image $y$, denablaation operation $A$, image noise level $sigma$, $sigma_k$ of denoiser prior model at $k$-th iteration for a total of $K$ iterations, trade-off parameter $lambda$. \
  *Output:* Restored image $z_K$.

  1. Initialize $z_0$ from $y$, pre-calculate $alpha_k > lambda sigma^2 / sigma_k^2$.
  2. *for* $k = 1, 2, dots, K$ *do*
  3. $quad x_k = arg min_x ||y - A(x)||^2 + alpha_k ||x - z_{k-1}||^2$ // _Solving data subproblem_
  4. $quad z_k = "Denoiser"(x_k, sigma_k)$ // _Denoising with deep DRUNet denoiser_
  5. *end*
]
