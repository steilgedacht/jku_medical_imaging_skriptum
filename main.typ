#import "@preview/classicthesis:0.1.0": *

#show: classicthesis.with(
  title: "Medical Imaging",
  subtitle: "Skriptum",
  author: "Benjamin Bergmann",
  date: "2026",
)
#show heading: it => {
  set text(tracking: 0pt)
  it.body
}

#set par(first-line-indent: 0pt)



= Inverse Problems 

== What is an Inverse Problem?

There exit a "Forward Problem" which estimate the effect from the cause and then there is inverse Problem which estimates the cause from the effect. In the medical context that would be finding the cause illness given from a certain syntom/effect.  
Typically, the forward problem is ”easy” and well described. The challenge here is: We need to solve the inverse problem given only the observed effect of the
forward problem. 

As an Example from the real world: 
forward problem: The street becomes wet when it rains.
backward problem would be: We observe that the street is wet. Why?

There are multiple different causes:
• Rain
• Fog
• Cleaning

And this can be already problematic as we have multiple different options for what the cause could be.

=== Example: Computed Tomography

*Forward Problem*  
X-ray emitter and detector rotating around the body.  
Detectors measure the number of photons passing through the body and hitting the detector  

*Inverse Problem*  
Reconstruct the interior of the body from the measured detector signals.

Note that a CT Scan can be very large in file size. A scan from shoulder to belt line is already 18GB of data for just a single scan.
So we bascially have $y$ and we want to get to $x$

=== Example: Deconvolution

*Forward Problem*  
Observe a blurred image $ f = k * u $ on a domain $ Omega subset RR^2 $.

*Inverse Problem*  
Estimate the sharp image $ u: Omega -> RR $ given the blur kernel $ k: Omega times Omega -> RR_+ $ 

On of the oldest clasical methods to do that is the Wiener Filter.
Deconvolution is linked to Fourier $F$:

$ f= k * u $ 

$ F(f) = F(k) dot.o F(u) $

If we wanna do the inverse:

$ F^(-1) {F(f)} = F^(-1){F(k) dot.o F(u)} = f $  

where $dot.o$ is a pointwise multiplication

So a estimate $accent(u, hat)$ would be 

$ accent(u, hat) = F^(-1) (F(f)/F(k)) $

The only problem here is when we have 0 frequencies in the kernel. The Wiener Filtering introduces

$ accent(u, hat) = F^(-1) (F(f)/(I sigma^2 F(k))) $

== What is an Inverse Problem? (formal)

Given a matrix $A in RR^(m times n)$ and a vector $x in RR^n$ the forward problem is $y = A x in RR^m$ 

The inverse problem is: Given $A$ and $y$, estimate $x$.

== Definition: Vector Space

A non-empty set $V$ is a vector space over a field $FF in {RR, CC}$ if there are operations  of vector addition: $V times V -> V$  and scalar multiplication: $FF times V -> V$ satisfying the following axioms.

*Vector addition*
1. $u+v in V quad forall u,v in V$
2. $u + v = v + u$
3. $(u + v) + w = u + (v + w) quad forall u,v,w in V$
4. $exists 0 in V: u + 0 = u$ 
5. $forall u in V: exists -u: u + (-u) = 0$

*Scalar multiplication*
1. $a u in V$
2. $a (u + v) = a u + a v$
3. $(a + b) u = a u + b u$
4. $a (b u) = (a b) u$
5. $1 u = u$


== Vector Space Examples

- $RR^n = (x_1, dots, x_n)^T$
- Continuous functions: $C(RR^n, RR)$
- Once continuously differentiable functions: $C^1(RR^n, RR)$
- Lebesgue space:
  $L^2(RR^n) = { f | integral_(RR^n) abs(f(x))^2 d x < infinity }$
- Sobolev space:
  $H^1(RR^n) = { f in L^2 | integral_(RR^n) abs(nabla f(x))^2 d x < infinity }$


== Definition: Inverse Problem

Let $X, Y$ be vector spaces and $A: X -> Y$. 
The forward problem is $y = A x $.
The inverse problem is to find $x in X$ such that $ A x = y $.

== Definition: Well-Posedness (Hadamard)

We can now start to categorize inverse problems:
The inverse problem $A x = y$ is well-posed if:

1. *Existence:* a solution exists (EXISTENCE)
2. *Uniqueness:* the solution is unique (UNIQUENESS)
3. *Stability:* the solution depends continuously on the data (STABILITY)

If one condition fails, the problem is ill-posed.

== Well-Posedness Example

Let $X = Y = RR $ and $A:RR arrow.r RR, x arrow.r x^2$
Is this example well posed?

- Existence: for $y = -1$ no solution exists (if it would be $RR^+$ it would be okay)
- Uniqueness: for $y = 1$, $x = plus.minus 1$
- Stability: yes, since $A$ is continuous


== Definition: Inner Product

An inner product is a mapping

$ ⟨., .⟩: Y times Y -> FF $

with properties:

1. Symmetry: $⟨x, y⟩ = ⟨y, x⟩ quad x,y in Y$
2. Additivity: $⟨x + z, y⟩ = ⟨x, y⟩ + ⟨z, y⟩ quad x,y,z in Y$
3. Homogeneity: $⟨a x, y⟩ = a ⟨x, y⟩ quad x,y in Y quad a in RR$
4. Positivity: $⟨x, x⟩ >= 0$ and $⟨x, x⟩ = 0 <==> x = 0$

== Definition: Vector Norm

A vector norm is a vector space Y over a field $F$ is a map $norm(.): Y -> RR$ with:

1. Non-negativity:$norm(x) >= 0$
2. Definiteness:$norm(x) = 0 <==> x = 0$
3. Homogeneity: $norm(a x) = abs(a) norm(x)$
4. Triangle inequality: $norm(x + y) <= norm(x) + norm(y)$


== Definition: Matrix Norm

Let $norm(.)_a$ on $RR^n$ and $norm(.)_b$ on $RR^m$.

For $A in RR^(m times n)$ the induced matrix norm is $norm(A)_(a,b) = sup_(x != 0) (norm(A x)_b / norm(x)_a)$ 

== Injection, Surjection, Bijection

Let $A: X -> Y$

- Injective: $ A x_1 = A x_2 => x_1 = x_2 $
- Surjective: $forall y in Y, exists x in X: A x = y$
- Bijective: injective and surjective


== Null Space and Range

- Null space: $N(A) = { x in X | A x = 0 }$
- Range: $R(A) = { A x | x in X }$

== Connection to Hadamard

- Existence ⇔ $R(A) = Y$
- Uniqueness ⇔ $N(A) = {0}$
- Well-posed ⇔ $A$ bijective (and stable)


== Singular Value Decomposition

Let $A in RR^(m times n)$ then $A = U Sigma V^T$ where $Sigma = (sigma_1, dots, sigma_p)$ with $sigma_i > 0$ and $p = (A)$.

== Least Squares (m > n)

Solve $A x = y$ by minimizing $arg min_(x) norm(A x - y)^2$
Normal equations:
$A^T A x = A^T y$

== Minimum Norm Solution (n > m)

Underdetermined system $A x = y$

Choose the minimum norm solution:

$arg min_(x) norm(x)$ 
$text("subject to") A x = y$

Solution:
$x = A^T (A A^T)^(-1) y$


== Generalized Inverse

Using the SVD:

$A^dagger = V Sigma^(-1) U^T$

This interpolates between least squares and minimum norm solutions.


== Regularization

Instead of solving $A x = y$, solve

$arg min_(x) norm(A x - y)^2 + lambda R(x)$


== Typical Regularization Terms

- Tikhonov: $R(x) = norm(x)^2$
- L2: $R(x) = norm(x)^2 $
- H1: $R(x) = norm(nabla x)^2$
- L1: $R(x) = norm(x)_1$
- Total Variation: $R(x) = norm(nabla x)_1$


== Tikhonov Optimality Condition

$(A^T A + lambda I) x = A^T y$

== Probabilistic Interpretation

Assume noisy measurements: $y = A x + epsilon$
$epsilon ~ (0, sigma^2 I)$

Bayes’ rule yields: $arg max_(x) log(p(x | y))$ which is equivalent to $arg min_(x) norm(A x - y)^2 - log(p(x))$

Hence, regularization corresponds to MAP estimation.

= X-rays and CT

== Discovery of X-rays

- In 1895 Wilhelm Röntgen discovered “rays of mysterious origin”, later called X-rays.
- On 22.12.1895 the first radiograph of the hand of Röntgen’s wife was produced.
- This immediate medical application marks the birth of medical imaging.


== Nature and Properties of X-rays

- X-rays are electromagnetic waves.
- They are a form of ionizing radiation, i.e. radiation with enough energy to eject electrons from atoms.


== Ionizing Radiation

Two main forms:

1. *Particulate radiation*  
   Subatomic particles (electrons, protons, neutrons) with sufficient kinetic energy.

2. *Electromagnetic radiation*  
   Acts as wave or particle (photon).

EM radiation is ionizing if photon energy exceeds the hydrogen binding energy: $E > 13.6 e V$

Relations:
$E = h nu $  
$lambda = c / nu $

== Interaction of Energetic Electrons with Matter

When electrons hit matter:

- *Collision transfer* (~99%):  
  Energy transferred to other electrons → heat.

- *Radiative transfer* (~1%):  
  a) Inner-shell ionization → characteristic X-rays  
  b) Braking near nucleus → bremsstrahlung radiation


== Interaction of X-rays with Matter

*Photoelectric effect*  
Photon ejects an inner-shell electron:

$E_e = h nu - E_B$

- Filling the vacancy emits characteristic X-rays.
- Alternatively produces Auger electrons.

*Compton scattering*  
Photon interacts with outer-shell electrons, losing energy and changing direction.


== Generation of X-rays

X-rays are generated using an X-ray tube:

- Heated cathode emits electrons
- High voltage accelerates electrons
- Electrons hit anode → X-rays produced


== Attenuation of Electromagnetic Radiation

Consider a narrow monoenergetic X-ray beam.

Let:
- $N(x)$ = number of photons
- $mu(x)$ = linear attenuation coefficient

Photon loss:

$d N = -mu(x) N d x$

Divide and integrate:

$d N / N = -mu(x) d x$

$ln(N / N_0) = - integral mu(x) d x$

Resulting intensity:

$N = N_0 exp(- integral mu(x) d x)$

== Narrow Beam vs Broad Beam

- Broad beam: scattering contributes to detector signal.
- Monoenergetic assumption fails due to energy loss.

Solution:
- Collimation
- Narrow-beam geometry

Then attenuation law holds approximately.


== Linear Attenuation Coefficient

$mu(x)$ depends on:
- material
- photon energy

Higher $mu$ → stronger attenuation.


== Projection Radiography

Basic imaging equation:

$I = integral S(E) exp(- integral mu(x, E) d x) d E$

Assuming effective monoenergetic spectrum:

$I = I_0 exp(- integral mu(x) d x)$

Taking logarithm:

$-ln(I / I_0) = integral mu(x) d x$


== Blurring in Projection Imaging

Sources of blur:
- Finite focal spot (penumbra)
- Detector blur
- Compton scattering outside field of view


== Noise in Projection Imaging

Photon detection is a counting process:

$N ~  (N)$

Variance:

$"Var"(N) = N$

Signal-to-noise ratio:

$"SNR" = N / sqrt(N) = sqrt(N)$

To increase SNR:
- Increase photon count
- Use contrast agents


== Tomography

Tomography = imaging by sectioning a volume.

From Greek:
- *tomos* = slice
- *grapho* = to write


== Computed Tomography (CT)

Basic principle:
- Acquire many projections
- Different orientations around object
- Reconstruct cross-sectional image


== CT Generations

- 1st generation: translate–rotate, pencil beam
- 2nd generation: fan beam, detector array
- 3rd generation: rotating source and detectors
- 4th generation: stationary detector ring


== Image Formation in CT

Using attenuation model:

$I = I_0 exp(- integral mu(x) d x)$

Define projection value:

$p = -ln(I / I_0) = integral mu(x) d x$

Thus each projection is a line integral of $mu$.


== Parallel-Ray Geometry

Parameterization:

$x(s) = s cos(theta) - t sin(theta)$  
$y(s) = s sin(theta) + t cos(theta)$

Projection:

$g(t, theta) = integral mu(x(s), y(s)) d s$

This is the *Radon transform*.


== Sinogram

- $g(t, theta)$ plotted over $t$ and $theta$
- Each object point traces a sinusoid
- Sinogram contains all projection data


== Backprojection

Idea:
- Smear each projection back over the image

Backprojection operator:

$f_"BP"(x, y) = integral g(x cos(theta) + y sin(theta), theta) d theta$

Produces blurred image.


== Fourier Slice Theorem

1D Fourier transform of projection:

$G(omega, theta) = F_1[g(t, theta)]$

Equals slice of 2D Fourier transform of image:

[$ F_2(mu)(u, v) $]

with:

$u = omega cos(theta)$  
$ v = omega sin(theta)$


== Filtered Backprojection (FBP)

Steps:
1. Filter projections with high-pass filter
2. Backproject filtered projections

Reconstruction:
$mu(x, y) = integral (g * h)(x cos(theta) + y sin(theta), theta) d theta$

where $h$ is the reconstruction filter.


== Iterative Reconstruction

- Start with initial guess
- Forward project
- Compare to measured data
- Update estimate

(Details skipped in lecture)


== CT Artifacts

- *Aliasing*: insufficient number of projections
- *Beam hardening*: low-energy photons absorbed more strongly

Results in streaks and cupping artifacts.


== Hounsfield Units

To standardize CT values:

[$ H U = 1000 (mu - mu_"water") / (mu_"water" - mu_"air") $]

Reference values:

- Air: −1000
- Water: 0
- Fat: −120 to −90
- Muscle: +35 to +55
- Bone: +300 to +1900
- White matter: +20 to +30
- Grey matter: +37 to +45


== Summary

- CT reconstructs attenuation coefficients from projections
- Based on Radon transform and Fourier theory
- Filtered backprojection is classical reconstruction method
- Regularization and learning-based methods improve reconstruction

= Learned Reconstruction

= MRI

= Image Registration

= Image Segmentation

= Federated Learning

= Microscopy

Here's some example text. Notice how the section heading uses
elegant spaced small caps.

=== A Subsection

Subsections use italic text for a subtle hierarchy.

#definition(title: "Important Concept")[
  A definition block with a distinctive left border. Use this to
  define key terms in your work.
]

#theorem(title: "Main Result")[
  A theorem block for stating important results. The numbering
  is automatic.
]

#example(title: "Practical Application")[
  An example block with a subtle gray background. Use this to
  illustrate concepts with concrete examples.
]

#remark()[
  A remark block for additional observations or notes that don't
  fit the formal structure of theorems and definitions.
]


Inline code looks like `this`, and code blocks are formatted cleanly:

```python
def hello_world():
    """A simple function."""
    print("Hello, ClassicThesis!")
```

== Tables and Figures

#figure(
  table(
    columns: (auto, auto, auto),
    table.header([*Item*], [*Description*], [*Value*]),
    [Alpha], [First item], [100],
    [Beta], [Second item], [200],
    [Gamma], [Third item], [300],
  ),
  caption: [A sample table with clean styling.],
)

// ============================================================================
// Part II
// ============================================================================

#part("Advanced Topics")

= Another Chapter

Continue your document with more chapters. Each chapter starts
on a new page with the elegant ClassicThesis heading style.

== References and Citations

Add your bibliography and citations as needed.

= Conclusion

Wrap up your work with a conclusion chapter.
