#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Image Segmentation

== What is Image Segmentation?


Image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects).

- Goal: Assign a semantic label to every pixel (2D) or voxel (3D) in an image.
- Output: A dense label map.

#example(title: "Medical Applications")[
  - Tumor delineation: Identifying the boundaries of a tumor in MRI or CT scans.
  - Organ segmentation: Separating liver, lungs, or heart from surrounding tissue.
  - Lesion burden estimation: Quantifying the total area or volume affected by a disease.
]

#figure(
  image("../assets/segmentation1.png", height: 150pt),
)

#figure(
  image("../assets/segmentation2.png", height: 150pt),
)
== Segmentation vs. Other Tasks


It is important to distinguish segmentation from other computer vision tasks:
- Image Classification: Assigning a single label to the entire image (e.g., "glioma" vs "no tumor").
- Object Detection: Identifying objects and drawing bounding boxes around them.
- Segmentation: Identifying the exact shape of the object at the pixel level.

#figure(
  image("../assets/segmentation3.png", height: 150pt),
)
== Clinical Significance


- Quantitative Analysis: Allows for precise measurement of volume, shape, and thickness.
- Treatment Planning: Essential for radiotherapy (delineating the target volume) and surgery.
- Longitudinal Follow-up: Comparing scans over time to check for progression or stability using criteria like RANO.
- Sensitivity: In medicine, small boundary errors can have a massive clinical impact.


== Mathematical Formulation


#definition(title: "Segmentation as a Labeling Function")[

  Let $Omega subset RR^d$ be the image domain ($d in {2, 3, ... }$). An image $x$ is a function $x: Omega arrow.r RR^C$ (color space). The segmentation problem is defined as finding a mapping:
  $ S: Omega arrow.r cal(L) $
  that assigns to every location a label and where $cal(L) = \{0, 1, dots, K\}$ is the set of labels ($0$ is usually the background).
]

  The labeling induces a partitioning of the domain into $Omega_k = \{i in Omega | S(i) = k\}$.
  These partitions must be:
  - Non-overlapping: $Omega_k inter Omega_l = emptyset$ for $k,l in cal(L) and k eq.not l$.
  - Complete: $union.big_(l in cal(L)) Omega_l = Omega$.
That basically translates to "I assign every pixel a label and only one".

== Types of Segmentation

- Binary: Separating a single foreground structure from the background ($K=1$).
- Multi-class: Segmenting multiple anatomical structures ($K > 1$).
- Semantic: All pixels of the same class (e.g., all cells) share one label.
- Instance: Separating individual objects (e.g., each individual cell gets a unique ID).
- Panoptic: Combines semantic and instance segmentation.

#figure(
  image("../assets/segmentation4.png", height: 100pt),
  caption: "Binary vs Multi-class segmentation"
)
#figure(
  image("../assets/segmentation5.png", height: 100pt),
)
== Classical Segmentation Methods


1. Thresholding: 
   - Global: One value for the entire image (e.g., Otsu's method).
   - Local: Adaptive thresholds based on local neighborhoods.
2. Region-based:
   - Region Growing: Starts with seed points and expands to similar neighbors.
   - Watershed: Interprets the image as a topographic map and "floods" it from local minima.
3. Graph Cuts:
   - Represents the image as a graph where pixels are nodes.
   - Minimizes an energy function $E(x)$ consisting of unary (likelihood) and pairwise (smoothness) costs.
   - Solved using min-cut/max-flow algorithms.

=== Graph Cuts
Let the image be represented as a graph $G(V, E)$.
Assume a binary segmentation task, i.e., $L = {0, 1}$.

A graph cut is the set of edges whose removal makes a graph disconnected. 
The cost of the cut is the sum of its edge weights.
The energy $E(x)$ is defined as:

$ E(x) = underbrace(sum_(p in V) x_p F_p + (1 - x_p) B_p, "unary term") + underbrace(lambda sum_(p,q in E) W_(p q) |x_p - x_q|, "pairwise term") $

Where:
- $x_p = cases(0 &"if background at pixel " p, 1 &"if foreground at pixel " p)$
- $F_p$: Cost of assigning foreground
- $B_p$: Cost of assigning background
- $W_(p q)$: Similarity of $p$ and $q$ (typically $e^(-alpha ||I_p - I_q||^2)$)
- $lambda$: Regularization weight

Algorithm Steps: 
1. Build graph $G$ with one node per pixel.
2. Add source/sink edges for unary costs (back- and foreground).
3. Add neighborhood edges with weights $W_{p q}$ (reflecting similarity of nodes $p$ and $q$).
4. Compute minimum $s-t$ cut (Boykov-Kolmogorov Algorithm).
5. Assign labels from cut.


=== Relation to Discrete TV

Note that the pairwise term can be written as:
$ sum_(p,q in E) W_(p q) |x_p - x_q| = ||nabla x||_W $
which is the discrete anisotropic TV.

Then, we can also simplify the unary term:
$ sum_(p in V) x_p F_p + (1 - x_p) B_p = sum_(p in V) B_p + sum_(p in V) x_p (F_p - B_p) = chevron.l 1, B chevron.r + chevron.l x, F - B chevron.r $

We get the following minimization problem:
$ min_(x in {0, 1}) chevron.l x, F - B chevron.r + lambda ||nabla x||_W $
#figure(
  image("../assets/threshold.png", height: 250pt),
  caption: "Global Thresholding vs Local Thresholding"
)

#figure(
  image("../assets/watershed.png", height: 120pt),
  caption: "Watershed Algorythm"
)
== Deep Learning for Segmentation


=== U-Net
The U-Net is the gold standard for medical image segmentation.
- Architecture: Symmetric encoder (contracting path) and decoder (expansive path).
- Skip Connections: Concatenate high-resolution features from the encoder to the decoder to preserve spatial detail.
- The Receptive Field gets exponentially bigger with each deeper layer of the encoder. This is important as the Receptive Field should be at least as large as the things we want to segment.
- Training Objective: Usually Weighted Cross Entropy ($C E$).

#figure(
  image("../assets/unet.png", height: 160pt),
  caption: "U-Net"
)

=== V-Net
Designed for volumetric (3D) medical images.
- Objective: Uses the Dice Loss to handle class imbalance (e.g., when the tumor is much smaller than the background).

$ D(theta) = 1 - (2 sum_(i=1)^I hat(p)_i (theta) s_i) / (sum_(i=1)^I hat(p)_i ( theta) + sum_(i=1)^I s_i) = 1 - underbrace(2 (chevron.l hat(p) (theta) , s chevron.r) / ( chevron.l hat(p) (theta), 1 chevron.r  + chevron.l s, 1 chevron.r), "Dice Coefficient") $
#figure(
  image("../assets/vnet.png", height: 190pt),
  caption: "V-Net"
)

=== Comparison of CE & Dice Loss

$ "CE" = - sum_i w_i [s_i log hat(p)_i + (1 - s_i) log (1 - hat(p)_i)] $

$ D = 1 - 2 (chevron.l hat(p) (theta) , s chevron.r) / ( chevron.l hat(p) (theta), 1 chevron.r  + chevron.l s, 1 chevron.r) 
    = 1 - 2 frac(A, B) $


Derivatives

$ frac(partial "CE", partial theta) &= - sum_i w_i [ frac(partial hat(p)_i, partial theta) frac(s_i, hat(p)_i) - frac(partial hat(p)_i, partial theta) frac(1 - s_i, 1 - hat(p)_i) ] \
&= - sum_i w_i frac(partial hat(p)_i, partial theta) frac(1, hat(p)_i (1 - hat(p)_i)) [(1 - hat(p)_i) s_i - hat(p)_i (1 - s_i)] \
&= - sum_i w_i frac(partial hat(p)_i, partial theta) frac(1, hat(p)_i (1 - hat(p)_i)) [s_i - cancel(hat(p)_i s_i) - hat(p)_i + cancel(hat(p)_i s_i)] \
&= - sum_i w_i frac(partial hat(p)_i, partial theta) frac(1, hat(p)_i (1 - hat(p)_i)) [s_i - hat(p)_i] $


$ frac(partial D, partial theta) &= 0 - frac(partial hat(p)_i, partial theta) 2 frac(S dot B - A dot 1, B^2) \
&= - frac(partial hat(p)_i, partial theta) dot 2 frac(s B - A 1, B^2) $


When we compare the gradients:
- local vs global information
- stable & no vanishing gradient for CE (in logits). This does not hold for the Dice
- CE has a strong signal initially during training

=== Advanced Architectures
nnU-Net: A "self-configuring" method that automatically adapts the U-Net architecture and hyperparameters to a specific dataset. It is thought as a Out of the box experience.

#figure(
  image("../assets/nnUnet.png", height: 190pt),
)
UNETR: Uses Transformers as the encoder to capture long-range dependencies, paired with a U-shaped decoder.

#figure(
  image("../assets/Unettransformer.png", height: 190pt),
)
Segment Anything Model (SAM): A promptable foundation model for segmentation, recently adapted for medical images (SAM-Med).

#figure(
  image("../assets/sam.png", height: 150pt),
)
#figure(
  image("../assets/sam2.png", height: 150pt),
)

== Segmentation Loss Odyssey



=== Distribution-based: 
  - *Weighted Cross Entropy*: Penalizes errors in rare classes more heavily.
  $ L_(C E)(theta) = - 1 / I sum_(i=1)^I sum_(k in cal(L)) w_k^i s_k^i log hat(p)_k^i (theta) $

  - *TopK*
  $ L_(T o p K)(theta) = - 1 / (sum_(i=1)^I sum_(k in cal(L)) bb(1) {s_k^i = k " and " hat(p)_k^i < t}) sum_(i=1)^I sum_(k in cal(L)) bb(1) {s_k^i = k " and " hat(p)_k^i < t} log hat(p)_k^i (theta) $
  - *Focal Loss*: Focuses on hard-to-classify pixels by down-weighting easy ones.
$ L_"Focal"(theta) = - 1/I sum_(i=1)^I sum_(k in cal(L)) (1 - hat(p)_k^i (theta))^gamma s_k^i log hat(p)_k^i (theta) $
=== Region-based: 
  - *Dice Loss*: Measure the overlap between prediction and ground truth.
  $ L_D (theta) = 1 - 2 frac(sum_(k in cal(L)) w_k sum_(i=1)^I hat(p)_i^k (theta) s_i^k, sum_(k in cal(L)) w_k sum_(i=1)^I (hat(p)_i^k (theta) + s_i^k)) $
  where $w_k = frac(1, (sum_(i=1)^I s_i^k)^2)$.

  - *Intersection over Union* (IoU)
  $ L_D (theta) = 1 - frac(sum_(k in cal(L)) sum_(i=1)^I hat(p)_i^k (theta) s_i^k, sum_(k in cal(L)) w_k sum_(i=1)^I (hat(p)_i^k (theta) + s_i^k - hat(p)_i^k (theta) s_i^k)) $

  - *Tversky Loss*: Generalization of Dice that allows controlling the trade-off between False Positives and False Negatives.

  $ L_D (theta) = frac(sum_(k in cal(L)) sum_(i=1)^I hat(p)_i^k (theta) s_i^k, sum_(k in cal(L)) sum_(i=1)^I hat(p)_i^k (theta) s_i^k + alpha sum_(k in cal(L)) sum_(i=1)^I hat(p)_i^k (theta) (1 - s_i^k) + beta sum_(k in cal(L)) sum_(i=1)^I (1 - hat(p)_i^k (theta)) s_i^k) $
=== Boundary-based:
  - *Hausdorff Distance (HD)*: Penalizes the distance between the boundaries of the predicted and ground truth masks.
  $ L_(H D - D T) (theta) = 1/I sum_(i=1)^I (s_i - hat(p)_k^i (theta)) dot (d_(S_i)^2 + d_(P_i)^2) $

  where $d_S$ and $d_P$ are the distance transforms of ground truth and segmentation.

  *Note*: this formula only approximated the Hausdorff Distance

#figure(
  image("../assets/loss_odyssey.png", height: 150pt),
)

#rect(
  fill: gray.lighten(80%),
  stroke: 1pt + black,
  inset: 12pt,
  width: 100%,
  align(center)[
    A *combination* of different loss terms is used in most cases. Typically, CE and DSC loss are combined.
  ]
)
== Evaluation

Checkout #link("https://metrics-reloaded.dkfz.de/")[Metrics Reloaded]

