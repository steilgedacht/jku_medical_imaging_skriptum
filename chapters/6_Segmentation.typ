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


== Segmentation vs. Other Tasks


It is important to distinguish segmentation from other computer vision tasks:
- Image Classification: Assigning a single label to the entire image (e.g., "glioma" vs "no tumor").
- Object Detection: Identifying objects and drawing bounding boxes around them.
- Segmentation: Identifying the exact shape of the object at the pixel level.

== Clinical Significance


- Quantitative Analysis: Allows for precise measurement of volume, shape, and thickness.
- Treatment Planning: Essential for radiotherapy (delineating the target volume) and surgery.
- Longitudinal Follow-up: Comparing scans over time to check for progression or stability using criteria like RANO.
- Sensitivity: In medicine, small boundary errors can have a massive clinical impact.

== Mathematical Formulation


#definition(title: "Segmentation as a Labeling Function")[
  Let $Omega subset RR^d$ be the image domain ($d=2, 3$). An image $x$ is a function $x: Omega arrow.r cal(C)$ (color space).
  
  The segmentation problem is defined as finding a mapping:
  $ S: Omega arrow.r cal(L) $
  Where $cal(L) = \{0, 1, dots, K\}$ is the set of labels ($0$ is usually the background).
]

#remark()[
  *Handwritten Note on Partitioning:*
  The labeling induces a partitioning of the domain into $Omega_k = \{i in Omega mid S(i) = k\}$.
  These partitions must be:
  - Non-overlapping: $Omega_k inter Omega_l = emptyset$ for $k eq.not l$.
  - Complete: $union.big_{l in cal(L)} Omega_l = Omega$.
]

== Types of Segmentation


- Binary: Separating a single foreground structure from the background ($K=1$).
- Multi-class: Segmenting multiple anatomical structures ($K > 1$).
- Semantic: All pixels of the same class (e.g., all cells) share one label.
- Instance: Separating individual objects (e.g., each individual cell gets a unique ID).
- Panoptic: Combines semantic and instance segmentation.

== Classical Segmentation Methods


1. Thresholding: 
   - *Global*: One value for the entire image (e.g., Otsu's method).
   - *Local*: Adaptive thresholds based on local neighborhoods.
2. Region-based:
   - *Region Growing*: Starts with seed points and expands to similar neighbors.
   - *Watershed*: Interprets the image as a topographic map and "floods" it from local minima.
3. Graph Cuts:
   - Represents the image as a graph where pixels are nodes.
   - Minimizes an energy function $E(x)$ consisting of unary (likelihood) and pairwise (smoothness) costs.
   - Solved using min-cut/max-flow algorithms.

#remark()[
  *Handwritten relation*: The pairwise term in Graph Cuts is related to anisotropic Total Variation (TV).
]

== Deep Learning for Segmentation


=== U-Net
The U-Net is the gold standard for medical image segmentation.
- Architecture: Symmetric encoder (contracting path) and decoder (expansive path).
- Skip Connections: Concatenate high-resolution features from the encoder to the decoder to preserve spatial detail.
- Training Objective: Usually Weighted Cross Entropy ($C E$).


=== V-Net
Designed for volumetric (3D) medical images.
- Objective: Uses the Dice Loss to handle class imbalance (e.g., when the tumor is much smaller than the background).

#theorem(title: "Dice Loss (Binary)")[
  $ D(theta) = 1 - (2 sum_{i=1}^I hat{p}_i s_i) / (sum_{i=1}^I hat{p}_i + sum_{i=1}^I s_i) $
]

=== Advanced Architectures
- nnU-Net: A "self-configuring" method that automatically adapts the U-Net architecture and hyperparameters to a specific dataset.
- UNETR: Uses Transformers as the encoder to capture long-range dependencies, paired with a U-shaped decoder.
- Segment Anything Model (SAM): A promptable foundation model for segmentation, recently adapted for medical images (SAM-Med).

== Segmentation Loss Odyssey


A combination of different loss terms is often used in practice.

- Distribution-based: 
  - *Weighted Cross Entropy*: Penalizes errors in rare classes more heavily.
  - *Focal Loss*: Focuses on hard-to-classify pixels by down-weighting easy ones.
- Region-based: 
  - *Dice Loss* and *IoU (Jaccard)*: Measure the overlap between prediction and ground truth.
  - *Tversky Loss*: Generalization of Dice that allows controlling the trade-off between False Positives and False Negatives.
- Boundary-based:
  - *Hausdorff Distance (HD)*: Penalizes the distance between the boundaries of the predicted and ground truth masks.

== Evaluation


Validation should follow the Metrics Reloaded recommendations to ensure results are clinically meaningful and statistically sound.

