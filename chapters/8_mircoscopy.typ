
#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Microscopy


== Why Microscopy matters in Medicine?


Microscopy reveals structure and function at the cellular and tissue level, which is critical for diagnosis, research, and therapy decisions.

*Key medical applications:*
- **Histopathology**: For example, cancer diagnosis through tissue examination.
- **Hematology**: Analysis of blood smears.
- **Infectious disease identification**: Detecting pathogens.
- **Cell biology & drug discovery**: Understanding cellular mechanisms.

#remark()[
  *Handwritten Workflow:*
  1. Endoscopic Biopsy $arrow$ 2. Gross Examination $arrow$ 3. Tissue Fixation/Embedding $arrow$ 4. Microtomy/Staining $arrow$ 5. Microscopic Evaluation.
]

== Why Machine Learning?


Traditional manual microscopy analysis is:
- **Time-intensive**: Pathologists must manually scan large slides.
- **Subjective**: High variability between different practitioners.
- **Hard to scale**: Difficult to handle large datasets of high-resolution slides.

*Machine Learning (ML) Advantages:*
- Automates repetitive tasks.
- Delivers quantitative measures (e.g., cell counts, morphology).
- Enables pattern discovery beyond human perception.


== Microscopy Modalities Overview


#figure(
  table(
    columns: (auto, auto, auto, auto),
    table.header([*Modality*], [*Contrast Mechanism*], [*Advantages*], [*Limitations*]),
    [Brightfield], [Absorption by stains (H&E)], [Cheap, clinical standard], [Requires staining],
    [Phase Contrast], [Phase shifts (refractive index)], [Live cell imaging (no stain)], [Low molecular specificity],
    [Fluorescence], [Fluorophore emission], [High specificity, multi-channel], [Photobleaching, blur],
    [Confocal], [Pinhole rejection], [3D optical sectioning], [Slower, phototoxicity],
    [Electron (TEM)], [Electron scattering], [Extremely high res (< 1 nm)], [Expensive, destructive],
  ),
  caption: [Comparison of common microscopy modalities.],
)


=== Brightfield Microscopy

White light passes through the sample, and the image is based on absorption by stains. This is the most used method in standard histology.

#definition(title: "Staining")[
  Biological tissues are largely transparent. Stains (like **Hematoxylin & Eosin / H&E**) bind selectively to cellular components (e.g., nuclei vs. cytoplasm) to convert biochemical differences into visible intensity differences.
]


=== Other Modalities
- **Fluorescence Microscopy**: Uses fluorophores that absorb excitation light and emit light at a longer wavelength.
- **Confocal Microscopy**: A laser scanning technique using a pinhole to reject out-of-focus light, allowing for 3D "optical sectioning".
- **Electron Microscopy**: Uses electrons instead of photons for resolution up to 1,000,000x. Includes **TEM** (internal structure) and **SEM** (surface topology).

== Key Challenges in Medical Imaging


1. **Data**: Expert annotations are expensive and time-consuming (pathologists spend hours per slide).
2. **Whole Slide Images (WSI)**: Images can be massive (e.g., $100,000 times 100,000$ pixels, ~10GB per image).
3. **Class Imbalance**: Tasks often involve "rare events" like mitoses.
4. **Domain Shifts**: Variations in scanner types, staining protocols, and patient populations.

== Multiple Instance Learning (MIL)


Due to the size of WSIs and the lack of pixel-level labels, we often use **Weakly Supervised Learning** through MIL.

#definition(title: "Multiple Instance Learning (MIL)")[
  Instead of individual labeled samples, we have **bags** of instances $X_j = {x_{j 1}, x_{j 2}, dots, x_{j K}\}$.
  - A bag is labeled $Y=0$ if all instances are negative.
  - A bag is labeled $Y=1$ if **at least one** instance is positive.
]

#theorem(title: "Permutation Invariance")[
  A MIL scoring function $S(X)$ must be symmetric (invariant to the order of instances). It can be decomposed as:
  $ S(X) = g(sum_{x \in X} f(x)) $
]

=== Deep MIL Approaches
1. **Instance-level approach**: $f$ is an instance classifier; scores are aggregated.
2. **Embedding-level approach**: $f$ maps instances to low-dimensional embeddings, which are then pooled to create a bag representation for the classifier $g$.

=== Attention-based MIL Pooling

The bag representation $z$ is computed as a weighted sum of instance embeddings $h_k$:
$ z = sum_{k=1}^K a_k h_k $
Where the attention weights $a_k$ are:
$ a_k = (exp(w^T tanh(V h_k))) / (sum_{j=1}^K exp(w^T tanh(V h_j))) $
*Gated variant*: $a_k prop exp(w^T (tanh(V h_k) dot.o sigma(U h_k)))$.
