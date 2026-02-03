
#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Microscopy


== Why Microscopy matters in Medicine?


Microscopy reveals structure and function at the cellular and tissue level, which is critical for diagnosis, research, and therapy decisions.

*Key medical applications:*
- Histopathology: For example, cancer diagnosis through tissue examination.
- Hematology: Analysis of blood smears.
- Infectious disease identification: Detecting pathogens.
- Cell biology & drug discovery: Understanding cellular mechanisms.

#figure(
  image("../assets/microscopy.png", height: 260pt),
 )

== Why Machine Learning?


Traditional manual microscopy analysis is:
- Time-intensive: Pathologists must manually scan large slides.
- Subjective: High variability between different practitioners.
- Hard to scale: Difficult to handle large datasets of high-resolution slides.

*Machine Learning (ML) Advantages:*
- Automates repetitive tasks.
- Delivers quantitative measures (e.g., cell counts, morphology).
- Enables pattern discovery beyond human perception.


== Microscopy Modalities Overview
#block()[
  #set text(size: 6pt)
  #table(
    columns: (auto, 1.5fr, 1.2fr, 1.3fr, 1.2fr, 1.2fr, 1fr),
    inset: 7pt,
    align: horizon,
    stroke: 0.5pt + gray,
    
    [*Modality*], [*Contrast Mechanism*], [*Typical Data Type*], [*Main Applications*], [*Advantages*], [*Limitations*], [*Typical Spatial Resolution*],
    
    [*Brightfield Microscopy*], 
    [Absorption of light by stains (e.g., H&E, IHC)], 
    [2D images, Whole Slide Images (WSI)], 
    [Histopathology, cancer diagnosis, tissue morphology], 
    [Cheap, standardized, clinically established], 
    [Requires staining, limited molecular specificity], 
    [~0.2–0.5 μm (light diffraction limit)],

    [*Phase Contrast Microscopy*], 
    [Optical phase shifts from refractive index differences], 
    [2D time-lapse images], 
    [Live cell imaging, cell motility, cell division], 
    [No staining, suitable for living cells], 
    [Limited molecular specificity], 
    [~0.2–0.5 μm],

    [*Fluorescence Microscopy (Widefield)*], 
    [Fluorophore excitation/emission], 
    [2D multi-channel images], 
    [Protein localization, biomarker detection], 
    [Molecular specificity, multichannel], 
    [Out-of-focus blur, photobleaching], 
    [~0.2–0.3 μm (lateral), ~0.5–0.7 μm (axial)],

    [*Confocal Microscopy*], 
    [Optical sectioning via pinhole rejection], 
    [2D slices, 3D stacks], 
    [3D tissue/cell imaging, morphology], 
    [High contrast, true 3D imaging], 
    [Slower, phototoxicity], 
    [~0.18–0.25 μm (lateral), ~0.5 μm (axial)],

    [*Electron Microscopy (TEM)*], 
    [Electron transmission and scattering], 
    [2D grayscale images], 
    [Subcellular ultrastructure, organelles], 
    [Extremely high resolution], 
    [Expensive, destructive, grayscale], 
    [ < 1 nm (≈0.1–0.5 nm)],

    [*Electron Microscopy (SEM)*], 
    [Electron surface scattering], 
    [2D surface topology], 
    [Cell surfaces, materials, morphology], 
    [3D-like surface detail], 
    [Limited internal structure], 
    [1–10 nm],
  )
]

=== Brightfield Microscopy

White light passes through the sample, and the image is based on absorption by stains. This is the most used method in standard histology.

#figure(
  image("../assets/brightfield_microscopy2.png", height: 260pt),
 )
#figure(
  image("../assets/brightfield_microscopy.png", height: 260pt),
 )

#definition(title: "Staining")[
  Biological tissues are largely transparent. Stains (like Hematoxylin & Eosin / H&E) bind selectively to cellular components (e.g., nuclei vs. cytoplasm) to convert biochemical differences into visible intensity differences.
]
#figure(
  image("../assets/staining.png", height: 200pt),
 )

=== Fluorescence microscopy
Uses fluorophores that absorb excitation light and emit light at a longer wavelength. We can prepare the sample with dye by fluroscent stain or use fluorescent proteins. With that we can get multiple images with different channels that all come from different stains. With that we can get highly specific imaging.

#figure(
  image("../assets/fluroscene.png", height: 200pt),
 )
#figure(
  image("../assets/fluroscene2.png", height: 100pt),
 )

=== Phase Contrast
Here we use the refrection index of the materials which accounts for different phase shifts 
#figure(
  image("../assets/phase_contrast.png", height: 150pt),
 )

 #figure(
   image("../assets/phase_contrast2.png", height: 150pt),
 )

With this you can also investigate still living entities.

=== Confocal Microscopy

A technique using a pinhole to reject out-of-focus light, allowing for 3D "optical sectioning". Only the things that are on the same height appear in-focus. 

 #figure(
   image("../assets/confocal.png", height: 150pt),
 )
 #figure(
   image("../assets/confocal2.png", height: 150pt),
 )


=== Electron microscopy

Uses electrons instead of photons for resolution up to 1,000,000x (optical microscopy up to 1,500x). It acchievs that by using electromagnetic lenses instead of glass. The sample preparation requires lot of work, no live imaging possible. The sample must be within vacuum. There are two types:
- Transmission EM: internal ultrastructure
- Sampling EM: surface morphology and topology

#figure(
  image("../assets/electron_microscopy.png", height: 190pt),
)

#figure(
  image("../assets/tem_vs_sem.png", height: 190pt),
)

== Key Challenges in Medical Imaging

+ Limited labeled data
  Expert annotations require
  - Pathologists
  - Biologists
  - Hours per slide
+ Class imbalance \
  Many tasks involve rare events (Mitoses, …)
+ Whole slide images (WSI) can be
  - 100,000 x 100,000 pixels
  - 10GB per image
  - Multiple channels (fluorescence)
+ Domain shifts
  - Scanner types
  - Straining protocol
  - population
+ Explainability
  - Interpretable predictions
  - Visual explanations

=== Image Level Challenges

#figure(
  image("../assets/image_level_challenges.png", height: 190pt),
)

=== Annotation Challenges: Weak labels
Truth labels are less accurate than required
- Case-level annotation
- Slide-level annotation
- Region-level annotation
- Tile-level annotation
- (Pixel-level annotation)

=== From WSI to Tiles

On the size of the tiles which are usually 224x224 the image classification happens. 

#figure(
  image("../assets/wsi.png", height: 190pt),
)


== Multiple Instance Learning (MIL)


Due to the size of WSIs and the lack of pixel-level labels, we often use Weakly Supervised Learning through MIL.

#definition(title: "Multiple Instance Learning (MIL)")[
  Instead of individual labeled samples, we have bags of instances/samples $X_j = {x_{j 1}, x_{j 2}, dots, x_{j K}\}$.
  - A bag is labeled $Y=0$ if all instances are negative.
  - A bag is labeled $Y=1$ if at least one instance is positive.
]

#theorem(title: "Permutation Invariance")[
  A MIL scoring function $S(X)$ must be symmetric (invariant to the order of instances). It can be decomposed as:
  $ S(X) = g(sum_(x in X) f(x)) $
  where $f$ and $g$ are sutiable transformations.
]
#theorem(title: "")[
For any $epsilon > 0$, a Hausdorff continuous symmetric $S: bb(R)^(K times D) arrow bb(R)$ can be arbitrarily approximated by

$ |S(bold(X)) - g(max_(bold(x) in bold(X)) f(bold(x)))| < epsilon. $
]
=== Deep MIL Approaches
+ Transform each instance $bold(x)$ using the transformation $f_theta$
+ Combine transformed instances using a symmetric (permutation-invariant) function $sigma$ (also called MIL *pooling* function)
+ Transform combined instances using a function $g_psi$ to obtain

Two main approaches:
- *Instance-level approach*\
  The transformation $f$ is an instance-level classifier. The instance-level scores are aggregated by MIL pooling $sigma$. The function $g$ is the identity.
- *Embedding-level approach*\
  The transformation $f$ maps instances to low-dimensional embeddings. MIL pooling $sigma$ used to obtain bag representation. $g$ becomes a bag-level classifier.


=== Attention-based MIL pooling
$ sigma(bold(h)_1, ..., bold(h)_K) = sum_(k=1)^K a_k bold(h)_k, $
where $bold(h)_k = f(bold(x)_k) in RR^M$ and the attention weights are computed as
$ a_k = (exp(w^top tanh(V bold(h)_k))) / (sum_(j=1)^K exp(w^top tanh(V bold(h)_j))) $
$w in RR^(L times 1)$ and $V in RR^(L times M)$ are learnable weight.

=== Gated attention variant
$ a_k = (exp(w^top (tanh(V bold(h)_k) dot.o sigma(U bold(h)_k)))) / (sum_(j=1)^K exp(w^top (tanh(V bold(h)_j) dot.o sigma(U bold(h)_j)))) $
where $U in RR^(L times M)$ is also learnable.

#figure(
  image("../assets/deep_multiple.png", height: 190pt),
)
