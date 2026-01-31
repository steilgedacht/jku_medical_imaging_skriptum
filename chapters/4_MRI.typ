
#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Magnetic Resonance Imaging

== From Spin to Magnetic Resonance Imaging

The study of MRI often begins from a classical physics viewpoint, where we accept the existence of nuclear spin without diving into the full quantum mechanics motivation. 

== Nuclear Spin, Magnetic Dipole Moment, and Torque

#figure(
  image("../assets/spin.png", height: 110pt),
)

A rotating object with mass $m$ leads to angular momentum:
$ arrow(L) = arrow(r) times (m arrow(v))$.
The spin of a proton  leads to magnetic angular momentum $arrow(I)$. It is modeled as a magnetic dipole with a moment $arrow(mu) = gamma arrow(I)$, where $gamma$ is the gyromagnetic ratio.

Gyromagnetic Ratios
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: (left, left),
  [*Element*], [*Gyromagnetic ratio $gamma$ (MHz/T)*],
  [$""^1 H$], [42.58],
  [$""^3 "He"$], [32.43],
  [$""^23 "Na"$], [11.26],
  [$""^31 P$], [17.24],
)

Exposure to an external magnetic field $arrow(B)_0$ leads to a torque $arrow(tau)$ that attempts to align the magnetic moment $arrow(mu)$:
$ arrow(tau) = arrow(mu) times arrow(B)_0 $


Thermal Motion: In the absence of a field, random orientation means no magnetization (humans are not inherently magnetic).
 In the presence of a field $B_0$, thermal motion is still present, but the magnetic moments align enough to create a small bulk magnetization $arrow(M) = sum_i arrow(mu)_i$ with magnitude   $ M = (rho gamma^2 planck B_0) / (4 k T) $
This alignment is the first effect we will later use for MRI.

#figure(
  image("../assets/MRI_B0.png", height: 110pt),
)

Another phenonom is precession: The magnetic momentum precesses around the external field

#figure(
  image("../assets/precession.png", height: 110pt),
)
#definition(title: "Larmor Frequency")[
  The frequency of precession is the Larmor frequency:
  $ omega_0 = gamma B_0 $
  For a proton ($""^1 H$), $gamma / (2 pi) approx 42.6$ MHz/T. This is a key equation for MR imaging.
]

=== Interaction with Radiofrequency field $B_1$


When an RF field $B_1$ is applied at the Larmor frequency, it tips the magnetization away from the longitudinal axis.
- Bloch Equation (simplified): $(d M) / (d t) = gamma (M times B_1)$.
- Flip Angle: $alpha = gamma B_1 t$.

The resulting magnetization has two components:
- Longitudinal component: Parallel to $B_0$.
- Transversal component: Perpendicular to $B_0$, which induces a current in the receiver coil (signal reception).

== Relaxation and Contrast


#definition(title: "Longitudinal Relaxation (T1)")[
  Recovery of the $M_z$ component after an RF pulse:
  $ M_z = M_0 (1 - e^(-t / T_1)) $
]

#definition(title: "Transversal Relaxation (T2)")[
  Decay of the $M_(x y)$ component:
  $ M_(x y) = M_0 e^(-t / T_2) $
]

=== Contrast Information
By tailoring the Repetition Time (TR) and Echo Time (TE), we can choose the most suitable contrast to differentiate structures:
- T1-weighted: Short TR, short TE.
- T2-weighted: Long TR, long TE ($M_(x y) = M_0 e^(-t / T_2)$).
- Proton Density (PD) weighted: Long TR, short TE ($M_z = M_0(1 - e^(-t / T_1))$).

== Image Encoding (Gradients)


To get an image, spatial information must be encoded using gradient fields $arrow(G)$. The local Larmor frequency becomes position-dependent:

- X-gradient: $omega(x) = omega_0 + gamma G_x x$
- Y-gradient: $omega(y) = omega_0 + gamma G_y y$
- Z-gradient: $omega(z) = omega_0 + gamma G_z z$

This allows for slice selection (z-axis) and frequency/phase encoding (x and y axes) to fill the k-space, which is then transformed into an image via a 2D Fourier Transform.
