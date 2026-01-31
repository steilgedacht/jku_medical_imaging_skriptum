
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

Another phenonom is precession: The magnetic momentum precesses around the external field. Not that these are not fully aligned with the magnetic field $B_0$

#figure(
  image("../assets/precession.png", height: 110pt),
)
#definition(title: "Larmor Frequency")[
  The frequency of precession is the Larmor frequency:
  $ omega_0 = gamma B_0 $
  For a proton ($""^1 H$), $gamma / (2 pi) approx 42.6$ MHz/T. This is a key equation for MR imaging.
]

=== Interaction with Radiofrequency field $B_1$


When an Radiofrequency field $B_1$ is applied at the Larmor frequency, it tips the magnetization away from the longitudinal axis. The $B_1$ field is so normal to the $B_0$ and the special thing is also that it rotates around the axis direction of $B_0$ with the angular speed of the Larmor Frequency. The resulting change of M can then be described by $ (d M) / (d t) = gamma (M times B_1) $

#figure(
  image("../assets/rfb1.png", height: 150pt),
)

After we set that up we can then turn of the $B_1$ so that the particles can relax again. As all of the particles spin now still for a bit with this precession from the $B_1$ field in a sychronized manner, they induce a current in the coil that produced the $B_1$ field due to the changing magnetic field from them which is called magnetic flux.

$ u = - (partial psi ) / (partial t) $

In MRI we get now different images from one scan, that are the different contrasts:
#figure(
  image("../assets/mri_scans.png", height: 150pt),
)


#definition(title: "Longitudinal Relaxation (T1)")[
  When we start again with just a $B_0$ field, then we can switch on the $B_1$ field and we make it so strong, that we have a flip angle of 90 degrees. So we just have XY magnetization anymore. Then we turn off the $B_1$ again and measure the time how long it take until the magnetization relaxes again in the $B_0$ direction. This is what this formula describes and the curve that we can see in the image: 
  $ M_z = M_0 (1 - e^(-t / T_1)) $
#figure(
  image("../assets/T1.png", height: 150pt),
)
We can then just measure which $M_z$ angle it reached again after the time $T_1$.  

]

#definition(title: "Transversal Relaxation (T2)")[
  We start with a Transversal $B_1$ field again and without a $B_1$ field. Then we turn the $B_1$ field off and measure the behaviour of the magnetic field which follows this line:
  $ M_(x y) = M_0 e^(-t / T_2) $
  #figure(
    image("../assets/T2.png", height: 150pt),
  )

]

=== Contrast Information
Different tissue types now have different $T_1$ and $T_2$ times. We measure not the z component, we measure the Transversal component.

#figure(
  image("../assets/repetiton_time.png", height: 150pt),
)
#figure(
  image("../assets/t2.png", height: 150pt),
)
By tailoring the Repetition Time (TR) and Echo Time (TE), we can choose the most suitable contrast to differentiate structures:
- T1-weighted: Short TR, short TE.
- T2-weighted: Long TR, long TE ($M_(x y) = M_0 e^(-t / T_2)$).
- Proton Density (PD) weighted: Long TR, short TE ($M_z = M_0(1 - e^(-t / T_1))$).

== How to get now spatial information?

To get an image, spatial information must be encoded using gradient fields $arrow(G)$. For that we need 3 more coils that introduce that gradient field. The local Larmor frequency becomes position-dependent, the particles rotate at different locations with different frequencies:

- X-gradient: $omega(x) = omega_0 + gamma G_x x$
- Y-gradient: $omega(y) = omega_0 + gamma G_y y$
- Z-gradient: $omega(z) = omega_0 + gamma G_z z$

This allows for slice selection (z-axis) and frequency/phase encoding (x and y axes) to fill the k-space, which is then transformed into an image via a 2D Fourier Transform.

#figure(
  image("../assets/gradient_field.png", height: 150pt),
)

#figure(
  image("../assets/pulse.png", height: 150pt),
)

So with a MRI device we are measuring Fourier coefficients. The formula for that is $ s(k_x, k_y) = integral_x integral_y m(x, y) e^(-i k_x x) e^(-i k_y y) d x d y $

So in the end we send that pulse via the Radiofrequency, then we choose a $G_z$ plane. Then we choose a $G_y$ and a $G_x$ and so to say measure line for line all the information of the Fourier domain. For every point you have to wait for around 6 seconds. And that is why we dont want to measure the whole Fourier Space, but rather interpolate or do some postprocessing. 
#figure(
  image("../assets/rastering.png", height: 150pt),
)
And because of the Fourier, MRI is also a linear inverse problem.

