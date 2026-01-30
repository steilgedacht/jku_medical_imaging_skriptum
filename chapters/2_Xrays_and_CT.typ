#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= X-rays and Computed Tomography


== Discovery of X-rays

In 1895, Wilhelm Röntgen noticed "rays of mysterious origin", which he called X-rays. Within a month (22.12.1895), the first radiograph of the hand of Röntgen's wife was made in Würzburg.
This immediate application to imagine the human body marks the birth of medical imaging.

== Nature and Properties of X-rays

X-rays are electromagnetic waves.
They are a form of ionizing radiation—radiation with enough energy to eject electrons from an atom.

What needs to hold: Bound energy < Unbound energy + Electron Energy. 

The binding energy is 13.6 eV which is the binding energy of hydrogen.
For a Medical CT you need around 100keV, for Mammography you need around 20keV. 

=== Forms of Ionizing Radiation
1. *Particulate Radiation*: Any subatomic particle (proton, neutron, electron) with enough kinetic energy to be ionizing.
2. *Electromagnetic Radiation*: Can act as a wave or a particle (photon). If energy $> 13.6$ eV (binding energy of hydrogen electron), it is considered ionizing.

#remark()[
  $ E = h nu $ and $ lambda = c / nu $
  Where:
  - $h$: Planck's constant
  - $nu$: frequency
  - $lambda$: wavelength
  - $c$: speed of light
]

== Interaction of Energetic Electrons with Matter
- *Collision transfer* (~99% $arrow$ heat): Collision with other electrons until kinetic energy is exhausted. If they bump into each other, then energy can be transfered to the other electorn which then will emit infrared photons, which is heat.
- *Radiative transfer* (~1% $arrow$ X-ray):
  - Eject inner shell electron, generating *characteristic X-ray radiation*.
  - Electron flies close to the atom nucleus and is braked by nucleus, generating *Bremsstrahlung X-ray*.

== Interaction of Electromagnetic Radiation with Matter
- *Photoelectrical Effect*: Photon ejects an inner shell electron. The energy is $h nu - E_B$. Filling the hole yields characteristic X-rays or Auger electrons.
- *Compton Scattering*: Photon interacts with outer-shell electrons, yielding a Compton electron and a scattered photon with less energy.

#remark()[
  *Handwritten formula for Compton energy:*
  $ E_c = h nu - h nu' = h(nu - nu') $
]

== Attenuation of Electromagnetic Radiation
Consider a narrow beam geometry with an X-ray source and a detector.

#definition(title: "Beer-Lambert Law Derivation")[
  Let $N$ be the number of photons leaving the source and $N'$ be the photons hitting the detector.
  Suppose $n$ photons are lost in a thickness $Delta x$:
  $ n = N mu Delta x $
  
  The change in photons is:
  $ Delta N = N' - N = -n = -mu N Delta x $

  In the limit $Delta x arrow 0$:
  $ d N = -mu N d x arrow (d N) / N = -mu d x $

  Integrating both sides:
  $ integral (d N) / N = - integral mu d x arrow log(N) = - integral mu d x + C $

  For $x=0$, $N(0) = N_0$, thus $C = log(N_0)$.
  $ N(x) = N_0 exp(- integral mu d x) $
]

Intensity $I$ is proportional to photon count, so $I = I_0 exp(- integral mu(s) d s)$.

=== Narrow Beam vs. Broad Beam
- *Broad beam*: Scattering (Compton effect) causes photons to hit the detector from multiple angles, and the monoenergetic assumption often fails.
- *Rescue*: Use detector collimation to ensure only primary (non-scattered) rays are measured.

== Projection Radiographic System
#definition(title: "Basic Imaging Equation")[
  $ I(x) = integral S_0(E) exp(- integral mu(x, E) d s) d E $
]

*Simplification*: Assuming monoenergetic X-rays with effective energy $E$:
$ y = -log(I / I_0) = integral mu(s) d s $

=== Blurring and Noise
- *Blurring sources*: Penumbra (due to focal spot size), Compton scattering, and detector resolution.
- *Noise*: Photon counting follows a Poisson distribution $N tilde "Pois"(bar(N))$, so the variance is $sigma^2 = bar(N)$.
- *Signal-to-Noise Ratio (SNR)*: To increase SNR, one can increase the photon count or use contrast agents.

== Computed Tomography (CT)
Tomography (from Greek *tomos* "slice" and *grapho* "to write") involves imaging by sectioning a volume using projected radiographs from different orientations.

=== Radon Transform
For a 2D object $f(x, y)$, the projection $g(theta, rho)$ at angle $theta$ and distance $rho$ is given by the line integral:
$ g(theta, rho) = integral_(-infinity)^(infinity) integral_(-infinity)^(infinity) f(x, y) delta(x cos theta + y sin theta - rho) d x d y $

A collection of these projections is called a *sinogram*.

=== Reconstruction Methods
1. *Backprojection*: Project the measured values back onto the image plane.
   $ b(x, y) = integral_0^pi g(theta, x cos theta + y sin theta) d theta $
   *Problem*: Results in a blurry image (1/r blurring).

2. *Filtered Backprojection (FBP)*:
   Apply a high-pass filter (Ramp filter $|q|$) to the projections in the frequency domain before backprojecting.
   #theorem(title: "Fourier-Slice Theorem")[
     The 1D Fourier Transform of a projection at angle $theta$ is equal to a slice of the 2D Fourier Transform of the original image at that same angle.
   ]

== Artifacts and Hounsfield Units
- *Aliasing*: Streak artifacts due to insufficient number of projections.
- *Beam Hardening*: Caused by energy-selective attenuation; low-energy photons are absorbed more easily, shifting the spectrum toward "harder" (higher energy) X-rays.

#definition(title: "Hounsfield Units (HU)")[
  Standardized scale to compare CT scans:
  $ h = 1000 dot (mu - mu_"Water") / (mu_"Water" - mu_"Air") $
]

| Substance | HU |
| :--- | :--- |
| Air | $-1000$ |
| Fat | $-120$ to $-90$ |
| Water | $0$ |
| Muscle | $+35$ to $+55$ |
| Bone | $+300$ to $+1900$ |


#remark()[
  *Historical Note*: The development of CT was funded in part by EMI (the Beatles' record label), leading to Hounsfield's Nobel Prize.
]
