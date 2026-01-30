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

  $ E = h nu quad$ and $quad lambda = c / nu $

  The variables stand for:
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

#figure(
  image("../assets/interaction_of_energetic_electrons_with_matter.png", width: 40%),
) 

== Interaction of Electromagnetic Radiation with Matter
- *Photoelectrical Effect*: Photon hit an atom and thereby ejects an inner shell electron. The energy loss of the photon is $h nu - E_B$. Afterwards the photon moves on with a smaller frequency. Then a free flying electorn can fill the hole again and that leads to an X-Ray. It could also happen that that released X-Ray directly hits again an electron on the outer-shell and that is then a Auger electron.
- *Compton Scattering*: Photon interacts with outer-shell electrons, yielding a Compton electron and a scattered photon with less energy. This energy loss depends on the deflection angle and is defined by this formula:
$ E_c = h nu - h nu' = h(nu - nu') $

#figure(
  image("../assets/interaction_of_energetic_electrons_with_matter_photon.png", width: 40%),
) 

== X-Ray Generation

You have a tube current that creates a tube voltage between a tungsten rotating anode and a static cathode. The voltage eject electrons which hit the rotating anode and then it kicks out X-rays.

#figure(
  image("../assets/X-ray-generation.png", width: 70%),
) 

We do want the high energy photons that can "leave the body" distribution. 
#figure(
  image("../assets/X-ray_spectrum.png", height: 150pt),
) 

== Attenuation of Electromagnetic Radiation
Consider a narrow beam geometry with an X-ray source and a detector.

#figure(
  image("../assets/narrow_beam_vs_broad_beam.png", height: 150pt),
) 

Let $N$ be the number of photons leaving the source and $N'$ be the photons hitting the detector.
Suppose $n$ photons are lost in a thickness $Delta x$:
$ n = N mu Delta x $

where $mu$ is some kind of material coefficient. The change in photons is:
$ Delta N = N' - N = -n = -mu N Delta x $

In the limit $Delta x arrow 0$:
$ d N = -mu N d x arrow (d N) / N = -mu d x $

Integrating both sides:
$ integral (d N) / N = - integral mu d x arrow log(N) = - integral mu d x + C $

For $x=0$, $N(0) = N_0$, thus $C = log(N_0)$.
$ N(x) = N_0 exp(- integral mu (x) d x) $

We have now 2 quantities to describe this setting:
- photon fluence rate (number of photons over some area of some time ) $psi = (N / A delta t)$
- itensity of a beam $I = psi  E$



=== Narrow Beam vs. Broad Beam
Scattering (Compton effect) causes photons to hit the detector from multiple angles with different energy levels and the monoenergetic assumption often fails. The rescue is to use the detector collimation to ensure only primary (non-scattered) rays are measured.

#figure(
  image("../assets/collimators.png", height: 150pt),
) 

== Attenuation of different tissue types

Different tissue types have different $mu$. That enables us to images something.

#figure(
  image("../assets/attenuation.png", height: 150pt),
) 

== Projection Radiographic System
#figure(
  image("../assets/basic_imaging_equation_ct.png", height: 150pt),
) 
$ N(x) = N_0 dot exp(- integral_0^x mu(hat(x), E) d hat(x)) $
We have the Spectrum $S$ as
$ S(x, E) = S_0(E) dot exp(- integral_0^x mu(hat(x), E) d hat(x)) $
So we get a formula for the intensity:
$ I(x) = integral_0^infinity underbrace(N(x) / (A dot Delta t), S(x, E) ) dot E d E $
$ I(x) = integral_0^infinity S_0(E) E exp(- integral_0^S mu(hat(x), E) d hat(x)) d E $

And this is rather complicated, so we do some simplification: Assuming monoenergetic X-rays with effective energy $E$:
$ S(E) = delta(overline(E)) $
We say here that the Spectrum should be describale by some $delta ( overline(E))$ which is the monoenergetic radiation

$ I(x) = I_0 dot exp(- integral_0^x mu(tilde(x), overline(E)) d tilde(x)) $
This is a non-linear problem. The unkown here are the $mu$. We already have the measurements. Let's further simplify:

$ I(x) / I_0 = exp(- integral_0^x underbrace(mu(tilde(x), overline(E)), mu(tilde(x))) d tilde(x)) $

$ log I(x) / I_0 = - integral_0^x mu(tilde(x)) d tilde(x) quad <=> quad underbrace(log I_0 / I(x), hat(y)) = integral_0^x mu(tilde(x)) d tilde(x) $

$ hat(y)_i = integral_0^x mu(hat(x)) d hat(x) $

With that we have a linear inverse problem. The only thing that changed was first I was realated to a Poisson Distribution and was related to counting. Now it is more gaussian. And that is somehing we can work with leaving us with essentially

$ hat(y)_i = arrow(a)_i^T arrow(mu) $
=== Blurring and Noise
Blurring sources are Penumbra (due to focal spot size), Compton scattering, and detector resolution.

#figure(
  image("../assets/point_hole_camera.png", height: 150pt),
) 
#definition(title: "local contrast")[
  $ C = (I_t - I_b) / I_b $ 
  $I_t$: target intensity (lesion), $I_b$: background intensity
] 


#definition(title: "Signal to noise ratio")[
  $ "SNR" = (I_t - I_b) / sigma_b = (C dot I_b) / sigma_b $
  $sigma_b$: Std in the background

] 

Recall $ I = (N dot E) / (A Delta t) => I_b = (N_b E) / (A_b Delta t) $

$N$ is the number of measured photons $=>$ counting discrete events. We know $ N tilde upright("Poisson")(N (E / (A_b Delta t))^2) $ with variance $ sigma_b^2 = N_b (E / (A Delta t))^2 $

$ => "SNR" = (C dot I_b) / sigma_b = (C dot (N_b E) / (A_b Delta t)) / (sqrt(N_b) E / (A_b Delta t)) = C dot sqrt(N_b) $

To increase SNR:
1. Increase $C$ (maybe with contrast agent)
2. Increase photon count $N_b$

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
