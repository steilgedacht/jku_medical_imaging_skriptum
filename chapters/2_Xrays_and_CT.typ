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

The binding energy of hydrogen is 13.6 eV.
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

- *Collision transfer* (~99% $arrow$ heat): Collision with other electrons until kinetic energy is exhausted. If they bump into each other, then energy can be transfered to the other electron which then will emit infrared photons, which is heat.

- *Radiative transfer* (~1% $arrow$ X-ray):
  - Eject inner shell electron, generating *characteristic X-ray radiation*.
  - Electron flies close to the atom nucleus and is braked by nucleus, generating *Bremsstrahlung X-ray*.

#figure(
  image("../assets/interaction_of_energetic_electrons_with_matter.png", width: 40%),
) 

== Interaction of Electromagnetic Radiation with Matter

- *Photoelectrical Effect*: Photon hit an atom and thereby ejects an inner shell electron. The energy loss of the photon is $h nu - E_B$. Afterwards the photon moves on with a smaller frequency. Then a free flying electron can fill the hole again and that leads to an X-Ray. It could also happen that the released X-Ray directly hits again an electron on the outer-shell and that is then a Auger electron.

- *Compton Scattering*: Photon interacts with outer-shell electrons, yielding a Compton electron and a scattered photon with less energy. This energy loss depends on the deflection angle and is defined by this formula:
$ E_c = h nu - h nu' = h(nu - nu') $

#figure(
  image("../assets/interaction_of_energetic_electrons_with_matter_photon.png", width: 40%),
) 

== X-Ray Generation

You have a tube current that creates a tube voltage between a tungsten rotating anode and a static cathode. The voltage ejects electrons which hit the rotating anode and then it kicks out X-rays.

#figure(
  image("../assets/X-ray-generation.png", width: 70%),
) 

We do want the high energy photons that come from the "leaving the body" distribution. 
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

Different tissue types have different $mu$. That enables us to image the inner world of the body.

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

With that we have a linear inverse problem. The only thing that changed was first I was realated to a Poisson Distribution and was related to counting. Now it is more gaussian. And that is something we can work with leaving us with essentially

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
2. Increase photon count $N_b$ (not so healthy for the patient)

== Computed Tomography (CT)

Tomography (from Greek *tomos* "slice" and *grapho* "to write") involves imaging by sectioning a volume using projected radiographs from different orientations. Compared to X-ray we have here a lot more unkowns as we do not need to reconstruct a projection, but we need to reconstruct every single slice. You basically have your X-ray source again and a line of detectors with a filter to compensate for the compton scattering.

#figure(
  image("../assets/CT.png", height: 150pt),
) 

In the first generation you just had 1 detector, so it took forever to scan.

#figure(
  image("../assets/CT2.png", height: 150pt),
) 

And nowadays you have detectors all around you and it only takes seconds to scan.

#figure(
  image("../assets/CT3.png", height: 150pt),
)

First the scanners measured the energy. Nowadays they are able to measure each photon individually. With that you get much sharper scans. And historic overview can be found here:

#block(text(size: 6pt, {

  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: left + horizon,
    [*Generation*], [*Source*], [*Source\ Collimation*], [*Detector*], [*Detector\ Collimation*], [*Source-Detector\ Movement*], [*Advantages*], [*Disadvantages*],
    
    [1G], [Single x-ray tube], [Pencil beam], [Single], [None], [Move linearly and rotate in unison], [Scattered energy is undetected], [Slow],
    
    [2G], [Single x-ray tube], [Fan beam, not enough to cover FOV], [Multiple], [Collimated to source direction], [Move linearly and rotate in unison], [Faster than 1G], [Lower efficiency and larger noise because of the collimation in detectors],
    
    [3G], [Single x-ray tube], [Fan beam, enough to cover FOV], [Many], [Collimated to source direction], [Rotate in synchrony], [Faster than 2G, continuous rotation using a slip ring], [More expensive than 2G, low efficiency],
    
    [4G], [Single x-ray tube], [Fan beam covers the FOV], [Stationary ring of detectors], [Cannot collimate detectors], [Detectors are fixed, source rotates], [Higher efficiency than 3G], [High scattering since detectors are not collimated],
    
    [5G (EBCT)], [Many tungsten anodes in single large tube], [Fan beam], [Stationary ring of detectors], [Cannot collimate detectors], [No moving parts], [Extremely fast, capable of stop-action imaging of beating heart], [High cost, difficult to calibrate],
    
    [6G (Helical CT)], [3G/4G], [3G/4G], [3G/4G], [3G/4G], [3G/4G plus linear patient table motion], [Fast 3D images], [A bit more expensive],
    
    [7G (Multiple-row detector CT)], [Single x-ray tube], [Cone beam], [Multiple arrays of detectors], [Collimated to source direction], [3G/4G/6G motion], [Fast 3D images], [Expensive]
  )
}))

=== Image formation

Recall from radiography: 
$ I_d = integral_0^(E_max) S_0(E) dot E dot exp(- integral_(Gamma_d) mu(s, E) d s) d E $

$S_0(E)$ is the polyenergetic spectrum from the X-ray source, so the distribution of the energy of the photons. $E$ is the Energy level. $- integral_(Gamma_d) mu(s, E) d s$ is the linear attenuation, the $mu(s,E)$ is the linear attenuation coefficient that we want to recover as it gives us the distribtuion of the different materials inside the body along a ray. $Gamma_d$ is the curve that the ray traverses, so in our case a straight line (we neglect the particle interaction inbetween).

Let's assume an $overline(E)$ exists and $overline(E)$ is the monoenergetic effective energy that yields the same intensity $I_d$. With that we can simplify to

$ I_d = I_0 exp(- integral_(Gamma_d) mu(s, overline(E)) d s) $
and this is a linear problem. So basically we assume that the spectrum is very peaky anyway, so that almost all photons have the same energy. When the spectrum is broader spreaded, this assumption breaks and we get a lot of artifacts that are called "beam hardening". This whole problem with the artifacts vanishes when we use the newer technology of just counting single photons.

$ g_d = -ln(I_d / I_0) = integral_(Gamma_d) mu(s, overline(E)) d s = underbrace(a_d^T, d^"th" "column of" A) mu => A x = y $

=== Parallel-Ray Reconstruction

In newer Scaners, we measure with a cone instead of parallel lines, as the source rotates, different rays from different time-points are parallel to each other and in post-processing this can be rearranged again to have parallel rays.

Let's fix a 2D-line: $ Gamma(l, theta) = { binom(x, y) : x dot cos theta + y dot sin theta = l } $

Then, the line integral reads as:
$ g(l, theta) = integral_(-infinity)^infinity f(x(s), y(s)) d s $

where we have $ binom(x(s), y(s)) = underbrace(binom(l dot cos theta, l dot sin theta), "original vector") + underbrace(binom(-sin theta, cos theta), "normal vector") s $

Then we get the Radon Transform $ g(l, theta) = integral_(-infinity)^infinity integral_(-infinity)^infinity f(x, y) dot underbrace(delta(x cos theta + y sin theta - l), "Dirac" delta  = cases(infinity &"if" lambda = 0, 0 &"else") "of" Gamma(l, theta) ) d x d y  $ 

For a fixed angle $theta$, we call $g(l, theta_i)$ a *projection* and for all theta we call it *sinogram*.

#figure(
  image("../assets/radon.gif", height: 150pt),
)
#figure(
  image("../assets/sinogram.png", height: 150pt),
)


=== Reconstruction Methods

=== Backprojection

Idea: Simply project or smear each measurement $g(l, theta)$ back onto the image, because at the spots with the heighest intensities there must have been the most material. For one angle:

$ b_theta (x, y) = g(x cos theta + y sin theta, theta) $

Taking all angles into account we:

$ f_b (x, y) = integral_0^pi g(x cos theta + y sin theta, theta) thin d theta $

The first image is the $b_theta$ and the middle image would be now the $f_b (x,y)$. The right image is the ground truth.

#figure(
  image("../assets/backpropagation.png", height: 100pt),
)

=== Projection-Slice Theorem (Central Slice Theorem)

$ g(l, theta) arrow.long^(1D "FFT") G(rho, theta) = cal(F)_(1D) (g(l, theta)) = integral_(-oo)^(oo) g(l, theta) dot exp(-i 2 pi rho l) d l $

$ = integral_(-oo)^(oo) integral_(-oo)^(oo) integral_(-oo)^(oo) f(x, y) delta(x cos theta + y sin theta - l) exp(-i 2 pi rho l) d l d y d x $
The only time when the $delta$ is not $0$ is when $l= x cos theta + y sin theta$. With that we can get rid of one integral:
$ = integral_(-oo)^(oo) integral_(-oo)^(oo) f(x, y) dot exp(-i 2 pi rho (x cos theta + y sin theta)) d x d y $


Recall definition of 2D Fourier transform: $u = rho cos theta, v = rho sin theta$

$ F(u, v) = cal(F)(f) = integral_(-oo)^(oo) integral_(-oo)^(oo) f(x, y) dot exp(-i 2 pi (u x + v y)) d x d y $

$ arrow.double underbrace(F(rho cos theta, rho sin theta),"2D Fourier transform of image") = underbrace(G(rho, theta),"1D Fourier transf. of projection") $


=== Filtered Backprojection (FBP):

$ f(x, y) = integral_(-oo)^(oo) integral_(-oo)^(oo) F(u, v) exp(i 2 pi (u x + v y)) d u d v $

Change of variables: $u = rho cos theta, v = rho sin theta arrow.double vec(u) = vec(rho cos theta, rho sin theta) $

$ integral_Gamma f(g(x)) d x = integral f(xi) dot |det frac(partial g, partial xi)| d xi $ 

$ frac(partial u, partial (rho, theta)) = mat(cos theta, rho sin theta; sin theta, - rho cos theta) arrow.double |frac(partial vec(u), partial (rho, theta))| = | - rho cos^2 theta - rho sin^2 theta | =  rho $

$ arrow.double f(x, y) = integral_(-oo)^(oo) integral_0^pi F(rho cos theta, rho sin theta) exp(i 2 pi rho (cos theta x + sin theta y)) |rho| d theta d rho $

$ = integral_(-oo)^(oo) integral_0^pi G(rho, theta) dot exp(i 2 pi rho (cos theta x + sin theta y)) |rho| d theta d rho $

#figure(
  image("../assets/sinogrampartial.png", height: 170pt),
)
== Artifacts and Hounsfield Units

*Aliasing*: Streak artifacts due to insufficient number of projections.

#figure(
  image("../assets/artifact_1.jpg", height: 170pt),
)

*Beam Hardening*: Caused by energy-selective attenuation; low-energy photons are absorbed more easily, shifting the spectrum toward "harder" (higher energy) X-rays.
#figure(
  image("../assets/artifact_2.jpg", height: 170pt),
)


#definition(title: "Hounsfield Units (HU)")[
  
  Standardized scale to compare different materials in CT scans:
  $ h = 1000 dot (mu - mu_"Water") / (mu_"Water" - mu_"Air") $

]
#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  align: horizon,
  stroke: (x, y) => if y == 0 { none } else { 0.5pt + gray.lighten(50%) },
  
  table.header(
    [*Substance*], [], [*HU*]
  ),
  
  [Air], [], [-1000],
  [Fat], [], [-120 to -90],
  [Bone], [Cancellous], [+300 to +1900],
  
  [Other blood], [Unclotted], [+13 to +50],
  [], [Clotted], [+50 to +75],
  
  [Fluids], [Water], [0],
  [], [Urine], [-5 to +15],
  [], [CSF], [+15],
  
  [Parenchyma], [Lung], [-700 to -600],
  [], [Kidney], [+20 to +45],
  [], [Liver], [60 ± 6],
  [], [Muscle], [+35 to +55],
  [], [White matter], [+20 to +30],
  [], [Grey matter], [+37 to +45],
)


#remark()[
  *Historical Note*: The development of CT was funded in part by EMI (the Beatles' record label), leading to Hounsfield's Nobel Prize.
]
