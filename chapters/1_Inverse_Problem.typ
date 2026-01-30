#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Inverse Problems 

== What is an Inverse Problem?

There exit a "Forward Problem" which estimate the effect from the cause and then there is inverse Problem which estimates the cause from the effect. In the medical context that would be finding the cause illness given from a certain syntom/effect.  
Typically, the forward problem is ”easy” and well described. The challenge here is: We need to solve the inverse problem given only the observed effect of the
forward problem. 

As an Example from the real world: 
forward problem: The street becomes wet when it rains.
backward problem would be: We observe that the street is wet. Why?

There are multiple different causes:
- Rain
- Fog
- Cleaning

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
Observe a blurred image $ f = k * u $ on a domain $Omega subset RR^2$.

*Inverse Problem*  
Estimate the sharp image $u: Omega -> RR$ given the blur kernel $k: Omega times Omega -> RR_+$ 
On of the oldest clasical methods to do that is the Wiener Filter.
Deconvolution is linked to Fourier $F$:

$ f= k * u $ 

$ F(f) = F(k) dot.o F(u) $

If we wanna do the inverse:

$ F^(-1) {F(f)} = F^(-1){F(k) dot.o F(u)} = f $  

where $dot.o$ is a pointwise multiplication. So a estimate $accent(u, hat)$ would be 

$ accent(u, hat) = F^(-1) (F(f)/F(k)) $

The only problem here is when we have 0 frequencies in the kernel. The Wiener Filtering introduces

$ accent(u, hat) = F^(-1) (F(f)/(I sigma^2 F(k))) $

== What is an Inverse Problem? (formal)

#definition(title: "Inverse Problem")[
  Given a matrix $A in RR^(m times n)$ and a vector $x in RR^n$ the forward problem is $y = A x in RR^m$ 

  The inverse problem is: Given $A$ and $y$, estimate $x$.
]

== Vector Space 

#definition(title: "Vector Space")[
  A non-empty set $V$ is a vector space over a field $FF in {RR, CC}$ if there are operations of vector addition: $+:V times V -> V$  and scalar multiplication: $dot:FF times V -> V$ satisfying the following axioms:

  *Vector addition*
  1. $u+v in V quad forall u,v in V$
  2. $u + v = v + u$
  3. $(u + v) + w = u + (v + w) quad forall u,v,w in V$
  4. $exists 0 in V: u + 0 = u quad forall u in V$ 
  5. $forall u in V: exists -w: u + (-w) = 0$

  *Scalar multiplication*
  1. $a v in V quad forall a in FF, forall v in V$
  2. $(a b) v = a ( b v) quad forall a,b in FF, v in V$
  3. $a (u + v)  = a u + a v quad forall a in FF, forall u,v in V$
  4. $(a + b)v = a v +  b v quad forall a,b in FF, forall v in V$
  5. $exists 1 in FF:1 *  u = u quad forall u in V$
]


*Vector Space Examples*

- $bb(R)^n = {(x_1, ..., x_n)^T : x_1, ..., x_n in bb(R)}$
- $cal(C)(bb(R)^n, bb(R))$ set of function $f: bb(R)^n -> bb(R)$ that are continuous
- $cal(C)^1(bb(R)^n, bb(R))$ set of function $f: bb(R)^n -> bb(R)$ that are continuous and once continuously differentiable
- $L^2(bb(R)^n, bb(R)) = {f: bb(R)^n -> bb(R) : integral_(bb(R)^n) |f(x)|^2 d x < oo}$ Lebesgue space
- $H^1(bb(R)^n, bb(R)) = {f in L^2(bb(R)^n, bb(R)) : integral_(bb(R)^n) |f'(x)|^2 d x < oo}$ Sobolev space ($p=2$), Hilbert space

== Inverse Problem
#definition(title: "Inverse Problem")[
  Let $X, Y$ be vector spaces and $A: X -> Y$. 
  The forward problem is defined as $y = A x$ for any $x in X$.
  The inverse problem is to find $x in X$ such that $A x = y$ for any $y in Y$.
]

So we want to get $A^(-1)(y) = accent(x, hat)$

== Well-Posedness (Hadamard)

We can now start to categorize inverse problems:

#definition(title: "Well-Posedness")[
  The inverse problem $A x = y$ is well-posed if:

  1. *Existence:* a solution exists (EXISTENCE)
  2. *Uniqueness:* the solution is unique (UNIQUENESS)
  3. *Stability:* the solution depends continuously on the data (STABILITY)

  If one condition fails, the problem is ill-posed.
]

*Well-Posedness Example*

Example 1)

Is this example well posed?
Let $X , Y = RR $ and $A:RR -> RR, x arrow.r x^2$

Answer:

- Existence: for $y = -1$ no solution exists (if it would be $RR^+$ it would be okay)
- Uniqueness: for $y = 1$, $x = plus.minus 1$ which is not unique
- Stability: yes, since $A$ is continuous

Example 2)

Let $X, Y = RR^2$ and $A = mat(2, 3; 1, 2) in RR^(2 times 2)$. Is the inverse problem $A x = y$ for $y in Y$ well-posed?

1. *EXISTENCE:* $exists A^(-1)$? Since $det(A) = 4 - 3 = 1 != 0$, the matrix is invertible.
2. *UNIQUENESS:* Yes, because $det(A) != 0$.
3. *STABILITY:* Yes, as $A^(-1)$ is continuous.

== Inner Product

#definition(title: "Inner Product")[
  An inner product on a vector space $Y$ over a $FF$ is a map $ ⟨., .⟩: Y times Y -> FF $  with the following properties:

  1. Symmetry: $⟨x, y⟩ = overline(⟨y, x⟩) quad x,y in Y$
  2. Additivity: $⟨x , y + z ⟩ = ⟨x, y⟩ + ⟨x, z⟩ quad x,y,z in Y$
  3. Homogeneity: $⟨lambda x, y⟩ = lambda ⟨x, y⟩ quad x,y in Y quad lambda in RR$
  4. Positivity: $⟨x, x⟩ >= 0$ and $⟨x, x⟩ = 0 <==> x = 0$
]


== Vector Norm

#definition(title: "Inner Product")[
  A vector norm is a vector space Y over a field $F$ is a map $norm(.): Y -> RR$ with:

  + *NON-NEGATIVITY* $norm(x) >= 0 quad forall x in V, norm(x) = 0 <=> x = 0$
  + *POSITIVE HOMOGENEITY* $norm(lambda x) = |lambda| norm(x) space forall x in Y, lambda in FF$
  + *TRIANGLE INEQUALITY* $norm(x + y) <= norm(x) + norm(y) space x, y in V$
]

*Example: $l_p$-norm*
$ norm(x)_p = root(p, sum_(i=1)^n |x_i|^p) quad x in X subset RR^n $
== Definition: Matrix Norm

#definition(title: "Inner Product")[
  Let $norm(dot)_a$ and $norm(dot)_b$ be vector norms on $RR^n$ and $RR^m$, respectively.
  Given a matrix $A in RR^(m times n)$, the *induced matrix norm* $norm(A)_(a,b)$ is defined as:

  $ norm(A)_(a,b) = max_(x in RR^n : norm(x)_a <= 1)  norm(A x)_b = sup_{x in RR^n \\ {0} }(norm(A x)_b) / (norm(x)_a) $

  $ norm(A x)_b <= norm(A)_(a b) norm(x)_a $
]

*Matrix Norm Examples*
- If $a,b=2$: $norm(A)_(2,2) =norm(A)_(2)  = sigma_(max) (A)=sqrt(lambda_max (A^T A))$ 
- If $a,b=1$: $norm(A)_(1,1)=norm(A)_(1)= max_j sum_i |A_(i j)|$ 
- If $a,b=infinity$: $norm(A)_infinity = max_i sum_j |A_(i j)|$ 

== Injection, Surjection, Bijection

These properties of mappings $A: X -> Y$ are defined as

- *Injection:* $A: X -> Y$ is injective if $A x_1 = A x_2 => x_1 = x_2$.
- *Surjection:* $A: X -> Y$ is surjective if $forall y in Y, exists x in X : A x = y$.
- *Bijection:* $A: X -> Y$ is bijective if it is both injective and surjective. $forall y in Y, exists! x in X : A x = y <=> exists A^(-1):x=A^(-1)y$.


== Null Space and Range

Let $A: X -> Y$ where $X, Y$ are vector spaces.
- *Nullspace of A:* $N(A) = {x in X : A x = 0}$
- *Range space of A:* $R(A) = {A x in Y : x in X}$

== Connection to Hadamard's Definition

- *Existence* $<=>$ Surjection $<=>$ $R(A) = Y$
- *Uniqueness* $<=>$ Injection $<=>$ $N(A) = {0}$
- *Existence & Uniqueness* $<=>$ Bijection

== Definition of the Linear Inverse Problem

Given $A: X -> Y$ and observation $y in Y$ the inverse problem is called linear if $A$ is linear which means that 
$A(alpha x_1 + beta x_2) = alpha A(x_1) + beta A(x_2)$

Example:

$A dots$ is the Radon transform

$ (A x)_i = y_i = integral_(Gamma_i) x(s) d s $

$ A(hat(x)) &= A(lambda_1 dot x_1 + lambda_2 x_2) = hat(y)_i = integral_(Gamma_i) hat(x)(s) d s = integral_(Gamma_i) lambda_1 x_1(s) + lambda_2 dot x_2(s) d s \ 
&= lambda_1 underbrace(integral_(Gamma_i) x_1(s) d s, y_i^1) + lambda_2 underbrace(integral_(Gamma_i) x_2(s) d s, y_i^2) = lambda_1 y_i^1 + lambda_2 y_i^2 = lambda_1 A(x_1)_i + lambda_2 A(x_2)_i $

Nullspace of linear A
$=> {0} in cal(N)(A)$

== Decomposition of Square Matrices

Let $A in RR^(n times n)$, recall Eigenvalues $lambda_i$ and Eigenvectors $v_i$:
$ A v_i = lambda_i v_i space "for" i = 1, dots, n $
$ det(A - lambda I) = 0 $

If $v_i$ are linearly independent: $A v_i = lambda_i v_i => A Q = Q Lambda => A = Q Lambda Q^(-1)$
Where $Q = (v_1, dots, v_n)$.

Remark: If $A$ is hermitian $<=> A^* = A$, we have that all $lambda_i$ are real & $v_i$ are orthonormal.

$ v_i^T v_j = 0 quad "for" i != j $

$ A = Q Lambda Q^T $

== Singular Value Decomposition

Let $X = RR^n, Y = RR^m$ be an inverse problem $A x=y$ with a $A in RR^(m times n)$. The Goal:
$ A = U Lambda V^T $
- $U in RR^(m times p)$, $Lambda in RR^(p times p)$, $V in RR^(p times n)$
- $p$ is the number of non-zero singular values $sigma_1 >= sigma_2 >= dots >= sigma_p > 0$.

*Link between SVD and Eigendecompostion*

$ A in RR^(m times n) $ 
$ cases(
  (1) quad A x = y,
  (2) quad A^T hat(x) = hat(y)
) 
<=> 
underbrace(mat(0, A; A^T, 0), B in RR^((m+n) times (m+n))) dot vec(hat(x), x) = vec(y, hat(y))$

#line(length: 20%, stroke: 1pt)

$B = B^T:$ $quad B w_i = lambda_i w_i$

$ mat(0, A; A^T, 0) vec(u_i, v_i) = lambda_i vec(u_i, v_i) <=> cases(
  "1st:" quad A v_i = lambda_i u_i,
  "2nd:" quad A^T u_i = lambda_i v_i
) $

#grid(
  columns: (1fr, 1fr),
  [
    *1st:*
    $ lambda_i A v_i = lambda_i^2 u_i $
    $ A (lambda_i v_i) = lambda_i^2 u_i $
    $ A A^T u_i = lambda_i^2 u_i $
    $ U = (u_1 | dots | u_m) $
  ],
  [
    *2nd:*
    $ A^T (lambda_i u_i) = lambda_i^2 v_i $
    $ A^T A v_i = lambda_i^2 v_i $
    $ V = (v_1 | dots | v_n) $
  ]
)

== Least Squares (m > n)

$A x = y quad A in RR^(m times n) quad m > n "overdetermined system"$

$ e_i = a_i^T x - y_i $

#underline[*Idea:*] minimize the squared error

$ hat(x) = arg min E(x) := 1/2 sum_(i=1)^m (a_i^T x - y_i)^2 = 1/2 ||A x - y||_2^2 = 1/2 ||e||_2^2 $

#h(1cm) where $e = A x - y$

How do we solve this optimization problem?

$ nabla E(x) = 0 = frac(partial e, partial x) frac(partial E, partial e) = frac(partial e, partial x) 1/2 2 e = A^T e = A^T (A x - y) = 0 $

#underline[$ nabla E(x) in RR^n $]

Least squares solution

$ A in RR^(m times n) $

$ nabla E = A^T (A x - y) = 0 $

$ (A^T A) x = A^T y $

$ x = (A^T A)^(-1) A^T y $

 Example: $2 times 2$ CT reconstruction

$ x in RR^4 quad y in RR^5 $

$ A x = y $

$ mat(
  1, 0, 1, 0;
  0, 1, 0, 1;
  1, 1, 0, 0;
  0, 0, 1, 1;
  1, 0, 0, 1;
) vec(x_1, x_2, x_3, x_4) = vec(y_1, y_2, y_3, y_4, y_5) $

== Solving Inverse Problems ($p = n > m$)

Let $A x = y$ with $A in RR^(m times n)$

*Remark:* Since $n > m$, $(A^T A)^(-1)$ does not exist. This is an *underdetermined system*.
Multiple solutions exactly solve $A x = y$. We pick one using a priori knowledge:

$ min_x 1/2 ||x||_2^2 quad "s.t." quad A x = y $ 


=== Recap: Lagrange Multipliers 
To solve $min E(x)$ subject to $C(x) = 0$

Define Lagrangian: $cal(L)(x, tau) = E(x) + chevron.l C(x), tau chevron.r$ 

Find solution by $nabla cal(L)(x, tau) = 0$

$ cases(partial / (partial x) cal(L) = (partial E) / (partial x) + (partial C) / (partial x) tau = 0, partial / (partial tau) cal(L) = C(x) = 0) $ 


=== Minimum Length Solution 

Find $x "s.t." A x=y$ and $norm(x)_2^2 arrow.r min$

$ E(x) = 1/2 ||x||_2^2 quad C(x) = y - A x = 0 <=> h(x, tau) = 1/2 norm(x)^2_2 + chevron.l y-A x, tau chevron.r $ 

$ partial / (partial x) cal(L) = x - A^T tau = 0 <=> x = A^T tau $ 
$ partial / (partial tau) cal(L) = y - A x = 0 <=> y = A x= A(A^T tau) = (A A^T) tau <=> tau = (A A^T)^(-1) y $ 

$ x = A^T (A A^T)^(-1) y $ 


== Generalized Inverse

Let $X = RR^n$, $Y = RR^m$ and the inverse problem $A x = y$ with $A in RR^(m times n)$.

Define the *generalized inverse* as:
$ A_g^(-1) = (U_p Lambda_p V_p^T)^(-1) = (V_p^T)^(-1) Lambda_p^(-1) U_p^(-1) = V_p Lambda_p^(-1) U_p^T $


=== Check if GI computes Exact, LS, ML:

==== I. $p = m = n$:
$ A_g^(-1) = V_p Lambda_p^(-1) U_p^T quad | dot A = U_p Lambda_p V_p^T $
$ A_g^(-1) A = V_p Lambda_p^(-1) underbrace(U_p^T U_p, I) Lambda_p V_p^T = V_p underbrace(Lambda_p^(-1) Lambda_p, I) V_p^T = I $

==== II. $p = m > n$:
$ x &= (A^T A)^(-1) A^T y \
   &= ((U_p Lambda_p V_p^T)^T (U_p Lambda_p V_p^T))^(-1) (U_p Lambda_p V_p^T)^T y \
   &= (V_p Lambda_p underbrace(U_p^T U_p, "Id") Lambda_p V_p^T)^(-1) (V_p Lambda_p U_p^T) y \
   &= (V_p Lambda_p^2 V_p^T)^(-1) V_p Lambda_p U_p^T y \
   &= V_p Lambda_p^(-2) underbrace(V_p^T V_p, "Id") Lambda_p U_p^T y \
   &= V_p Lambda_p^(-1) U_p^T y = A_g^(-1) y $


==== III. $p = m < n$: 
$ x &= A^T (A A^T)^(-1) y \
   &= (V_p Lambda_p U_p^T) (U_p Lambda_p V_p^T V_p Lambda_p U_p^T)^(-1) y \
   &= (V_p Lambda_p U_p^T) (U_p Lambda_p^2 U_p^T)^(-1) y \
   &= V_p Lambda_p underbrace(U_p^T U_p ) Lambda_p^(-2) U_p^T y \
   &= V_p Lambda_p^(-1) U_p^T y = A_g^(-1) y $

=== IV. $0 < p < min(m, n)$:
However, $A_g^(-1)$ still exists. It computes a solution that *interpolates between LS & ML solutions*.

== Regularization

Consider polynomial regression
$ p(a) = sum_(i=1)^n x_i a^(i-1) = x_1 dot 1 + x_2 dot a + ... + x_n a^(n-1) $

Where $x$ represents the *coefficients of the polynomial*.

$ arrow.l.r.double A x = mat(
  1, a_1, a_1^2, ..., a_1^(n-1);
  dots.v, dots.v, dots.v, , dots.v;
  1, a_m, a_m^2, ..., a_m^(n-1);
) dot vec(x_1, dots.v, x_n) = vec(y_1, dots.v, y_m) $

The matrix $A$ has dimensions $[m times n]$.

*How to choose $n$?*
- Manually
- Very large + regularization


=== Incorporating Prior Knowledge

*Least squares problem + regularization:*
$ hat(x) = arg min_x 1/2 ||A x - y||_2^2 + R(x) $

*Example:* $R(x) = lambda/2 ||x||_2^2$ (Tikhonov regularization, aka weight decay)

*Derivation:*
$ 1/2 ||A x - y||^2 + lambda/2 ||x||_2^2 arrow.r min $
$ 1/2 dot 2 dot A^T (A x - y) + lambda/2 dot 2 x = 0 $
$ A^T A x + lambda x = A^T y arrow.l.r.double x = (A^T A + lambda I d)^(-1) A^T y $


=== Regularization Types and Intuition

#table(
  columns: (auto, 1fr, 2fr),
  inset: 10pt,
  align: horizon,
  table.header([*Name*], [*$R(x)$*], [*Intuition*]),
  [Tikhonov], [$lambda ||G x||_2^2$], [Existence of Inverse],
  [$L^2$], [$lambda ||x||_2^2$ \ (#emph[G = Id])], [Minimum length/norm],
  [$H^1$], [$lambda ||nabla x||_2^2$ \ (#emph[G = $nabla$])], [Smooth gradients],
  [$L^1$], [$lambda ||x||_1$], [Sparse solutions],
  [Total variation (TV)], [$lambda ||nabla x||_1$], [Sparse gradients (piece-wise constant solutions)],
)

== The Proximal Mapping

=== 1. Projection onto a set $S$
$ "proj"_S (x) = arg min_(y in S) 1/2 ||x - y||_2^2 $

=== 2. Proximal mapping of a function $g(x)$
$ "prox"_g (x) = arg min_y 1/2 ||x - y||_2^2 + g(y) $

If we define $g(y)$ as the indicator function:
$ g(y) = cases(0 &"if" y in S, infinity &"else") $

==== Example: $g(x) = |x|$
To find the proximal mapping for the absolute value (L1 norm), we solve:
$ "prox"_(|dot|) (x) = arg min_y 1/2 (x - y)^2 + |y| $

The subdifferential of $|x|$ is:
$ d/(d x) |x| = cases(1 &x > 0, [-1, 1] &x = 0, -1 &x < 0) $

To minimize, we set the subgradient to zero:
$ x - y + partial g(y) "contents" 0 $

- *Case $y > 0$:* $-(x - y) + 1 = 0 arrow.double y = x - 1 > 0 arrow.double x > 1$
- *Case $y < 0$:* $-x + y - 1 = 0 arrow.double y = x + 1 < 0 arrow.double x < -1$
- *Case $y = 0$:* $-x + 0 + [-1, 1] "contents" 0 arrow.double x in [-1, 1]$

Thus, the Soft Thresholding operator is:
$ "prox"_(|dot|) (x) = cases(x - 1 &"if" x > 1, x + 1 &"if" x < -1, 0 &"else") $


== Regularization IV: A Probabilistic Perspective

Assume observed measurements $y$ follow a Gaussian distribution:
$ y ~ cal(N)(A x, Sigma) <==> p(y|x) = |2 pi Sigma|^(-1/2) exp(-1/2 ||A x - y||_(Sigma^-1)^2) $

Moreover, we assume the solution (or its gradients) follows a Gaussian prior:
$ nabla x ~ cal(N)(0, eta "Id") <==> p(x) = |2 pi eta I|^(-1/2) exp(-1/2 eta ||x||^2) $

Using Bayes' Rule to find the posterior distribution:
$ p(x|y) = (p(y|x) dot p(x)) / p(y) $

Taking the logarithm:
$ log p(x|y) = log p(y|x) + log p(x) - log p(y) $
$ log p(x|y) = -1/2 ||A x - y||_(Sigma^-1)^2 - log Z_1 - 1/(2 eta) ||x||^2 - log Z_2 - log p(y) $

Since $Z_1, Z_2,$ and $p(y)$ are constants that do not depend on $x$:
$ max_x log p(x|y) = max_x -1/2 ||A x - y||_Sigma^-1^2 - 1/(2 eta) ||x||^2 $
$ min_x -log p(x|y) = min_x underbrace(1/2 ||A x - y||_(Sigma^-1)^2, D(x,y) " (Data Fidelity)") + underbrace(1/(2 eta) ||x||^2, R(x) " (Regularizer)") $

*Conclusion:* The variational formulation of inverse problems corresponds to the Maximum A Posteriori (MAP) estimation.
