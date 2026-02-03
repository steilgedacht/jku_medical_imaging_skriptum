
#import "@preview/classicthesis:0.1.0": *
#import "../template.typ": *

= Federated Learning


== Data Protection in Healthcare
Data protection is critical in healthcare due to the high sensitivity of patient records (medical history, genetics, diagnoses). It is regulated by laws such as:
- *GDPR (EU)*: General Data Protection Regulation, which regulates how personal data of EU residents is collected, stored, and processed.
- *HIPAA (USA)*: Health Insurance Portability and Accountability Act, which sets national standards for protecting sensitive patient health information.

Breaches of these regulations can lead to identity theft, loss of trust, and severe legal penalties.

== Personal Data and Re-identification


#definition(title: "Personal Data (GDPR)")[
  Personal data is any information relating to an identified or identifiable living individual. Data that has been de-identified or pseudonymized but can still be used to re-identify a person remains personal data.
]

*Anonymization*: To be truly anonymized, the process must be irreversible.

#remark()[
  *Re-identification Risk*: A famous study by Sweeney (2000) showed that $87%$ of US citizens can be uniquely identified using only their ZIP code, birth date, and sex. Thats why you dont use the birth date anymore, but the age.
]

== From Centralized to Federated Learning


- Centralized ML: Training data from all sources is moved to a central server.
- Distributed On-Site Learning: Models are trained locally at each site with no information exchange.
- Federated Learning (FL): A collaborative learning approach where data remains at the source, and only model updates (weights) are shared with a central server.


=== Comparison: Centralized vs. Federated Learning
#table(
  columns: (1fr, 1fr),
  inset: 10pt,
  align: (left, left),
  
  [CENTRALIZED LEARNING], [FEDERATED LEARNING],
  

  [Trained on centralized data], [Trained on distributed data],
  
  [Data resides on the cloud or centralized server], [Data resides at the various nodes in the network],
  
  [Training takes place primarily in the cloud], [Training happens primarily at the edge],
  
  [Nodes/edge devices share local data], [Nodes/edge devices share local version of the mode],
  
  [Cannot operate on heterogeneous data], [Can operate on heterogeneous data],
  
  [Low user data privacy], [High user data privacy],
)

=== Centralized vs Decentralized Federated Learning

#figure(
  image("../assets/central_vs_decentral.png", height: 150pt),
)
== Centralized Federated Learning
Let's define the terms:

$t$ communication round

Server
- $w^t$ model weights
- $T$ number of rounds
- $C$ fraction of sampled clients of each round

Client
- $w_k^t$ weights on the client site
- $E$ number of local epochs
- $eta$ learning rate
- $P_k$ subset of the data for client k
- $n_k$ number of samples at the k client

#figure(
  image("../assets/centralized_fl.png", height: 150pt),
)

=== Mathematical Formulation

Let $D = (x_i, y_i)^n_(i=1)$ be a dataset distributed to $K$ clients $C_k$ where $k in {1, ... , k}$. We denote by $P={1,... n}$ and each client has a subset $P_k$ such that $P= union.big_(k=1)^n P_k$. The goal is to solve:
$ min_w f(w) = min 1/n sum_(i=1)^N f_i (w) $
where $f_i (w)= l(x_i, y_i, w)$ is a loss function. Then we have that the total loss function 
$ f(w) = 1/n sum^n_(i=1) f_i (w) = sum^n_(i=1) 1/n f_i (w) = sum^K_(k=1) 1/n n_f F_k (w) $
with $F_k (w) = 1/n_k sum_(i in P_k) f_i (w)$ which is the loss at the distributed clients. So the loss function of the sample is the same, but now we combined the indices into the clients and then we write it by $n_k$. It is still the same thing, we just shifted the indices. At the client we do the same thing as globally. 


=== Algorithms: FedSGD and FedAVG
==== FedSGD
A simple version where each client performs one step of gradient descent per round.

#block(fill: rgb("#f9f9f9"), inset: 10pt, radius: 4pt, stroke: 0.5pt + gray, width: 100%)[
Server executes:
#set enum(indent: 1em)
+ *for* each round $t = 1, 2, ..., T$ *do*
  + $S_t subset.eq {1, ..., K}$
  + *for* each client $k in S_t$ *do in parallel:*
    + $g_k = nabla F_k (w_t) = nabla (1/n_k sum_(i in P_k) f_i (w_t))$
  + we add the gradients weighted by the sample number on the client side \ $g_t arrow.l sum_(k in S_t) g_k n_k / n$ 
  + $w_(t+1) arrow.l w_t - eta g_t$
]
The advantage is that it is a simple algorithm and theory of SGD applies but we have a high server-client communication utilization, which might be slow.

==== FedAVG

#block(fill: rgb("#f9f9f9"), inset: 10pt, radius: 4pt, stroke: 0.5pt + gray, width: 100%)[
at server initialize $w^0$. \
*for* each round $t = 1, 2, ..., T$ *do*
#set enum(indent: 1em)
+ sample clients $S_t subset.eq {1, ..., K}$
+ $w_k^1 = w^t$
+ *for* each _local_ epoch $e = 1, 2, ..., E$ *do*
  + compute mini-batch gradient $g_k (w_k^e)$
  + $w_k^(e+1) arrow.l w_k^e - eta g_k (w_k^e)$
+ return $w_k^E$ to server

at the server we aggregate the weights and add them together weighted by their sample number:
$ m_t arrow.l sum_(k in S_t) n_k $
$ w^(t+1) arrow.l sum_(k in S_t) n_k / m_t w_k^E $
]
The advantage is that it substantially reduced communication utilization with a simple aggregation, but the convergence is slower compared to SGD in theory.

#figure(
  image("../assets/fedsdg_vs_fedavg.png", height: 150pt),
)


== Non-IID Data Challenges


Typically, we assume that data samples $(x_i, y_i) tilde P(X, Y)$ are i.i.d. (independent and identically distributed).

We have now a 2-step modelling approach:
1. $k tilde p(k)$ : draw a client from the distribution of clients
2. $(x, y) tilde P_k(X, Y)$ : draw data sample from distribution at client

We assume clients are *non-IID* (which is the fact in reality most of the time) if $P_k != P_l$ for $k, l in {1, ..., K}, l != k$.
Let us assume that we can decompose $P_k$ into $P_k (y | x) dot P_k (x)$ or $P_k (x | y) dot P_k (y)$

=== Non-IID Cases:
+ *Feature distribution skew:* So here the feature will be different \
  $P_k (x) != P_l (x)$ but $P_k (y | x) = P_l (y | x)$ \
  _e.g., shift in demographics, different devices_

+ *Label distribution skew:* So we have the same labels, but the distribution of labels is different \
  $P_k (y) != P_l (y)$ but $P_k (x | y) = P_l (x | y)$ \
  _e.g., certain deseases only occur for elderly people_

+ *Concept shift:* same label but different features \
  $P_k (x | y) != P_l (x | y)$ but $P_k (y) = P_l (y)$ \
  _e.g., houses around the globe are all look very different and have different features, but are still houses_

+ *Concept shift:* inter-reader variability \
  $P_k (y | x) != P_l (y | x)$ but $P_k (x) = P_l (x)$ \
  _e.g., personal preferences, certain doctors scale the images a bit larger_

== Scaffold

In FedSGD, we compute the updates as:
$ w_h^(e+1) arrow.l w_h^e - eta 1/K sum_(k=1)^K underbrace(nabla F_k (w_h^e), g_k (w_h^e))  $

Since computing the full sum of all gradients is often not possible, control variables are introduced to get rid of the stochastisity of the gradient. If we for example have only one summand here, then it is not a good estimate. If we take all clients into account, then 2 clients can go into completely different directions, but the overall averaged direction is the option that works for everyone. On the otherhand if we go first in the one direction, then in the other and then back again with another, is not so good. Therefore we introduce control variables to get rid of this stochastisity:
- $c_k approx nabla F_k (w_h^e)$ (local control variable)
- $c approx 1/K sum_(k=1)^K nabla F_k (w_h^e)$ (global control variable)

Then, SCAFFOLD updates as:
$ g_k (w_h^e) - c_k + c approx 1/K sum_(k=1)^K nabla F_k (w_h^e) $

So we take our existend graident $g_k$ and substract the local aspect $c_k$ and then add our global gradient $c$. So we always have at every step the influence of the global best gradient.


#block(fill: rgb("#f9f9f9"), inset: 10pt, radius: 4pt, stroke: 0.5pt + gray, width: 100%)[
*for each round* $t = 1, 2, ... T$:
  - sample clients $S_t subset.eq {1, ..., K}$
  - distribute $(w_t, c)$ to $S_t$
  - *for each client* $k in S_t$ *do in parallel*:
    - $w_k^1 arrow.l w_t$
    - *for* $e = 1, ..., E$ *do*:
      - compute the minibatch gradient $g_k (w_k^e)$
      - $w_k^(e+1) arrow.l w_k^e - eta (g_k (w_k^e) - c_k + c)$
    - $c_k^+ arrow.l g_k (w_t)$ or $c_k - c + 1/(E eta) (w_t - w_k^E)$
    - communicate deltas to server: $(Delta w_k, Delta c_k) arrow.l (w_k^E - w_t, c_k^+ - c_k)$
    - $c_k arrow.l c_k^+$
  - *on the server do*:
    - $(Delta w_t, Delta c_t) = 1/(|S_t|) sum_(k in S_t) (Delta w_k, Delta c_k)$
    - $w_(t+1) arrow.l w_t + eta Delta w_t$
    - $c_(t+1) arrow.l c_t + (|S_t|)/n Delta c_t$
]
#figure(
  image("../assets/scaffold.png", height: 150pt),
)



#remark()[
  During Covid-19  this was tested world-wide and it brought in an performance boost compared to the loacl model.
#figure(
  image("../assets/covid19.png", height: 150pt),
)

]

== Personalization Techniques
Do we want to have the same model everywhere? Somethimes it makes sense to let it adapt to local cirucumstances. 
To improve performance on heterogeneous data, models can be personalized:

Personalization Layers: Splitting the model into global layers (shared, blue) and local layers (private to each client).
#figure(
  image("../assets/personalization.png", height: 150pt),
)
If we now have a classification task, which part of the network should we make global and which make me local? The last layers are responsible for the classification so it would make sense to locallize the head of the model.

=== FedBN 
Keeping Batch Normalization parameters local to account for feature shifts turned out to work really well.
#figure(
  image("../assets/fedbn.png", height: 150pt),
)

=== Hypernetworks

What is this hypernetwork doing? It gives us client specific weights. We take a vector, embedding or whatever and feed it to a network and the network gives us weights for the network that we want to train. The output is then a regression problem because we want to predict continuous values and we have a huge output $arrow.r$ number of model parameters. So this is a hard task. The basic principle again is then that we give the predicted models from the server to the client and the client gives us then the weights back.

$ cal(L) = arg min_(phi, {v_k}_(k=1)^K) 1/K sum_(k=1)^K F_k (h_phi (v_k)) $
where $w_k = h_phi (v_k) $ and $v_k$ is the input we give the hypernetwork and $h_phi$ is the hypernetwork itself.

#block(fill: rgb("#f9f9f9"), inset: 10pt, radius: 4pt, stroke: 0.5pt + gray, width: 100%)[
Algorithm for training hypernetwork:
+ *for* each round $t = 1, 2, ..., T$ *do*
  + sample clients $S_t subset.eq {1, ..., K}$
  + set $w_k = h_phi (v_k)$ for all $k in S_t$ and $overline(w)_k = w_k$
  + *for* each $e in {1, ..., E}$ *do*
    + sample mini-batch $B$
    + $overline(w)_k^e arrow.l overline(w)_k - eta nabla_w F_k (B)$
  + $Delta w_k = overline(w)_k^E - w_k$ transfer to server
  + *aggregate at server:*
      $ phi = phi - alpha (nabla_phi h_phi (v_k)) Delta w_k $ 
      $ v_k = v_k - alpha (nabla_v h_phi (v_k)) Delta w_k $
  ]
== Privacy and Security in Federated Learning

Despite data staying local, FL is vulnerable to several attacks:
1. Inference Attacks: Inferring class representatives, membership, or even training samples from gradients (Deep Leakage from Gradients).
2. Malicious Server: A server using a GAN to reconstruct client data.
3. Poisoning Attacks: Backdoor or replacement attacks to manipulate the global model.

#example(title:"Inference of class representatives")[
  #figure(
  image("../assets/class_represenatives.png", height: 150pt),
 )
  #figure(
  image("../assets/class_represenatives2.png", height: 100pt),
 )

]
#example(title:"Inference of Training Samples and Labels")[
  #figure(
  image("../assets/training_samples.png", height: 180pt),
 )
]
Sharing the gradient can give away a lot of information.
#example(title:"Poison attack")[
  #figure(
  image("../assets/poison.png", height: 180pt),
 )
]
== Federated Learning with Differential Privacy (DP)

We want to make it harder for attackers to retrieve images from the client side. We can achieve this with differential privacy.

#figure(
  image("../assets/differential_privacy.png", height: 180pt),
 )

#definition(title: "differential privacy")[
A randomized mechanism (algorithm) $cal(M) : X^n -> RR$ satisfies $(epsilon, delta)$-DP if for all measurable sets $S subset.eq cal(R)$ and for any two adjacent datasets $D, overline(D) subset cal(X)^n$ (i.e., differing in one individual's data)

$ PP[cal(M)(D) in S] <= exp(epsilon) dot PP[cal(M)(overline(D)) in S] + delta, $

where $epsilon, delta > 0$.

*Note:* If $delta = 0$, $cal(M)$ is called pure $epsilon$-differential private.

- $M$: The "Mechanism" or algorithm (analysis/query) running on the data.
- $D$ and $overline(D)$: Two "adjacent" datasets. They are identical except one contains a specific individual's data (e.g., "Joe's Data") and the other does not.
- $S$: Any possible outcome of the analysis.
- $epsilon$ (Epsilon): The privacy budget.
  - A smaller $epsilon$ means higher privacy (probabilities are closer).
  - A larger $epsilon$ allows for more divergence, favoring utility over privacy.
- $delta$ (Delta): The "failure probability." The small chance the privacy guarantee might fail.
]



#definition(title: "Sensitivity")[
Let $W$ be a metric space with distance function $d_W (dot, dot)$. 
The sensitivity $S_W (h)$ of a function $h : X^n -> W$ is the amount that the function value varies when a single entry changes:

$ S_W (h) := sup_(w, overline(w) : d_cal(w)(w, overline(w)) = 1) d_W (h(w), h(overline(w))) $

$arrow.double$ Note the relation to the Lipschitz constant of a function.
]


=== Sensitivity Analysis
Assume the model weights $w_k$ are bounded: $||w_k|| <= C$.
Then, the sensitivity of the $k$-th client update in FedAVG is given by:

$ S_k &= sup_(P_k, overline(P)_k) || arg min_w F_k(w; P_k) - arg min_w F_k (w; overline(P)_k) || \
&= sup_(P_k, overline(P)_k) || arg min_w 1/(|P_k|) sum_(i in P_k) f_i (w) - arg min_w 1/(|P_k|) sum_(j in overline(P)_k) f_j (w) || \
&= 1/(|P_k|) sup_(P_k, overline(P)_k) || w_k - overline(w)_k || \
&= (2C) / (|P_k|) $

To ensure that the local training mechanism 
$ M_epsilon = [arg min_w 1/(|P_k|) sum_(i in P_k) f_i (w)] + n, $
where $n ~ N(0, sigma^2 I)$, preserves $(epsilon, delta)$-DP, we need to add noise with level:
$ sigma_k >= c dot S_k / epsilon $
where $c >= sqrt(2 ln(1.25/delta))$.


== FedAVG with DP Algorithm

#block(fill: rgb("#f9f9f9"), inset: 10pt, radius: 4pt, stroke: 0.5pt + gray, width: 100%)[
*For each round* $t = 1 ... T$:
1. *Sample client* $k in {1, ..., K}$
2. *Update the local weights:*
   $ w_k^t arrow.l arg min_w { F_k (w) + mu/2 ||w - tilde(w)^(t-1)||_2^2 } $
3. *Clip the local weights:*
   $ w_k^t arrow.l w_k^t / max(1, (||w_k^t||) / C) $
4. *Add noise:*
   $ tilde(w)_k^t arrow.l w_k^t + n_k^t, quad n_k^t ~ cal(N)(0, sigma_k) $
5. *Send to server*
6. *At server:*
   $ w^t = sum_(i=1)^k n_k / n w_k^t $
7. *The server broadcasts:*
   $ tilde(w)^t = w^t + n_s^t, quad n_s^t ~ cal(N)(0, sigma_s) $
 ]
#figure(
  image("../assets/FLwithDP.png", height: 180pt),
 )
