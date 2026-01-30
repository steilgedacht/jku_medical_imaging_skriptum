
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
  *Re-identification Risk*: A famous study by Sweeney (2000) showed that $87%$ of US citizens can be uniquely identified using only their ZIP code, birth date, and sex.
]

== From Centralized to Federated Learning


- Centralized ML: Training data from all sources is moved to a central server.
- Distributed On-Site Learning: Models are trained locally at each site with no information exchange.
- Federated Learning (FL): A collaborative learning approach where data remains at the source, and only model updates (weights) are shared with a central server.


=== Comparison: Centralized vs. Federated Learning
| Feature | Centralized | Federated |
| :--- | :--- | :--- |
| Data location | Cloud / Central server | Distributed nodes (edge) |
| Training | Primarily in the cloud | Primarily at the edge |
| Communication | Nodes share local data | Nodes share model weights |
| Privacy | Low user data privacy | High user data privacy |
| Heterogeneity | Cannot handle easily | Can operate on heterogeneous data |


== Centralized FL - Mathematical Formulation


Let $D = (x_i, y_i)^n_(i=1)$ be a dataset distributed to $K$ clients $C_k$ where $k in {1, ... , k}$. We denote by $P={1,... n}$ and each client has a subset $P_k$ such that $P= union.big_(k=1)^n P_k$. The goal is to solve:
$ min_w f(w) = min 1/n sum_(i=1)^N f_i(w) $
where $f_i(w)= l(x_i, y_i, w)$ is a loss function. Then we have that the total loss function 
$ f(w) = 1/n sum^n_(i=1) f_i(w) = sum^n_(i=1) 1/n f_i(w) = sum^K_(k=1) 1/n n_f F_k(w) $
with $F_k(w) = 1/n_k sum_(i in P_k) f_i(w)$ which is the loss at the distributed clients. So the loss function of the sample is the same, but now we combined the indices into the clients and then we write it by $n_k$. It is still the same thing, we just shifted the indices. At the client we do the same thing as globally. 

#remark()[
  *Iterative Learning Concept*:
  1. Central server chooses a model and transmits it to nodes.
  2. Nodes train the model locally with their own data.
  3. Nodes upload local updates to the server.
  4. Server pools results and generates a new global model.
]

=== Algorithms: FedSGD and FedAVG
- FedSGD: A simple version where each client performs one step of gradient descent per round.
- FedAVG: Substantially reduces communication by allowing clients to perform multiple local epochs before aggregating.

#theorem(title: "FedAVG Update Rule")[
  The server aggregates weights from a subset of sampled clients $S_t$:
  $ w_(t+1) arrow.r sum_(k in S_t) n_k/n w_(t+1)^k $
]


== Non-IID Data Challenges


In FL, data is typically not independent and identically distributed (Non-IID).
1. Feature distribution skew: Different demographics or devices ($P_k(x)$ varies).
2. Label distribution skew: Different distribution of labels ($P_k(y)$ varies).
3. Concept shift: Same feature, different labels (e.g., inter-reader variability).

*Solution*: SCAFFOLD uses control variables to correct for "client drift" caused by non-IID data.

== Personalization Techniques
To improve performance on heterogeneous data, models can be personalized:
- Personalization Layers: Splitting the model into global layers (shared) and local layers (private to each client).
- FedBN: Keeping Batch Normalization parameters local to account for feature shifts.
- Hypernetworks: Using a central network to predict personalized model parameters for each client based on their data distribution.

== Privacy and Security in FL


Despite data staying local, FL is vulnerable to several attacks:
1. Inference Attacks: Inferring class representatives, membership, or even training samples from gradients (Deep Leakage from Gradients).
2. Malicious Server: A server using a GAN to reconstruct client data.
3. Poisoning Attacks: Backdoor or replacement attacks to manipulate the global model.

== Federated Learning with Differential Privacy (DP)


#definition(title: "Differential Privacy")[
  A mechanism $M$ satisfies $(epsilon, delta)$-DP if for any two adjacent datasets $D, D'$ differing by one individual:
  $ P[M(D) in S] lt.eq e^epsilon P[M(D') in S] + delta $
]

#remark()[
  *Handwritten Sensitivity derivation for FedAvg*:
  The sensitivity $S_k$ of the update is given by the maximum change in weights. To ensure DP, updates must be:
  1. Clipped: $bar(w) = w / max(1, norm(w)_2 / C)$.
  2. Noised: Adding Gaussian noise $n tilde N(0, sigma^2 I)$ proportional to the sensitivity.
]


=== FedAVG with DP (Pseudocode)
- For each round $t$:
  - Sample clients $k$.
  - Clients update local weights $w^k$.
  - Clip local updates.
  - Add noise to the clipped updates.
  - Server aggregates noised weights and broadcasts the new global model.
