# Summary — P19: P vs NP and the Missing Axiom of Finite Measurement

**Paper ID:** P19  
**Title:** *P vs NP and the Missing Axiom of Finite Measurement: A Geofinitist Stress Test of the Clay Formulation*  
**Author:** Kevin R. Haylett, Manchester, UK  
**Date:** June 2026  
**Length:** 47 pages  
**Status:** Working draft (chapter collection)  
**Primary College:** College of Attralucian Studies  
**Secondary Colleges:** College of Philosophy; College of Finite Symbolic Mechanics; College of Finite Measurements  
**Primary Pillars:** P5 (Finite Reality), P2 (Approximations / Measurements), P4 (Useful Fiction)  
**Secondary Pillars:** P3 (Dynamic Flow), P1 (Geometric Container)

---

## Abstract

P19 applies the Finite Symbolic Mechanics (FSM) framework to P vs NP — the most celebrated open problem in computer science. It argues that the Clay Mathematics Institute formulation of the problem omits a foundational axiom: the **Missing Axiom of Finite Measurement**, designated **Bridge Axiom B**. Without B, the classical formulation operates on an infinite symbolic space (Σ\*) that has never been measured, treating algorithmic complexity as if it were a timeless formal property rather than an observable property of finite, initialised, resource-bounded computation.

The paper re-expresses P vs NP as an empirical separability question — *do solver and verifier trajectories exhibit robust finite separability over measured computational ranges?* — and demonstrates that this reframing, while narrower than the classical formulation, is the only version of the question that makes contact with the physical world. The paper presents four chapters and a Safety Layer essay; its principal contributions are: (1) a Geofinitist reconstruction of complexity classes as measured trajectory families; (2) a reanalysis of the Travelling Salesman Problem (TSP) as an **attractor-finding problem in symbolic phase space**; (3) a systematic audit of 15 fracture points in the Clay formulation; (4) a set of 10 Geofinitist axioms for finite computation; and (5) a Safety Layer specifying the discipline of explicit FST initialisation.

---

## Chapter 1 — The P vs NP Problem: A Geofinitist Lens

### Classical Background

P vs NP asks whether every problem whose solution can be *verified* in polynomial time can also be *solved* in polynomial time. The Clay Millennium formulation (Cook 2000) is the canonical statement. It is phrased over Turing machines operating on the infinite symbolic alphabet Σ\*, with runtime measured by the number of computation steps as a function of input length.

### FSM Reframing

FSM treats computation as a trajectory through measured symbolic space. A computation is not just a Turing machine transition function — it is a **Functional Symbolic Trajectory (FST)**:

$$T_S \sim \langle s_0 \xrightarrow{r_1} s_1 \cdots \xrightarrow{r_n} s_n \,|\, \alpha, \delta, H, C, B, K \rangle$$

where α is the alphonic limit, δ is measurement uncertainty, H is provenance, C is consensus, B is the base/representation, and K is cost.

The FSM reformulation introduces:
- A **measured instance registry** I_n ⊂ M^{d(n)}: instances are treated as finite measured objects, not abstract strings
- **Finite separability**: the empirical question *do solver and verifier trajectories exhibit robust divergence over measured computational ranges?*
- Complexity classes as **regions in measured scaling space** rather than as formal language-membership predicates

Under this lens, the classical formulation conflates two distinct questions: (i) a formal question about infinite language classes and (ii) an empirical question about the measurable cost of finite computation. P19 is primarily concerned with (ii) and argues that (i) cannot be answered empirically — only formally — and that the formal answer, if one exists, will not resolve the empirical question about actual computation.

### Five Pillars Applied

The paper applies FSM's five pillars to complexity:
- **P1 (Geometric Container):** computation space S_n as finite phase space; symbolic dynamics as geometrically structured
- **P2 (Approximations/Measurements):** all complexity statements are conditional on measurement protocol (C, α, H, δ)
- **P3 (Dynamic Flow):** computation as trajectory evolution; attractor structure as the object of study
- **P4 (Useful Fiction):** infinite Σ\* as a useful analytical fiction; clock complexity as a model tool, not a physical observable
- **P5 (Finite Reality):** physical computation is always finite; any axiom system for complexity must accommodate this

---

## Chapter 2 — The TSP as an Attractor-Finding Problem

### Classical TSP

The Travelling Salesman Problem: given n cities with pairwise distances, find the shortest tour visiting each city exactly once. TSP is NP-hard; its decision version is NP-complete. It is the paradigm case of combinatorial optimisation and appears throughout complexity theory as a reference problem.

### FSM Reanalysis

P19 reconstructs TSP as a **dynamical system in symbolic phase space**:

- Tour space S_n is the finite set of all Hamiltonian cycles over n nodes — the **FST phase space** of the problem
- A tour evaluation procedure defines a **dynamical system (S_n, L, Φ)**, where L is the length functional and Φ is the search trajectory
- **Local attractors** are tours that are locally length-minimising; **global attractors** are optimal tours
- The TSP is then: *find the global attractor of (S_n, L, Φ)*

### Optimality as Global Basin Exclusion

A key result: **proof of optimality = global basin exclusion**. To prove that tour T\* is optimal is to demonstrate that no other basin of attraction can be lower — that is, to construct a symbolic trajectory that:

1. Characterises the full attractor landscape
2. Shows that T\*'s basin is unique at the minimum

This means:

> *The TSP proof is a second symbolic trajectory — a trajectory about the space of trajectories.*

Formally: the optimality gap G(I) ∼ 0 | (C, α, H, δ) — it is not that the gap is literally zero, but that the FST representation certifies that no measured trajectory can lie below it under the specified parameters.

### Four Kinds of Difficulty

P19 identifies four distinct kinds of difficulty that cluster under the NP-hard label but are conceptually separable:
1. **Search difficulty**: the cost of finding a good solution in S_n
2. **Proof difficulty**: the cost of certifying that a solution is globally optimal
3. **Fragility difficulty**: the sensitivity of the search trajectory to small perturbations in initial conditions or measurement
4. **Representation difficulty**: the cost of expressing and transmitting the solution in finite symbolic form

Classical complexity theory does not distinguish these four. The FSM framework separates them and demands that any complexity claim specify which kind of difficulty it is addressing.

### The Measured Research Protocol

P19 proposes a measured research protocol for TSP instances: record A(I) = (v_T(I), ε_T(I), U(I), B_lower(I), P_A), where:
- v_T(I) is the best tour length found under trajectory T
- ε_T(I) is measurement uncertainty
- U(I) is the computed upper bound
- B_lower(I) is the best lower bound
- P_A is the algorithm provenance (parameters, initialisation, random seed)

This protocol applies to any combinatorial optimisation problem and makes the empirical content of complexity claims explicit.

### Two Layers of the P vs NP Problem

P19 argues there are **two distinct layers** of P vs NP:
- **Classical (formal) layer**: P ?= NP — the formal language-class question over Turing machines and Σ\*
- **Measured (empirical) layer**: T̂_S(n) ?∼ T̂_V(n) | (C, α, H, δ) — do measured solver and verifier scaling trajectories exhibit robust divergence?

These are different questions. A solution to the formal layer need not resolve the empirical layer; the empirical layer is the one with operational relevance for computational science.

---

## Solving, Verification, and the Measurement Boundary

This unnumbered chapter establishes the ontological distinction between *solving* and *verifying* as the conceptual core of P vs NP.

### Classical Formulation's Conflation

The classical formulation phrases both P (polynomial solvability) and NP (polynomial verifiability) over the same Turing machine model with the same asymptotic clock. This conflates:
1. The **construction problem** (build a solution where none was known)
2. The **measurement problem** (confirm a solution whose structure is already given)

### The FSM Distinction

FSM treats the two as ontologically distinct trajectory types:
- **Solving** is **endogenous construction**: the trajectory must generate the solution from internal symbolic resources; the outcome is not given in advance; the trajectory must stabilise in a new region of S_n
- **Verification** is **exogenous measurement**: the trajectory begins with a candidate solution s and measures it against the problem constraints; the question is whether s satisfies P — a trajectory through known, finite measurement space

The formal expression:

$$\text{Construction} \not\equiv \text{Verification}$$

$$\text{Solve}(I) \stackrel{?}{\sim} \text{Verify}(I, s) \,|\, (C, \alpha, H, \delta)$$

The `?∼` notation: the question is whether the two trajectory types are finitely separable under measured conditions. The `≁≡` notation: they are not definitionally equivalent, even if they happen to cost the same for particular instances.

### Six Missing Distinctions in the Classical Formulation

P19 identifies six distinctions the classical formulation does not make:
1. Construction vs measurement (endogenous vs exogenous)
2. Formal complexity vs measured complexity
3. Worst-case vs typical-case vs measured-range trajectory
4. Single Turing machine vs finite measured computational system
5. Infinite Σ\* vs finite measured instance space
6. Asymptotic scaling vs empirically observed scaling trajectory

Each distinction corresponds to a fracture point in the Clay formulation (catalogued in Chapter 3).

---

## Chapter 3 — The P vs NP Challenge and the Missing Axiom of Measurement

### Ten Geofinitist Axioms for Finite Computation

P19 states ten axioms that FSM requires any formulation of a computational complexity problem to satisfy:

1. **Finite Symbol Axiom**: every symbol is a finite, measured, instantiated mark
2. **Symbolic Extent Axiom**: every computation occupies bounded symbolic extent at every step
3. **Finite Claim Axiom**: every complexity claim is conditioned on (C, α, H, δ)
4. **Proof-as-Trajectory Axiom**: every proof is a Functional Symbolic Trajectory that must be admissible
5. **Verification-as-Measurement Axiom**: verification is an exogenous measurement process, not an abstract predicate
6. **Measurement Uncertainty Axiom**: every measurement carries δ > 0; exact membership predicates are limiting idealisations
7. **Infinite Commitment Axiom**: any claim ranging over Σ\* or all n ∈ ℕ is a formal/useful-fiction claim, not a measured claim
8. **Initialisation Axiom**: every FST must be explicitly initialised (C, α, H, δ stated) before the trajectory begins
9. **Bridge Axiom B**: any claim connecting formal complexity classes to measured computational behaviour must supply the bridge between the formal symbolic space and the measured physical system — this is the Missing Axiom
10. **Missing Measurement Axiom**: the classical formulation violates axioms 1–9; P vs NP as classically stated is therefore an **underdetermined problem** from the FSM perspective

### Fifteen Fracture Points in the Clay Formulation

The chapter audits the Clay formulation against the ten axioms and identifies 15 fracture points:

1. **Infinite alphabet to Σ\***: finite alphabets are extended to infinite string spaces without measurement protocol
2. **Acceptance as membership**: t_M(w) is defined as a step count but collapses to a binary membership predicate
3. **Non-termination as infinite time**: t_M(w) = ∞ for non-accepting paths — an infinite commitment disguised as a runtime
4. **Universal polynomial bound "for all n"**: the ∀n quantifier ranges over all natural numbers; no measured n is specified
5. **Existential witness vs construction**: NP witness existence asserted without construction protocol
6. **Polynomial reduction ≤_p**: reductions are formal language maps, not measured trajectory transformations
7. **Clock as model**: the step-count clock is a formal model, not a measured physical time
8. **Machine universality**: the universal Turing machine abstracts away from specific computational substrates and their measurement properties
9. **Asymptotic conflation**: O(n^k) hides the range of n over which the bound holds and the constant factor
10. **Oracle conflation**: relativised complexity classes mix formal and measured properties
11. **Witness completeness**: the NP witness is existentially quantified; its provenance and constructability are unspecified
12. **Alphabet size independence**: no measurement of the cost of the alphabet itself
13. **No initialisation**: the Turing machine is not initialised (C, α, H, δ not specified)
14. **Bridge Axiom B absent**: no mechanism specified for connecting formal classes to measured computation
15. **Classical solution would solve only the formal basin**: even a positive resolution (P = NP) would demonstrate formal language-class equivalence, not measured computational equivalence

### The Missing Axiom of Measurement

The Missing Axiom is Bridge Axiom B: a specification of the mapping from the formal symbolic space of Turing machine complexity to the measured physical space of actual computation. Without B:

$$\text{P vs NP} \,|\, (C, \alpha, H, \delta, \mathbf{B})$$

is the correct FSM statement. Classical P vs NP omits B. Therefore the classical formulation makes claims that the FSM framework cannot certify as physically grounded.

The chapter argues this does not make P vs NP meaningless — it is a powerful and well-posed formal question — but it makes it an instance of **Pillar P4 (Useful Fiction)**: a model whose answer illuminates structure without directly measuring reality.

---

## The Safety Layer — Explicit Initialisation of Functional Symbolic Trajectories

The Safety Layer essay is a practical manifesto for working within the FSM framework. Its central principle:

> **Explicit initialisation of the FST (C, α, H, δ) is not optional — it is the first act of the trajectory, the condition that makes the rest of the trajectory interpretable.**

### Why Initialisation Matters

An uninitialised FST may:
- Import assumptions from previous trajectories without disclosure
- Blend registers (formal, empirical, speculative) without demarcation
- Generate conclusions that appear to follow from premises but depend on unstated representational conditions
- Fail to specify the receiver for whom the trajectory is constructed

These failures do not always produce contradiction. They produce **drift**: a trajectory that appears coherent but is carrying undisclosed commitments.

### Four Explicit FST Parameters

Before any substantive move in a symbolic trajectory, FSM requires:
- **C (consensus field)**: which community, which commitments, which shared symbolic conventions govern this trajectory
- **α (alphonic limit)**: the finest distinctions the system can make — the resolution floor
- **H (provenance)**: the historical path by which the concepts being used acquired their current meaning
- **δ (uncertainty)**: the measurement uncertainty carried into this trajectory from prior steps

### Implications for LLMs and AI Systems

The Safety Layer includes an essay on LLM implications. Language models operate as FST engines without explicit initialisation: they do not state (C, α, H, δ) before beginning, they blend registers freely, and their provenance is opaque. The Safety Layer essay argues that:
- LLM output should be understood as a trajectory without stated initialisation parameters
- Users who consume LLM output without supplying their own initialisation are importing unknown (C, α, H, δ) from the training distribution
- The discipline of explicit initialisation, applied by the human user, is the corrective

### The Generonic Boundary Flow

The essay concludes with a diagram of the **Generonic Boundary flow** — the full cycle from physical event to symbolic trajectory:

> Physical interaction → Transducer → Generonic Boundary → Finite symbol → Reference comparison → Measurement → Trajectory stabilisation → Rules, proofs, models → New exogenous measurement

Each arrow in this flow is a place where (C, α, H, δ) can be corrupted if not maintained. The Safety Layer is the discipline of keeping the parameters explicit at each crossing.

---

## Key Contributions

1. **Empirical separability reframing**: P vs NP is a question about finite trajectory families, not formal language classes — the two questions are distinct and must be addressed separately
2. **TSP as attractor-finding**: proof of optimality = global basin exclusion = second-order symbolic trajectory; four kinds of difficulty separated
3. **Missing Axiom of Finite Measurement (Bridge Axiom B)**: identified as the axiom absent from the Clay formulation; its absence means classical P vs NP is formally well-posed but empirically underdetermined
4. **15 fracture points in Clay**: systematic audit showing where each of the ten Geofinitist axioms is violated
5. **Construction ≁≡ Verification**: established as an ontological distinction, not merely a computational one
6. **Safety Layer**: explicit FST initialisation articulated as a discipline with implications for AI systems and research practice
7. **Generonic Boundary flow**: integrates the paper's framework into the broader FSM architecture

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
