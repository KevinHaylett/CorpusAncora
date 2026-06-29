# Lesson P19-L — P vs NP and the Missing Axiom of Finite Measurement

**Lesson ID:** P19-L  
**Source paper:** P19  
**Title:** *P vs NP and the Missing Axiom of Finite Measurement: A Geofinitist Stress Test of the Clay Formulation*  
**Difficulty:** Advanced  
**Prerequisites:** P18-L (Axiom of Finite Representation — essential); ATT_08-L (Geofinitism — essential); P14-L (Admissibility and Measurement — strongly recommended); ATT_81-L (Functional Symbolic Trajectory — strongly recommended)  
**Estimated study time:** 90 minutes

---

## Learning Objectives

After completing this lesson you will be able to:

1. State the classical P vs NP problem and explain what the Clay formulation claims
2. Explain the FSM empirical separability reframing: what is being asked, and why it differs from the classical formulation
3. Construct the measured instance registry I_n ⊂ M^{d(n)} and explain why it replaces abstract Σ\*
4. Reconstruct TSP as an attractor-finding problem in symbolic phase space; define local and global attractors; explain why proof of optimality = global basin exclusion
5. Distinguish the four kinds of difficulty (search, proof, fragility, representation) and explain why the classical NP-hard label conflates them
6. State the FSM distinction between Construction and Verification as ontologically distinct trajectory types; write the formal expression
7. List the ten Geofinitist axioms for finite computation and identify which Clay fracture points each axiom exposes
8. State Bridge Axiom B and explain why its absence makes the classical formulation empirically underdetermined
9. Apply the Safety Layer discipline: given a symbolic trajectory, specify (C, α, H, δ) before beginning substantive work
10. Trace the Generonic Boundary flow and identify where FST parameters can be corrupted

---

## Key Idea 1 — Two Different Questions Inside One Problem

### The Classical Formulation

P vs NP asks: does P = NP?

- **P** is the class of decision problems solvable by a deterministic Turing machine in polynomial time
- **NP** is the class of decision problems whose solutions can be *verified* in polynomial time
- **The question**: since every problem in P is in NP, is the converse true — is NP ⊆ P?

The Clay formulation (Cook 2000) is posed over deterministic and nondeterministic Turing machines operating on Σ\* — the infinite set of all finite strings over a finite alphabet. The runtime is the number of head-moves as a function of input length n, asymptotically.

### The FSM Decomposition

P19 argues that the classical formulation contains two distinct questions that have been fused into one:

**Question A (formal):** In the formal model of Turing machines and language classes, do the symbolic structures of P and NP coincide?

**Question B (empirical):** In actual computation on finite physical systems, do solver and verifier processes exhibit robust scaling divergence over measured computational ranges?

These are different questions. A positive answer to A would say something important about the structure of formal symbolic systems. But it would not directly answer B — because the formal model operates on Σ\* (infinite, unmeasured) while actual computation operates on specific instances with specific resources and specific measurement protocols.

FSM proposes that the question worth asking empirically is:

$$\hat{T}_S(n) \stackrel{?}{\sim} \hat{T}_V(n) \,|\, (C, \alpha, H, \delta)$$

*Do the measured scaling trajectories of solvers and verifiers exhibit robust finite separability?*

This is narrower than the classical question but makes contact with the physical world.

### Exercise 1.1

(a) "P vs NP is not one question — it is two questions that have been fused." Is this fusion a mistake, or was it deliberate? Why might a mathematician formulating the problem in 1971 have chosen to work with the formal (Question A) framing rather than the empirical (Question B) framing? What tools were unavailable in 1971 that FSM relies on?

(b) P19 says "even a positive resolution (P = NP) would demonstrate formal language-class equivalence, not measured computational equivalence." Explain this claim. If someone proved that P = NP, would it tell you anything about how long it takes to solve TSP on a real computer with 1000 cities? Why or why not?

(c) The **measured instance registry** I_n ⊂ M^{d(n)} replaces abstract Σ\* with finite measured objects. What does d(n) represent? What does M represent? Write out what the registry contains for a specific problem — say, SAT with n variables — and contrast it with what Σ\* contains.

---

## Key Idea 2 — TSP as Attractor-Finding

### The Classical View

Travelling Salesman Problem: find the minimum-length Hamiltonian cycle over n cities. Classically: NP-hard for the optimisation version, NP-complete for the decision version. The paradigm hard problem.

### FSM Reconstruction

P19 reconstructs TSP as a **dynamical system in symbolic phase space**. The reconstruction has three components:

**Phase space**: Tour space S_n — the finite set of all Hamiltonian cycles over n nodes — is the FST phase space of the problem. Every candidate tour is a point in S_n.

**Dynamics**: A tour evaluation and search procedure defines a dynamical system (S_n, L, Φ), where L: S_n → ℝ is the length functional and Φ: S_n → S_n is the search trajectory update.

**Attractor structure**: Under Φ, tours with low L-values are attractors:
- **Local attractors**: tours that are locally length-minimising (no single move reduces L)
- **Global attractors**: globally optimal tours (no tour anywhere in S_n has lower L)

TSP is therefore: *find the global attractor of (S_n, L, Φ).*

### Proof of Optimality = Global Basin Exclusion

This is the key insight of Chapter 2. To prove that tour T\* is optimal is not just to exhibit T\* and measure its length. It is to demonstrate that the entire remaining phase space cannot contain a lower-length tour. This requires **excluding all other basins of attraction**.

Formally: optimality gap G(I) ∼ 0 | (C, α, H, δ)

The trajectory certifying this is a **second-order FST** — a trajectory about the space of trajectories. It must:
1. Characterise the full attractor landscape of (S_n, L, Φ)
2. Demonstrate that T\*'s basin is globally minimal

This is why TSP proofs are hard: they are not just search problems, they are **trajectory certification problems**.

### Four Kinds of Difficulty

P19 separates what the NP-hard label conflates:

| Difficulty type | What it measures | Example in TSP |
|---|---|---|
| **Search difficulty** | Cost of finding a good solution in S_n | Exponential tour-space search |
| **Proof difficulty** | Cost of certifying global optimality | Global basin exclusion argument |
| **Fragility difficulty** | Sensitivity to initial conditions / measurement noise | Different heuristic seeds give different local optima |
| **Representation difficulty** | Cost of expressing and transmitting the solution in finite symbolic form | Encoding the optimal tour unambiguously for a specific receiver |

These four are **logically independent**. A problem can be search-easy but proof-hard (finding a good tour is fast; proving it is optimal is not). A problem can be fragile without being hard in the search sense.

### Exercise 2.1

(a) "The TSP proof is a second symbolic trajectory — a trajectory about the space of trajectories." What does this mean practically? If you have found a tour T\* with length L(T\*) = 1247, what further work is required to convert this into a proof of optimality? What lower-bound machinery is required?

(b) Distinguish a local attractor and a global attractor in TSP. What does it mean for a search algorithm to get stuck in a local attractor? Give a concrete example with a small number of cities (say, 5).

(c) Apply the **measured research protocol** to a specific TSP instance. The protocol records A(I) = (v_T(I), ε_T(I), U(I), B_lower(I), P_A). For a 50-city TSP instance solved with simulated annealing, specify what each of these five components would contain. What is ε_T(I), the measurement uncertainty, for a deterministic algorithm vs a randomised algorithm?

---

## Key Idea 3 — Construction ≁ Verification

### The Classical Framing

NP is defined via verification: a problem is in NP if, given a solution (witness), a deterministic polynomial-time verifier can confirm its correctness. The P vs NP question asks whether the *existence of a fast verifier* implies the *existence of a fast solver*. The classical formulation treats these as questions about Turing machine complexity classes over the same formal model.

### The FSM Ontological Distinction

P19 establishes that solving and verifying are **not merely different in cost — they are different in kind**:

**Solving (endogenous construction):**
- The trajectory must generate the solution from internal symbolic resources
- The outcome is not given in advance
- The trajectory must reach and stabilise in a previously unoccupied region of S_n
- Construction is an act of symbolic generation — the trajectory creates the witness

**Verification (exogenous measurement):**
- The trajectory begins with a candidate solution s given from outside
- It measures s against the problem constraints — a trajectory through known, finite measurement space
- The question is whether s satisfies P: a measurement decision
- Verification does not construct; it compares

Formally:

$$\text{Construction} \not\equiv \text{Verification}$$

$$\text{Solve}(I) \stackrel{?}{\sim} \text{Verify}(I, s) \,|\, (C, \alpha, H, \delta)$$

The `?∼` asks: are the two trajectory types finitely separable under measured conditions? This is an empirical question, not a formal one.

### Six Missing Distinctions in the Classical Formulation

P19 identifies six distinctions the classical formulation does not make but that FSM requires:

1. **Construction vs measurement** — endogenous trajectory generation vs exogenous comparison
2. **Formal complexity vs measured complexity** — Turing machine step counts vs measured physical resource consumption
3. **Worst-case vs typical-case vs measured-range trajectory** — asymptotic analysis vs empirical distribution over actual instances
4. **Single Turing machine vs finite measured computational system** — abstract model vs physically realised system with (C, α, H, δ)
5. **Infinite Σ\* vs finite measured instance space** — formal string space vs measured instance registry I_n
6. **Asymptotic scaling vs empirically observed scaling trajectory** — O(n^k) as n→∞ vs measured trajectory T̂(n) over actual n range

### Exercise 3.1

(a) "Construction is endogenous; verification is exogenous." A chess engine solving a position vs a human checking the engine's move. Identify which is construction and which is verification. Is the verification simpler? Always? Does the FSM framing explain why checking moves is easier than finding them?

(b) P19 says "even if Solve(I) and Verify(I,s) happen to cost the same for particular instances, they are not definitionally equivalent." Give an example from mathematics or computation where construction and verification cost the same — and one where they differ dramatically. What does the FSM framing add to the classical observation that "verification is easy"?

(c) Write the formal FSM expression for the P vs NP empirical separability question. What would "finite separability confirmed" mean operationally — what data would you need, and how would you measure it?

---

## Key Idea 4 — Ten Axioms and Fifteen Fracture Points

### The Ten Geofinitist Axioms

P19 states ten axioms that FSM requires any formulation of a computational complexity problem to satisfy. Read them not as rigid rules but as diagnostic questions: *where does the classical formulation fail each axiom?*

| Axiom | Name | Core claim |
|---|---|---|
| A1 | Finite Symbol | Every symbol is a finite, measured, instantiated mark |
| A2 | Symbolic Extent | Every computation occupies bounded symbolic extent at every step |
| A3 | Finite Claim | Every complexity claim is conditioned on (C, α, H, δ) |
| A4 | Proof-as-Trajectory | Every proof is an FST that must be admissible |
| A5 | Verification-as-Measurement | Verification is exogenous measurement, not an abstract predicate |
| A6 | Measurement Uncertainty | Every measurement carries δ > 0; exact membership predicates are limiting idealisations |
| A7 | Infinite Commitment | Claims ranging over Σ\* or all n ∈ ℕ are formal/useful-fiction claims, not measured claims |
| A8 | Initialisation | Every FST must be explicitly initialised (C, α, H, δ stated) before the trajectory begins |
| A9 | Bridge Axiom B | Claims connecting formal complexity classes to measured computation must supply the bridge B |
| A10 | Missing Measurement | Classical P vs NP violates A1–A9; it is empirically underdetermined |

### Bridge Axiom B — The Missing Axiom

The most important axiom is A9, Bridge Axiom B. It asks: *what is the mapping from the formal symbolic space (Σ\*, Turing machine step counts) to the measured physical space (actual computation on finite systems with specific resources)?*

The Clay formulation does not supply B. Therefore:

$$\text{P vs NP} \,|\, (C, \alpha, H, \delta, \mathbf{B})$$

is the correct FSM statement, where B is absent in the classical version.

This does not make P vs NP meaningless. It makes it a **Pillar P4 (Useful Fiction) claim** — a formal model whose resolution illuminates structure without measuring reality directly. A proof that P = NP or P ≠ NP would be a major result in the formal theory of computation. It would not, by itself, tell us whether measured solver trajectories are faster than measured verifier trajectories on real computational systems.

### Selected Fracture Points

Three of the fifteen fracture points illustrate the structure of the full audit:

**Fracture 3 — Non-termination as infinite time**: The Clay formulation sets t_M(w) = ∞ when the machine does not halt on input w. But actual computation cannot run for infinite time. This is an infinite commitment disguised as a runtime. FSM requires that non-termination be flagged as an admissibility condition, not assigned an infinite value.

**Fracture 9 — Asymptotic conflation**: O(n^k) notation hides (a) the range of n over which the bound holds, (b) the constant factor, and (c) the measurement uncertainty at each n. Two algorithms that are both O(n^2) may have measured trajectories that diverge dramatically for n in [10, 1000]. Asymptotic class membership does not determine measured behaviour.

**Fracture 14 — Bridge Axiom B absent**: No mechanism is specified in the Clay formulation for connecting polynomial-time Turing machine computation to polynomial-time computation on any physical device. The bridge is assumed to be unproblematic. FSM requires it to be stated explicitly.

### Exercise 4.1

(a) Apply Axiom A7 (Infinite Commitment) to the statement "no polynomial-time algorithm exists for TSP." What does this claim range over? Is it a formal claim, an empirical claim, or a claim that bridges both? How does A9 (Bridge Axiom B) interact with it?

(b) Fracture point 4 is "universal polynomial bound 'for all n'." Explain why the ∀n quantifier creates a problem for FSM. What should replace it in a Geofinitist complexity claim?

(c) "A proof that P = NP would be a major result in formal computation theory but would not resolve the measured empirical question." A sceptic might say: "In practice, if P = NP, someone would find a polynomial-time algorithm for NP-complete problems, and that would tell us everything we need to know." Respond to this sceptic using FSM's framework.

---

## Key Idea 5 — The Safety Layer and Explicit Initialisation

### The Discipline

The Safety Layer is the practical manifesto of P19. Its core principle:

> **Explicit initialisation of the FST — stating (C, α, H, δ) before substantive motion begins — is the first act of the trajectory, the condition that makes the rest of the trajectory interpretable.**

An uninitialised FST is not necessarily wrong. It may produce valid results by accident, or because the reader supplies the missing parameters implicitly. But it cannot be trusted, because there is no explicit record of what the trajectory was tracking.

### What Initialisation Prevents

An uninitialised FST may:
- **Import hidden assumptions**: prior trajectories carry their (C, α, H, δ) forward invisibly
- **Blend registers**: formal, empirical, speculative, and normative claims may appear in the same trajectory without demarcation
- **Generate undisclosed commitments**: conclusions appear to follow from premises but depend on unstated representational conditions
- **Fail to specify the receiver**: who is this trajectory for? Under what consensus conditions is it admissible?

These failures produce **drift**: the trajectory appears coherent but is carrying undisclosed structure.

### The Four Parameters in Practice

Before beginning any complex symbolic task — a proof, a research paper, a software design, a model build — FSM requires:

- **C (consensus field)**: state explicitly which community's standards, commitments, and definitions govern this work. (Example: "This analysis operates within classical probability theory as standardly interpreted in the statistics literature up to 2025.")
- **α (alphonic limit)**: state the finest distinctions being made. (Example: "We distinguish p < 0.05 from p < 0.01 but do not subdivide further.")
- **H (provenance)**: state the conceptual ancestry of the key terms. (Example: "Complexity is used in the Cook–Karp sense, not the Kolmogorov sense.")
- **δ (uncertainty)**: state the uncertainty imported into this trajectory from measurement, estimation, or prior inference. (Example: "All running times are measured to within ±5% on the benchmark hardware specified in §2.")

### LLM Implications

The Safety Layer includes an extended essay on language model implications. Language models are FST engines without explicit initialisation: they generate trajectories without stating (C, α, H, δ) and blend registers freely. Their provenance — the training distribution — is opaque to the user.

The corrective is not to distrust LLM output categorically, but to apply the Safety Layer discipline as the human user: before using LLM-generated content in consequential reasoning, supply the initialisation that the model did not.

### The Generonic Boundary Flow

The full cycle from physical event to symbolic trajectory:

> Physical interaction → Transducer → Generonic Boundary → Finite symbol → Reference comparison → Measurement → Trajectory stabilisation → Rules, proofs, models → New exogenous measurement → *(cycle repeats)*

Each arrow is a place where (C, α, H, δ) must be maintained or the trajectory loses traceability. The Safety Layer is the discipline of keeping the parameters explicit at each crossing.

### Exercise 5.1

(a) "Initialise the FST before beginning." Take a research claim you are familiar with from your field. Write an explicit (C, α, H, δ) initialisation for it. What does the initialisation reveal about assumptions you usually leave implicit?

(b) "LLM output is a trajectory without stated initialisation parameters." When you receive output from a language model, what default values of (C, α, H, δ) might the model be operating with? How would you supply your own initialisation before using the output in consequential work?

(c) Identify one arrow in the Generonic Boundary flow — from physical interaction to finished symbolic trajectory — where a real-world error (a measurement artefact, a transcription mistake, a misleading summary) corrupted the trajectory. Describe what happened in FSM terms: which parameter was distorted and how did the corruption propagate?

---

## Synthesis — What P19 Contributes to the School

P19 is the School's direct engagement with the most celebrated open problem in computer science, and it stands as the fullest demonstration of what it means to apply the FSM lens to a problem that has resisted solution for over fifty years.

**The main contribution is not an answer to P vs NP.** It is a **re-diagnosis**: the classical formulation is empirically underdetermined because it lacks Bridge Axiom B. Before the formal question can be answered in a way that has implications for real computation, the bridge must be supplied. This is not a limitation of the Clay formulation — it is a deliberate abstraction. But abstracting the bridge away means the formal resolution leaves the empirical question untouched.

**The second contribution is structural**: the four-kinds-of-difficulty decomposition, the construction/verification ontological distinction, and the ten Geofinitist axioms are tools that apply to any formal complexity claim, not just P vs NP. They constitute a measurement framework for computational complexity that can be used independently of the P vs NP question.

**The third contribution is the Safety Layer**: the most directly applicable piece of the corpus for practitioners — researchers, engineers, model builders — who work in symbolic systems every day. The discipline of explicit FST initialisation is the operational form of FSM at work.

P19 sits at the intersection of P18 (foundations of mathematics), P14 (admissibility and measurement), and ATT_81 (the FST framework). Together, these four papers form the formal core of the School's engagement with the limits of symbolic systems.

---

## Consolidation Questions

1. P19 proposes that classical P vs NP is a **Useful Fiction (P4)** claim rather than an empirical claim. Does this downgrade its importance? Can a formal question illuminate empirical reality without being an empirical claim itself? Give an example from physics or mathematics where a formal result changed empirical practice.

2. "Proof of optimality = global basin exclusion." This is also, in FSM terms, a proof that no admissible trajectory can reach a lower point. How does this relate to P18's admissibility condition for PEM? Is the TSP optimality proof a case where PEM is admissible or inadmissible?

3. The Safety Layer says that explicit (C, α, H, δ) initialisation is the first act of any serious trajectory. But many great mathematical results were discovered informally, without explicit initialisation. Did Euler, Gauss, or Ramanujan initialise their trajectories? Is P19 claiming that uninitialised trajectories cannot produce valid mathematics — or something subtler?

4. P19 identifies 15 fracture points in the Clay formulation. Is this a critique of the Clay Prize, or a critique of the mathematical tradition in which the Clay Prize operates? What would a Clay Prize formulation look like that satisfied all ten Geofinitist axioms?

5. "The Safety Layer is the first act of the trajectory." In everyday research, writing an explicit (C, α, H, δ) initialisation adds overhead. Is this overhead justified? Under what circumstances would you *not* initialise explicitly — and what are the risks?

---

## Further Reading

- **P18-L** (Axiom of Finite Representation) — the prior axiom; P19's Bridge Axiom B is its extension to formal complexity theory
- **P14-L** (Admissibility and Measurement) — the measurement framework; P19's fracture-point audit applies P14's admissibility concept to the Clay formulation
- **ATT_81-L** (Functional Symbolic Trajectory) — the FST framework P19 applies throughout; essential for Key Ideas 2 and 3
- **ATT_08-L** (Geofinitism) — the five pillars; P19's opening chapter applies all five to complexity theory
- **ATT_38** (The Generonic Boundary) — the Generonic Boundary flow appears in the Safety Layer; ATT_38 develops it as a standalone concept
- **ATT_27** (Alphonic Logic) — the α (alphonic limit) parameter; the finest distinctions and how they are managed
- **ATT_28** (Commitment, Admissibility) — the admissibility framework; P19's Axiom A4 (Proof-as-Trajectory) applies it to computational proofs
- **P12** (Trajectory-Based Computation) — establishes computation as trajectory; P19 extends this to complexity classes

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
