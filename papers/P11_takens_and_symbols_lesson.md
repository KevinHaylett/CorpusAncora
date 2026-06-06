# Lesson P11-L — Takens' Theorem Applies to Discrete Symbol Sequences

**Lesson ID:** P11-L  
**Source paper:** P11  
**Title:** *Takens' Theorem Applies to Discrete Symbol Sequences: A Formal Note on Language as a Dynamical System*  
**Difficulty:** Intermediate  
**Prerequisites:** P01-L (TBT/MARINA architecture), P04-L (Finite-Symbol Embedding Theorem), basic familiarity with dynamical systems  
**Estimated study time:** 45 minutes

---

## Learning Objectives

After completing this lesson you will be able to:

1. State precisely where the smoothness conditions in Takens' theorem apply — and where they do not
2. Explain why discrete and symbolic sequences are legitimate inputs to Takens delay embedding
3. Construct the formal chain from continuous dynamical system → smooth measurement function → symbolic quantization → delay embedding
4. Describe what "topological conjugacy" means in this context and why it is sufficient for practical applications
5. Position P11 within the Geofinitist programme as the formal mathematical licence for the TBT architecture

---

## Key Idea 1 — Where Smoothness Lives (and Where It Does Not)

### The Setup

Takens' theorem says: given a smooth dynamical system ϕ: M → M on a compact manifold M of dimension d, and a smooth measurement function h: M → ℝ, the delay embedding map

$$\Phi_{h,\phi}(x) = \bigl(h(x),\, h(\phi(x)),\, h(\phi^2(x)),\, \ldots,\, h(\phi^{2d}(x))\bigr)$$

is an embedding (injective, with continuous inverse) from M into ℝ^(2d+1).

### The Three Smoothness Conditions

| Condition | Required? | What it means |
|---|---|---|
| M is a differentiable manifold | ✓ Yes | The state space has a well-defined geometry |
| ϕ is smooth | ✓ Yes | The dynamics evolve continuously |
| h: M → ℝ is smooth | ✓ Yes | The measurement function varies continuously with state |
| The observed sequence {yₜ} is smooth | ✗ No | This is never stated in the theorem |

### The Source of the Confusion

Most textbook examples use real-valued measurements — temperature from a thermometer, voltage from a circuit. The *data* varies smoothly in these examples, which creates the false impression that the theorem requires smooth data. It does not. The measurements are whatever the function h produces when applied to the continuous state. If h rounds, quantizes, or assigns categories, the output is discrete — and the theorem still applies.

> "The sequence {h(ϕᵗ(x₀))} is a set of discrete samples. It can be real-valued, integer-valued, or categorical. The theorem's proof uses the smoothness of ϕ and h to guarantee that nearby states yield nearby measurement sequences — but the measurements themselves are just numbers (or symbols) at discrete times."

**Worked example:** A thermometer reads to the nearest degree. The output sequence 18, 19, 19, 20, 20, 21... is discrete integers. Nobody objects that this sequence cannot be delay-embedded — we do it routinely. The *sensor* is a smooth function of temperature; the *readout* is quantized. The same logic applies to words.

---

## Key Idea 2 — Words as Symbolic Measurements

### The Formal Construction

Define a symbolic measurement function hₛ: M → Σ (where Σ is a vocabulary of words) as the composition:

$$h_s = \kappa \circ h_r$$

- **hᵣ: M → ℝᵏ** — a smooth real-valued measurement of the underlying cognitive/articulatory/semantic state
- **κ: ℝᵏ → Σ** — a quantization map; assigns to each state the word whose region in state space contains it

The underlying system ϕ is smooth. The real-valued measurement hᵣ is smooth. The symbolic measurement hₛ is a coarsened version of hᵣ — it throws away information about exactly where in a word's region the state sits, but preserves which region it is in.

### What This Means for Language

In language production, assume:
- A continuous dynamical system (neural / cognitive / articulatory) evolves in a low-dimensional manifold M
- Each word w corresponds to a region Rw ⊂ M in this manifold
- The produced word at time t is wₜ = f(xₜ): the symbol whose region contains the current state

The sequence {wₜ} is a discrete symbolic measurement of a continuous dynamical system. Delay embedding this sequence:

$$W_t = (w_t, w_{t-\tau}, w_{t-2\tau}, \ldots, w_{t-(E-1)\tau})$$

reconstructs a space equivalent to M up to diffeomorphism — provided the embedding dimension E and delay τ are chosen appropriately.

### Exercise 2.1

A speaker produces the word sequence: *the cat sat on the mat*. Treating each word as a symbolic measurement:

(a) What is the underlying continuous system being measured?  
(b) What does the "region" Rw for the word "cat" correspond to in the cognitive state space?  
(c) Why does the discreteness of the word sequence not prevent Takens reconstruction?

---

## Key Idea 3 — The Topological Conjugacy Theorem

### The Formal Result

**Theorem 1 (Topological Conjugacy of Symbolic Reconstructions).**  
Assume:
1. ϕ: M → M is smooth and generically hyperbolic
2. hᵣ: M → ℝᵏ is smooth and generic (Takens-sense)
3. The partition P is **generating** with respect to ϕ and hᵣ

Then for sufficiently large embedding dimension E ≥ 2·dim(M) + 1, the symbolic delay embedding map is injective on a dense open subset of M, and the reconstructed symbolic dynamics are **topologically conjugate** to the original dynamics on the attractor.

### Unpacking "Topological Conjugacy"

Two dynamical systems are topologically conjugate if there exists a homeomorphism (continuous, bijective, with continuous inverse) between their state spaces that maps trajectories of one to trajectories of the other. In practical terms: the symbolic reconstruction and the original system have the same qualitative dynamics — the same attractor topology, the same stability structure, the same basins.

Topological conjugacy is stronger than "approximately similar" but weaker than "exactly the same." It is exactly the right notion for our purposes: we do not need to recover the exact continuous state from a word sequence; we need to recover enough structure to do classification, generation, and meaning-inference.

### The Key Step in the Proof

The quantization map κ is discontinuous (it jumps at partition boundaries), but it is **constant** on open sets of ℝᵏ except at those boundaries. Generically, the measure of states whose delay embedding vector lands on a boundary is zero. Therefore, on a full-measure subset of M, the symbolic delay embedding Ψ behaves as continuously as the real-valued embedding Φ. Injectivity on a dense open set then follows from the generating partition condition.

### Why "Generating Partition" Matters — and Can Be Relaxed

A generating partition is one that, in the limit of an infinite symbolic sequence, uniquely determines the underlying state. This is a strong condition. In practice, for language data, we do not need exact state recovery. We need enough geometric structure to:
- Distinguish different semantic regions (classification)
- Continue a trajectory coherently (generation)
- Preserve relative distances between states (meaning inference)

Empirical results from MARINA confirm that word sequences over a natural language vocabulary satisfy these weaker practical conditions, even if the generating partition condition in its strict form is not verified.

### Exercise 3.1

Explain in your own words why the following statement is true: "Quantization commutes with delay embedding up to topological equivalence." What does "up to" mean here, and what structure might be lost in the coarsening?

---

## Key Idea 4 — Empirical Precedents and the Burden of Proof

### Four Fields That Already Do This

The "smooth signals" objection, if correct, would invalidate a large body of existing scientific practice:

| Field | Discrete/Symbolic Data | Takens Applied? |
|---|---|---|
| Symbolic dynamics | Discrete partitions of continuous maps (e.g., logistic map → {0,1}) | ✓ Yes, routinely |
| Boolean networks | Binary state vectors | ✓ Yes, state reconstruction standard |
| Cellular automata | Discrete cell states | ✓ Yes, Takens-like embeddings work |
| Digital EEG/ECG | Integer sample values at finite precision | ✓ Yes, nobody objects |
| MARINA/TBT | Word token sequences | ✓ Yes, converges stably, generates coherent text |

The objection "Takens cannot apply to language" is empirically falsified by a working model (MARINA) and theoretically unsubstantiated by any rigorous treatment of the theorem.

### The Burden of Proof Principle

P11 makes explicit a methodological point with broad application in Geofinitism: when an objection to a method is (a) not stated in the theorem's rigorous formulations, and (b) empirically contradicted by working implementations, the burden of proof falls on the objector, not the practitioner.

This connects to ATT_28 (Commitment, Consensus, Admissibility): the "smooth signals" claim requires demonstrated evidence within the Geofinite framework — not an appeal to intuition about what "must" be required.

---

## Synthesis — P11's Place in the Programme

P11 occupies a very specific and important position in the architecture of the Geofinitism programme. It is not developing new theory — it is providing formal mathematical clearance for theory that already exists.

| Paper | What it establishes |
|---|---|
| P04 | Takens applies to symbol sequences — Geofinitist-theoretic proof |
| P11 | Takens applies to symbol sequences — standard dynamical systems proof |
| P01 | MARINA: empirical implementation of Takens on language |
| P05 | Language as a nonlinear dynamical system — linguistic framing |
| P08 | Autoregression is not Takens — the contrast case |

P04 and P11 approach the same result from different directions: P04 argues within the Geofinitist framework using Nexils, Alphons, and Measured Numbers; P11 argues within the standard mathematical literature using smooth maps, generating partitions, and topological conjugacy. Their agreement is the strongest available evidence that the theoretical foundation is sound.

The Geofinitist reading of this: a theorem is not a measurement until it has been confirmed from at least two independent vantage points. P04 and P11 together constitute the measurement of this theorem.

---

## Consolidation Questions

1. A critic says: "The logistic map produces real numbers, not symbols — so symbolic dynamics is different from applying Takens to language." Rebut this objection using the framework of P11.

2. Under what conditions might the generating partition condition fail for natural language? What practical consequences would this have for a Takens-based language model?

3. The Topological Conjugacy Theorem guarantees injectivity on a "dense open subset of M." What is the measure-zero set where injectivity might fail, and does this matter practically?

4. P08 argues that autoregression is not Takens because it reconstructs statistical distributions rather than geometric manifolds. How does P11's result sharpen this distinction?

5. The Geofinitist framework identifies words as Nexils — minimal discrete symbols produced by measurement. How does this Geofinitist account of a word relate to the formal account in P11 of a word as a symbolic measurement hₛ = κ ∘ hᵣ?

---

## Further Reading

- **P04** (Finite-Symbol Embedding Theorem) — the Geofinitist-theoretic companion to this note
- **P01** (TBT/MARINA) — the working implementation that empirically demonstrates the theorem
- **P05** (Language as Nonlinear Dynamical System) — the linguistic framing of the continuous dynamics underlying language
- **P08** (Autoregression Is Not Takens) — the contrast case that motivates why the Takens licence matters
- **ATT_01** (Words as Transducers) — the Geofinitist account of the word as a measurement boundary, complementing the formal account here

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
