# P11 — Takens' Theorem Applies to Discrete Symbol Sequences

**Full title:** Takens' Theorem Applies to Discrete Symbol Sequences: A Formal Note on Language as a Dynamical System  
**Paper ID:** P11  
**Author:** Kevin R. Haylett — Manchester, UK  
**Date:** May 24, 2026  
**Journal:** Journal of Geofinitism  
**Primary College:** College of Machine Intelligence  
**Secondary Colleges:** College of Language Dynamics; College of Attralucian Studies  
**Primary Pillars:** P5 (Finite Reality), P1 (Geometric Container)  
**Secondary Pillars:** P2 (Approximations/Measurements), P3 (Dynamic Flow)  
**Status:** Stable  
**Pages:** 7  
**Source:** `P11_takens_and_symbols.pdf`

---

## Abstract (verbatim)

> A common objection to applying Takens' delay embedding theorem to language data is that the theorem requires "smooth" or "continuous" signals. This note formally demonstrates that the objection is unfounded. Takens' theorem applies to any sequence of measurements from a deterministic dynamical system, regardless of whether those measurements are real-valued, discrete, or symbolic. Words, as discrete symbols, are legitimate measurements. The reconstructed attractor captures the underlying continuous dynamics. The "smoothness" requirement applies to the unknown dynamical system and the measurement function, not to the measured data themselves. We clarify the mathematics and show that language models based on Takens embedding are theoretically sound. An appendix provides a formal proof that quantization commutes with delay embedding up to topological equivalence.

---

## Architectural Note

P11 is a targeted formal rebuttal — a mathematical clarification note responding to a specific objection raised against the TBT/MARINA architecture (P01). It occupies a precise niche in the programme: it is the rigorous licence for everything built on Takens embedding of language data. Where P04 (the Finite-Symbol Embedding Theorem) establishes the Geofinitist-theoretic foundation for Takens applied to symbols, P11 addresses the objection on its own mathematical terms — arguing within the standard dynamical systems literature, citing Takens (1981) and subsequent rigorous treatments, to show the objection misunderstands the theorem's conditions.

The note is short (7 pages), precise, and complete. Its Appendix contains a formal proof sketch (Theorem 1: Topological Conjugacy of Symbolic Reconstructions) that constitutes a standalone mathematical result.

---

## Core Thesis

The "smooth signals" objection to applying Takens' theorem to language is a mathematical misunderstanding. Smoothness in Takens' theorem applies to the underlying dynamical system ϕ and the measurement function h — not to the sequence of observations {h(ϕᵗ(x₀))}. That sequence can be real-valued, integer-valued, or categorical. Words are discrete measurements of a continuous cognitive/articulatory dynamical system, and delay embedding of such sequences is both theoretically licensed and empirically validated.

> "The objection 'Takens requires smooth signals' is a mathematical misunderstanding. The burden of proof now rests on those who claim otherwise to show why symbolic measurements violate the theorem — a claim they will not find in Takens' original paper or any subsequent rigorous treatment."

---

## Key Concepts

### The Misunderstanding — Where Smoothness Applies

Takens' theorem has three smoothness requirements:
1. The state space M — a differentiable manifold
2. The dynamics ϕ: M → M — a smooth map
3. The measurement function h: M → ℝ — smooth

Smoothness does **not** apply to the sequence {yₜ} where yₜ = h(ϕᵗ(x₀)). This sequence is a set of discrete samples. The confusion arises because textbook examples use real-valued measurements (temperature, voltage), but the theorem makes no such restriction.

### Words as Symbolic Measurements

Let Σ be a finite set of symbols (words). A symbolic measurement function hₛ: M → Σ can be written as a composition:
$$h_s = \kappa \circ h_r$$
where hᵣ: M → ℝᵏ is a smooth real-valued measurement and κ: ℝᵏ → Σ is a quantization or symbol assignment function.

The underlying system ϕ is smooth; hᵣ is smooth; the symbolic sequence is a coarsened version of the real-valued sequence. For a generating partition, delay embedding of symbolic sequences still reconstructs the attractor topology.

In language production: a word wₜ = f(xₜ) returns the symbol whose manifold region contains the current state xₜ. The sequence {wₜ} is a discrete symbolic measurement of a continuous dynamical system.

### The Delay Embedding Map for Language

From a symbolic sequence {wₜ}:
$$W_t = (w_t, w_{t-\tau}, w_{t-2\tau}, \ldots, w_{t-(E-1)\tau})$$
where E is the embedding dimension and τ the delay. This is the foundation of the Takens-based language model. Reconstruction recovers a space equivalent to the original manifold M up to diffeomorphism.

### Appendix: Topological Conjugacy of Symbolic Reconstructions (Theorem 1)

The formal result: under three conditions — (1) ϕ is smooth and generically hyperbolic; (2) hᵣ is smooth and generic (Takens-sense); (3) the partition P is **generating** with respect to ϕ and hᵣ — the symbolic delay embedding map

$$\Psi(x) = \lim_{T\to\infty} (s_0, s_{-\tau}, \ldots, s_{-(E-1)\tau})$$

is injective on a dense open subset of M, and the reconstructed symbolic dynamics are **topologically conjugate** to the original dynamics on the attractor.

**Key step:** since κ is constant on open sets of ℝᵏ (except on partition boundaries), and since the set of points whose delay embedding vector falls on a boundary has measure zero generically, Ψ is continuous on a dense open set. Injectivity follows from the generating partition condition and the sufficient embedding dimension E ≥ 2·dim(M) + 1.

**Corollary:** quantization commutes with delay embedding up to topological equivalence. Symbolic sequences preserve the topological structure of the attractor.

**Remark:** the generating partition condition can be relaxed in practice. For language data, exact injectivity is not required — only that the symbolic delay embedding preserves enough structure to distinguish meanings and support downstream tasks. Empirical results (MARINA) confirm this.

### The Claim-Reality Table

| Claim | Reality |
|---|---|
| "Takens requires smooth signals" | No — it requires smooth dynamics and measurement function, not smooth data |
| "Discrete symbols violate the theorem's assumptions" | No — symbols are valid measurements if they arise from a continuous system |
| "You can't apply Takens to text" | Yes you can — text is a discrete-time symbolic sequence from an underlying continuous process |
| "The reconstruction won't work with discrete data" | Symbolic dynamics, Boolean networks, and cellular automata all apply delay embedding to discrete states successfully |

### Empirical Precedents

Four established areas confirm that Takens embedding of discrete and symbolic data is standard practice:
- **Symbolic dynamics** — the entire field studies discrete symbols from continuous maps (logistic map, etc.)
- **Boolean networks** — state reconstruction via delay vectors is standard
- **Cellular automata** — Takens-like embeddings work with discrete states
- **MARINA/TBT** — a fully functioning language model using Takens embedding, achieving comparable or better performance than static embeddings, without classical vector embeddings

### Appendix B: Eight Clarification Points

A structured summary of the foundational arguments:
1. Smoothness applies to the system, not the data
2. Integers are a subset of reals — token index sequences are real-valued sequences
3. Words are measurements, not the system itself
4. Symbolic dynamics provides direct precedent
5. Quantization preserves topological structure
6. Relative curvature is sufficient (not full injectivity)
7. Digital sampling provides a direct analogy — all practical Takens applications use discrete samples
8. Empirical validation supersedes a priori objections — MARINA is proof by construction

---

## Five Pillars in P11

| Pillar | Role in P11 |
|--------|-------------|
| **P5 — Finite Reality** (primary) | All measurements are finite and bounded; symbolic measurements are physically grounded in finite substrates; the "smooth signals" objection treats discreteness as a defect rather than a physical reality |
| **P1 — Geometric Container** (primary) | Meaning as trajectory in the semantic manifold; the manifold M is the geometric container from which symbolic measurements are drawn; the delay embedding reconstructs this container |
| **P2 — Approximations/Measurements** (secondary) | Words as discrete measurements carrying measurement structure; the symbolic sequence as a coarsened but structure-preserving measurement of the underlying continuous state; generating partitions as measurement granularity |
| **P3 — Dynamic Flow** (secondary) | Language as a dynamical system; the sequence {wₜ} as a trajectory; delay embedding as reconstruction of the flow |

---

## Connections to Other Work

- **P01** (TBT/MARINA): P11 provides the formal mathematical licence for the TBT architecture — the theorem establishing that Takens embedding of word sequences is theoretically sound; MARINA is cited as empirical proof of concept
- **P04** (Finite-Symbol Embedding Theorem / Takens-Haylett Theorem): P04 establishes the Geofinitist-theoretic foundation; P11 establishes the same result within the standard dynamical systems literature — the two papers address the same question from different directions and reinforce each other
- **P05** (Language as Nonlinear Dynamical System): P11 provides the formal justification for P05's central claim; the generating partition argument is the bridge between P05's linguistic dynamical systems framing and standard Takens theory
- **P08** (Autoregression Is Not Takens): P11 complements P08 — P08 shows what autoregression lacks; P11 establishes what Takens-based methods have; together they define the contrast that motivates TBT
- **ATT_01/ATT_03** (Transducers, Tranfictors): the measurement function h: M → Σ is a formal version of the transducer — a finite symbol-producing boundary between continuous state space and the symbolic domain

---

## References

1. Takens, F. (1981). Detecting strange attractors in turbulence. *Dynamical Systems and Turbulence*, Springer, 366–381.
2. Kennel, M. B., & Buhl, M. (2003). Estimating good discrete dimensions for time series. *Physical Review E*, 67(4), 046216.
3. Shalizi, C. R., & Crutchfield, J. P. (2001). Computational mechanics: Pattern and prediction, structure and simplicity. *Journal of Statistical Physics*, 104(3), 817–879.
4. Sauer, T., Yorke, J. A., & Casdagli, M. (1991). Embedology. *Journal of Statistical Physics*, 65(3), 579–616.

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
