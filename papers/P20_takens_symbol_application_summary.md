# Summary — P20: Takens' Theorem Applies to Discrete Symbol Sequences

**Paper ID:** P20  
**Title:** *Takens' Theorem Applies to Discrete Symbol Sequences: A Formal Note on Language as a Dynamical System*  
**Series:** Selected Communications  
**Author:** Kevin R. Haylett, Manchester, UK  
**Date:** June 2026  
**Length:** 5 pages  
**Primary College:** College of Attralucian Studies  
**Secondary Colleges:** College of Language Dynamics; College of Finite Symbolic Mechanics; College of Machine Intelligence  
**Primary Pillars:** P3 (Dynamic Flow), P5 (Finite Reality)  
**Secondary Pillars:** P2 (Approximations / Measurements)

---

## Purpose

P20 is a formal note written to address a specific and persistent objection to the Takens-based language modelling programme: the claim that Takens' delay embedding theorem requires smooth or continuous signals, and therefore cannot apply to discrete symbolic data such as words. The paper demonstrates that this objection rests on a misreading of the theorem. It provides a direct refutation, a formal proof in the appendix, and empirical evidence from related fields.

---

## The Objection

The standard formulation of Takens' theorem (1981) involves a smooth dynamical system φ : M → M on a compact differentiable manifold M of dimension d, and a smooth measurement function h : M → ℝ. The delay embedding map

$$\Phi_{h,\phi}(x) = \bigl(h(x),\, h(\phi(x)),\, h(\phi^2(x)),\, \ldots,\, h(\phi^{2d}(x))\bigr)$$

is an embedding of M into ℝ^{2d+1}.

The common misreading: since h maps to ℝ and the examples in textbooks use real-valued measurements (temperature, voltage, position), critics assume that the data {h(φ^t(x₀))} must itself be smooth or continuous. On this reading, discrete symbolic sequences — such as words — appear to violate the theorem's conditions.

---

## The Refutation

P20 identifies the precise location of the smoothness requirement:

| Component | Smoothness required? |
|---|---|
| State space M | Yes — a differentiable manifold |
| Dynamics φ : M → M | Yes — a smooth map |
| Measurement function h : M → ℝ | Yes — smooth |
| Data sequence {h(φ^t(x₀))} | **No** — can be real, integer, discrete, or symbolic |

The theorem's proof uses the smoothness of φ and h to guarantee that nearby states yield nearby measurement sequences. But the measurements themselves are just values at discrete times. They can be real-valued, integer-valued, or categorical. The confusion arises entirely from textbook examples that happen to use real-valued data.

---

## Symbolic Measurements Are Allowed

Let Σ be a finite set of symbols (e.g., words). P20 defines a symbolic measurement function h_s : M → Σ as a composition:

$$h_s = \kappa \circ h_r$$

where h_r : M → ℝ^k is a smooth real-valued measurement function, and κ : ℝ^k → Σ is a quantization or symbol-assignment function. As long as φ is smooth and h_r is smooth, the symbolic sequence s_t = h_s(φ^t(x₀)) inherits the deterministic structure of the underlying system. Takens' theorem applies to the real-valued sequence h_r(φ^t(x₀)); the symbolic sequence is a coarsened version of the same information.

---

## Words as Measurements of a Continuous System

P20 sets out the interpretive model for language:

- There exists a continuous dynamical system (neural, articulatory, or semantic) evolving in a low-dimensional manifold M
- Each word w corresponds to a region R_w ⊂ M
- The produced word at time t is w_t = f(x_t), where f returns the symbol whose region contains the current state x_t

The sequence {w_t} is therefore a discrete symbolic measurement of a continuous underlying dynamical system. By Takens' theorem, from a sufficiently long such sequence one can reconstruct a space equivalent to M (up to diffeomorphism) via delay vectors:

$$W_t = (w_t,\, w_{t-\tau},\, w_{t-2\tau},\, \ldots,\, w_{t-(E-1)\tau})$$

where E is the embedding dimension and τ the delay. This is the theoretical foundation of the Takens-based language model developed elsewhere in the corpus.

---

## Practical Evidence

P20 cites four domains in which delay embedding of discrete or symbolic sequences has been successfully applied:

1. **Symbolic dynamics** — the entire field studies discrete symbols arising from continuous maps (e.g., logistic map symbolic dynamics); Takens-like reconstruction is the standard tool
2. **Boolean networks** — state reconstruction via delay vectors using discrete {0,1} states
3. **Cellular automata** — Takens-like embeddings applied to discrete state sequences
4. **The Takens-based language model** (finitemechanics.com) — a fully functioning language model using Takens embedding of word sequences, achieving comparable or better performance than static embeddings on certain tasks without classical vector embeddings

---

## Appendix — Quantization Commutes with Delay Embedding Up to Topological Equivalence

The appendix provides a formal proof of the core mathematical claim.

**Setup:** Given a smooth dynamical system (M, φ) with M ⊂ ℝ^m compact, a smooth measurement h_r : M → ℝ^k, and a finite partition P = {P₁, ..., P_N} of ℝ^k, define the quantized symbolic measurement h_s(x) = i if h_r(x) ∈ P_i. The symbolic delay embedding vector is S_t = (s_t, s_{t-τ}, ..., s_{t-(E-1)τ}) ∈ {1,...,N}^E.

**Key step:** The quantized embedding satisfies Ψ(x) = κ^E ∘ Φ(x), where Φ is the Takens delay embedding and κ^E applies the quantization componentwise. Since κ is constant on open sets of ℝ^k (except at partition boundaries, which have measure zero), Ψ is continuous on a dense open subset of M.

**Injectivity:** Under the assumption that P is a *generating partition* with respect to φ and h_r — meaning the symbolic sequence {s_t} uniquely determines the asymptotic state in M up to a set of measure zero — the map Ψ is injective almost everywhere for embedding dimension E ≥ 2 dim(M) + 1.

**Conclusion:** Ψ provides a topological conjugacy between the original dynamical system and the symbolically reconstructed system on a full-measure subset of M. Quantizing a smooth measurement does not break the applicability of Takens' theorem — it merely coarsens the reconstruction, but the topological structure of the attractor remains recoverable.

The generating partition condition is strong but can be relaxed in practice. For language data, exact injectivity is not required — only that the delay embedding preserves enough structure to distinguish meanings and support downstream tasks.

---

## Significance for the School

P20 is the formal mathematical defence of the foundational move that P04 (Takens-Haylett Theorem), P05 (Language as NDS), P08 (Autoregression Is Not Takens), and P11 (Takens Applies to Symbol Sequences) all depend on. The objection it refutes — "Takens requires smooth signals" — is the most technically credible challenge to the entire Takens-based language modelling programme. P20's answer is decisive: the objection locates the smoothness requirement in the wrong place. Smooth data is neither required nor implied by the theorem.

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
