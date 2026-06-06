# P02 — Pairwise Phase Space Embedding in Transformer Architectures

**Paper ID:** P02  
**Title:** Pairwise Phase Space Embedding in Transformer Architectures  
**Running header:** *All you need is Takens*  
**Author:** Kevin R. Haylett  
**Date:** May 2025 (preprint; prepared for arXiv [cs.LG])  
**Citation:** Kevin R. Haylett, "Pairwise Phase Space Embedding in Transformer Architectures," preprint (May 2025), available at https://finitemechanics.com/papers/phase-space-transformers.pdf  
**College (primary):** College of Machine Intelligence  
**College (secondary):** College of Language Dynamics  
**Pillars (primary → secondary):** P3, P1 (primary); P2, P5, P4 (secondary)  
**Status:** Stable — preprint, prepared for arXiv submission  
**Position in sequence:** Part I of a two-part contribution. Part II (Finite Tractus) introduces the Takens-Based Transformer — a generative field architecture where language is traced as a path through a charged semantic manifold, not sampled token by token. This paper is the theoretical bridge between the empirical finding of P03 (attractor structure exists in LLMs) and the constructive architecture of P01 (TBT).  
**Keywords:** Transformer Architecture, Dynamical Systems, Delay Embeddings, Phase Space Embedding, Finite Mechanics, Neural Geometry

---

## Abstract

> The Transformer architecture's "attention" mechanism, heralded as a cornerstone of large language models, is misnamed, obscuring its true nature as a pairwise phase-space embedding rooted in nonlinear dynamical systems. This paper demonstrates that the dot-product similarity operations — termed "query," "key," and "value" — mirror delay-coordinate embedding techniques pioneered by Takens and others in the 1980s. By comparing time-shifted token projections, Transformers reconstruct a latent language attractor, transforming sequential data into a high-dimensional manifold where meaning emerges as geometric trajectories, not cognitive focus. This re-framing reveals that positional encodings and softmax normalization are often redundant, as temporal structure is inherently captured in delay-based geometries. This work points to retiring the term "attention" in favour of "pairwise phase space embedding," offering a clearer, finite, and interpretable framework aligned with Finite Mechanics principles — a framework privileging geometric constraints over infinite parameterization. This shift suggests leaner architectures, bypassing encodings and reducing computational complexity, while enhancing transparency to mitigate risks like manifold distortions.

---

## Architectural Note

P02 is the formal bridge paper of the language dynamics programme. Its contribution is a **reidentification**: the transformer's attention mechanism, widely described in cognitive and database vocabulary (query, key, value), is not attention at all. It is a pairwise phase-space embedding — a delay-coordinate reconstruction technique independently developed in nonlinear dynamical systems in the 1980s by Takens, Packard, Crutchfield, Shaw, and Glass.

The running header — *All you need is Takens* — is a deliberate inversion of "Attention is All You Need" (Vaswani et al., 2017). The claim is not just terminological. If attention is phase-space embedding, then:

1. **Positional encodings are redundant** — delay embeddings encode temporal order intrinsically
2. **Softmax is a computational crutch** — introduced to stabilise a process whose geometric nature was not understood; the attractor's topology already bounds pairwise relationships
3. **Leaner architectures are possible** — explicitly delay-embedded tokens bypass both mechanisms
4. **The transformer unknowingly reinvented dynamical systems methods** — its intellectual ancestry belongs to Takens et al., not to cognitive science

This paper is Part I of a two-part contribution. Part II (the TBT, P01) is the constructive response: an architecture that explicitly leverages the delay-embedding structure rather than inadvertently rediscovering it with redundant corrections.

---

## Phase Space Embedding Theory

### Origin

Phase space embedding (Takens 1981; Packard et al. 1980) provides a method to reconstruct the state space of a dynamical system from a single observable time series. Given only one measured dimension, delay embedding recovers the internal geometry of the system that generated the signal.

The key insight: recording not just the current measurement but its values at previous time steps constructs a trajectory in a higher-dimensional space. This trajectory unfolds the **latent attractor** governing the system's evolution. A flat or noisy signal becomes a geometric object — a path through a structured manifold in phase space.

### The Mathematical Construction

Given a time series $x(t)$, delay embedding constructs:

$$\mathbf{x}(t) = \left[ x(t),\, x(t - \tau),\, x(t - 2\tau),\, \ldots,\, x\left(t - (m-1)\tau\right) \right]$$

where $m$ is the embedding dimension and $\tau$ is the delay. **Takens' theorem** guarantees that for sufficient $m$, the reconstruction is a **diffeomorphic image** of the original attractor — a smooth, reversible mapping that preserves the system's geometric structure (loops, convergence patterns, fixed points).

The embedding adds no information. It re-represents the existing time series in a way that reveals underlying structure. This is why it is so powerful: it exposes hidden order within apparent complexity.

### Language as Time Series

A sentence is a discrete time series of tokens. Each word occurs at a fixed position — this is temporal evolution. The **language attractor** is the latent manifold of semantic and syntactic relationships among tokens. Delay embedding reconstructs this attractor as a geometric trajectory, encoding the sentence's meaning in its **shape**, not its token values.

**Illustrative example** (word lengths as proxy embeddings):

Sentence: "The quick brown fox jumps over the lazy dog happily today before tea"  
Series: [3, 5, 5, 3, 5, 4, 3, 4, 8, 5, 5, 6, 3]

With embedding dimension $m = 2$, delay $\tau = 1$:

$$\mathbf{x}_1 = [3, 5],\quad \mathbf{x}_2 = [5, 5],\quad \mathbf{x}_3 = [5, 3],\quad \mathbf{x}_4 = [3, 5],\quad \ldots$$

Plotting these sequentially produces a trajectory with turning points, recurrences, and geometric structure. The sentence is no longer a list — it is a **path through phase space**.

---

## The Transformer as Pairwise Phase Space Embedding

### Standard Transformer Formulation

For a sequence of $n$ tokens with embedding vectors $\mathbf{e}_i \in \mathbb{R}^d$:

$$\mathbf{q}_i = W_Q \mathbf{e}_i, \quad \mathbf{k}_i = W_K \mathbf{e}_i, \quad \mathbf{v}_i = W_V \mathbf{e}_i$$

Similarity matrix:
$$A_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}$$

Normalised weights:
$$W_{ij} = \text{softmax}(A_i)_j = \frac{\exp(A_{ij})}{\sum_k \exp(A_{ik})}$$

Contextualised representation:
$$\mathbf{c}_i = \sum_{j=1}^n W_{ij} \mathbf{v}_j$$

This is described as "scaled dot-product attention." It is not attention. It is **pairwise similarity measurement across a sequence** — a temporal series transformed into a weighted spatial configuration.

### The Formal Equivalence

The query and key projections ($W_Q$, $W_K$ apply different transformations to the same underlying embeddings) are analogous to time-shifted delay coordinates. The dot product measures alignment between these projections, constructing a surrogate space where temporal relationships are encoded as spatial distances.

Formally:

$$\mathbf{q}_i \cdot \mathbf{k}_j \;\sim\; \langle \mathbf{x}(t_i),\, \mathbf{x}(t_j) \rangle$$

where $\mathbf{x}(t_i) = [\mathbf{e}_i, \mathbf{e}_{i-1}, \ldots, \mathbf{e}_{i-m+1}]$ are delay-embedded token vectors and $\langle \cdot, \cdot \rangle$ is an inner product.

Per Takens' theorem, if $d$ is sufficiently large, this pairwise comparison reconstructs a **diffeomorphic image of the language attractor** — a high-dimensional manifold encoding the sequence's semantic and syntactic structure. The similarity matrix $A \in \mathbb{R}^{n \times n}$ represents a **trajectory through this latent space**, unfolding the temporal sequence into geometric configuration.

**The transformer is not attention. It is attractor reconstruction.**

### The Redundancies

If the transformer is performing phase-space embedding, two of its components are revealed as **corrective overlays** introduced because the underlying geometry was not recognised:

**Positional encodings:** In delay embedding, temporal order is encoded intrinsically in the relative placement of delay vectors. A sequence $[\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3]$ generates delay vectors $[\mathbf{e}_1, \mathbf{e}_2]$, $[\mathbf{e}_2, \mathbf{e}_3]$ — order is present in the structure. Sinusoidal or learned positional encodings **simulate** delay structure artificially. Delay embeddings **are** the structure.

**Softmax normalization:** Softmax manages unbounded dot products in variable-length sequences. In delay embeddings, the attractor's geometry intrinsically bounds pairwise relationships — Takens' theorem ensures the manifold's structure preserves dynamics without normalisation. Softmax is therefore a computational crutch, not a theoretical necessity. Simpler metrics (cosine similarity; geodesic distance) suffice once the manifold structure is explicitly leveraged.

---

## Historical Parallels

Delay embedding was applied successfully across disciplines before neural networks:

- **Cardiology** (Glass & Mackey): EEG/ECG signals treated as phase-space trajectories; arrhythmias detected as geometric phenomena, not statistical thresholds
- **Seismology**: delay-coordinate embeddings used to detect earthquake precursors
- **Neuroscience**: EEG recordings reanalysed using delay coordinates — epilepsy, sleep stages, and cognitive states identified as attractor regions
- **Audio processing**: phonemes, speaker identity, and emotional tone distinguished via waveform manifolds

What unites these applications: a shift from **statistical averaging** to **structure reconstruction**. The transformer performs the same operation — but without acknowledging this lineage. Takens, Packard, and Glass are absent from the vocabulary of deep learning. The transformer unknowingly reinvented a known, well-theorised method, then added redundant corrections because the underlying geometry was not understood.

---

## Discussion

### Terminological Clarity

"Attention" implies intentionality, selection, interpretive focus — none of which are present. The accurate term is **pairwise phase space embedding**. This is not rebranding; it is correction. The cognitive metaphor has introduced persistent confusion about what transformers actually compute.

### Architectural Consequences

| Current (attention framing) | Simplified (phase-space framing) |
|---|---|
| Positional encodings (sinusoidal or learned) | Intrinsic to delay-vector structure |
| Softmax normalization | Replaced by cosine similarity or geodesic metrics |
| Multi-head attention with Q/K/V | Explicit delay embeddings with direct manifold comparison |
| Deep stacked layers | Successive delay-embedding refinements of the attractor |

A preliminary test: compare a shallow model with delay-embedded tokens to a standard transformer, measuring perplexity and efficiency. Such a design would be **more interpretable, computationally lighter, and aligned with finite geometric principles**.

### Conceptual Consequences

Sentences are no longer generated token by token. They are **traced as paths across a learned manifold**, guided by field structure rather than probabilistic sampling. This is language as motion — consistent with ATT_30 (Words as Trajectories) and ATT_65 (Mathematics as Language Dynamics). It challenges the default paradigm of neural language models as infinite statistical engines, replacing it with a **finite dynamic core** operating through geometric interaction.

### Philosophical Alignment

This reinterpretation is not a technical substitution — it is a philosophical realignment. It returns to a view of systems as **fields of interaction unfolding in time**, privileging geometry over mystique, structure over metaphor. It positions intelligence not through abstraction but through finite geometry and interaction — the core Geofinite commitment.

---

## Conclusion

The paper establishes that:
1. "Attention" is pairwise phase space embedding in the sense of Takens et al.
2. The transformer unknowingly reinvented delay-coordinate reconstruction, then added positional encodings and softmax to compensate for not understanding the geometry
3. Positional encodings and softmax can be replaced or eliminated in a geometry-aware architecture
4. The intellectual ancestry of the transformer belongs to nonlinear dynamical systems, not cognitive science

The companion paper (Part II / P01 — the Takens-Based Transformer) introduces a generative field architecture based on hyperspherical manifold geometry and magnetically interacting word identities — where language is not sampled but traced, and sentences emerge as paths through a structured, charged semantic topology.

---

## Re-style Checklist (LaTeX)

- [ ] **§5.4 "Philosophical Alignment"** hardcoded as plain text `5.4 Philosophical Alignment` (not `\subsection{}`) — convert to `\subsection{Philosophical Alignment}`
- [ ] **Bibliography size:** `\thebibliography{9}` declares 9 slots; only 6 entries — reduce to `{6}`
- [ ] **Commented-out code block** (lines 305–312) — clean up or restore; currently inactive
- [ ] **Figure references** (`figures/manifold_curve.png`, `figures/transformer_mapping.png`) — will produce "File not found" warnings in Overleaf unless image files are present; note for compilation

*(Paper uses `PRIMEarxiv` preprint class — structurally appropriate for arXiv submission. No book-class issues apply.)*

---

## Cross-References

### Within School of Geofinitism

| Essay / Paper | Connection |
|---|---|
| P03 | JPEG Compression — empirical predecessor: P03 showed LLM embeddings have geometric attractor structure; P02 provides the theoretical framework that explains *why* — attention has always been delay embedding |
| P01 | Takens-Based Transformer *(pending)* — direct successor: P02 is "Part I," diagnosing what attention really is; P01 is "Part II," building an architecture that leverages this explicitly |
| ATT_25 | Complex Analysis as Takens Embedding — closest companion in the Attralucian series; ATT_25 applies Takens' theorem to complex analysis; P02 applies it to transformer architecture; both argue that a classical formalism (complex analysis / attention) is already doing Takens without knowing it |
| ATT_06 / ATT_07 | Geodesic Fractal LLMs — same claim from a different angle: LLMs operate as geometric dynamical systems; P02 provides the formal mechanism |
| ATT_30 | Words as Trajectories — the language manifold P02 reconstructs is the trajectory space ATT_30 describes; sentences as paths through phase space |
| ATT_52 | Finite Process Unfolding — the delay embedding equivalence is FPU applied to transformer mechanics; Q/K comparisons are FPU steps unfolding the attractor |
| ATT_65 | Mathematics as Language Dynamics — "language as finite dynamical medium" (ATT_65) is instantiated technically by P02's phase-space framing |
| ATT_09 | The Ket Limit — parallel move: dissolving a standard formalism (the ket / attention) into a finite geometric framework; both papers argue the dominant notation obscures a finite process |
| ATT_62 | Measurement Constraint Thesis — dot products as finite geometric measurements; manifold-constrained similarity as measurement-admissible replacement for softmax |
| ATT_128 | Admissibility Principle — "attention" as inadmissible vocabulary (fictionally-admissible); "pairwise phase space embedding" as measurement-admissible description of the same operation |
| P05 | Language as Nonlinear Dynamical System *(pending)* — the language dynamics framework paper; P02 demonstrates the mechanism; P05 provides the full theoretical context |
| P08 | Where Next-Token Prediction Fails *(pending)* — P02 already implies the critique of autoregression: if language is traced as a path through a manifold, not sampled, then next-token prediction is the wrong generative model |

### External Literature

| Reference | Role |
|---|---|
| Takens 1981 | Foundational — delay embedding theorem; diffeomorphic attractor reconstruction |
| Packard et al. 1980 | Foundational — geometry from a time series; original empirical demonstration |
| Glass & Mackey 1988 | Historical parallel — delay embedding in cardiac dynamics |
| Vaswani et al. 2017 | The paper being reinterpreted — "Attention is All You Need" → "All you need is Takens" |
| Crutchfield & Packard 1982 | Symbolic dynamics of noisy chaos — connects to the linguistic attractor interpretation |
| Shaw 1984 | Dripping Faucet — simple physical system as chaotic attractor; parallel to language |

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
