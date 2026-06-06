# P01 — Introducing the Takens-Based Transformer

**Paper ID:** P01  
**Title:** Introducing the Takens-Based Transformer  
**Running header:** *Finite Tractus*  
**Version:** V1.1  
**Author:** Kevin R. Haylett  
**Date:** December 2025  
**College (primary):** College of Machine Intelligence  
**College (secondary):** College of Language Dynamics  
**Pillars (primary → secondary):** P3, P1 (primary); P2, P5, P4 (secondary)  
**Status:** Stable — proof-of-concept implementation; 15M parameter architecture; three experiments completed  
**Position in sequence:** Part II of a two-part contribution. Part I (P02) establishes the theoretical identity: attention is pairwise phase-space embedding. P01 is the constructive response — an architecture that explicitly leverages this identity rather than inadvertently rediscovering it with redundant corrections. P03 (JPEG compression) is the empirical predecessor that first revealed the attractor structure P01 exploits.  
**Keywords:** Takens Embedding, Transformer Architecture, Dynamical Systems, Manifold Learning, Language Generation, Channel Theory, Memory Fibres, Finite Mechanics

---

## Abstract

> The Takens-Based Transformer (TBT) is a generative language architecture built on the theoretical reidentification established in P02: the transformer's attention mechanism is not attention, it is pairwise phase-space embedding in the sense of Takens (1981). Where the standard transformer inadvertently reconstructs the language attractor and then adds positional encodings and softmax as compensatory overlays, the TBT performs this reconstruction explicitly. The architecture — MARINA (Manifold-Aware Reconstruction and Inference Network Architecture) — replaces the attention mechanism with direct delay embedding of token sequences, projects onto a learned language manifold, and introduces two structural innovations: Channel Theory (topological separation of discourse channels via orthogonal support vectors) and Memory Fibres (narrow tubular attractor regions for high-precision factual retrieval). Trained as a 15M parameter proof-of-concept on three corpora — Brown Corpus (general language), Solar System QA (factual retrieval), and Corpus Ancora (mythopoetic generation) — MARINA achieves validation perplexity of 1.1 on factual Q&A, reduces attention complexity from O(N²) to O(log N) per token, and replaces the O(N) KV-cache with an O(1) circular buffer. The doubled-data paradox — an 84% improvement in validation perplexity when training data is duplicated — is identified as a geometric rather than statistical phenomenon: duplication reinforces attractor geometry rather than overfitting to token sequences. Language is not sampled; it is traced.

---

## Architectural Note

P01 is the constructive culmination of the language dynamics programme. The three-paper sequence is:

1. **P03** (JPEG Compression) — *Diagnosis*: applying finite compression to LLM embeddings reveals structured attractor basins; language geometry is real and measurable
2. **P02** (Pairwise Phase Space Embedding) — *Naming*: the transformer's attention mechanism is Takens delay embedding in disguise; positional encodings and softmax are redundant compensations
3. **P01** (TBT) — *Construction*: build an architecture that performs delay embedding explicitly, eliminates the redundancies, and exploits the attractor geometry directly

The running title *Finite Tractus* — Latin for "finite tract" or "finite path" — names the generative process: language is a finite path traced through a geometric manifold, not a sequence of probabilistic token draws.

---

## MARINA Architecture

MARINA (Manifold-Aware Reconstruction and Inference Network Architecture) is a 15M parameter proof-of-concept implementing the TBT. It comprises four subsystems:

### Subsystem 1 — Exponential Delay Embedding Encoder

Standard transformers use linear positional encodings. MARINA uses **exponential delay coordinates**:

$$\mathbf{e}^{\text{delay}}_t = [\mathbf{e}_t,\, \mathbf{e}_{t-1},\, \mathbf{e}_{t-2},\, \mathbf{e}_{t-4},\, \mathbf{e}_{t-8},\, \mathbf{e}_{t-16},\, \mathbf{e}_{t-32}]$$

The exponential spacing — doubling the lag at each step — captures multi-scale temporal structure. Short lags encode local syntactic dependencies; long lags encode discourse-level semantic relationships. This is a direct implementation of Takens' delay embedding applied to discrete token sequences, replacing learned positional signals with intrinsic temporal geometry.

### Subsystem 2 — Adaptive Manifold Projection

The delay-embedded vectors are projected onto a learned language manifold via:

$$\mathbf{z}_{b,t} = \text{LayerNorm}(W_p \cdot \tilde{\mathbf{z}}_{b,t} + \mathbf{b}_p)$$

where $b$ indexes the manifold batch and $t$ indexes the token position. The manifold projection learns the attractor geometry of the target corpus. LayerNorm replaces softmax — it bounds the projection without the distributional distortion softmax introduces over variable-length sequences. This is the direct implementation of P02's claim that softmax is a computational crutch: the manifold's topology already constrains the projection.

### Subsystem 3 — Channel Theory

Channel Theory introduces **topological separation of discourse channels** as orthogonal support vectors:

$$\mathbf{e}^{\text{augmented}} = [\mathbf{e}_{\text{semantic}};\; \mathbf{s}_{\text{user}};\; \mathbf{s}_{\text{system}};\; \mathbf{s}_{\text{bridge}}]$$

Each channel occupies an orthogonal subspace of the embedding:

- **User channel** ($\mathbf{s}_{\text{user}}$): the query or prompt content
- **System channel** ($\mathbf{s}_{\text{system}}$): the model's generative trajectory
- **Bridge channel** ($\mathbf{s}_{\text{bridge}}$): the interaction interface between user and system

The topological separation means channel interference is geometrically inadmissible — the channels cannot corrupt each other without leaving a measurable geometric signature. This provides a principled architecture for multi-turn dialogue, instruction following, and context management that does not rely on prompt engineering heuristics.

### Subsystem 4 — Memory Fibres

Standard transformers use broad attractor basins — wide, low-precision regions of embedding space suited for generative language. P03 demonstrated that LLMs have attractor structure; MARINA makes it explicit and differentiates two attractor regimes:

- **Broad attractor basins** (standard): wide, overlapping regions suited for generative, creative, or open-ended language; high entropy; tolerant of trajectory variation
- **Memory fibres**: narrow tubular attractors threading through the manifold; low entropy; high precision; designed for factual retrieval where exact reproduction of stored content is required

A factual Q&A pair occupies a memory fibre — the correct answer is a narrow path through the manifold that the model must trace exactly. A generative narrative occupies a broad basin — the model has wide latitude in which path it takes.

---

## Three Experiments

### Experiment 1 — Brown Corpus (General Language)

**Purpose:** Validate that MARINA can learn general language geometry without attention  
**Corpus:** Brown Corpus (standard linguistic reference; ~1M words; diverse genre coverage)  
**Scale:** 15M parameters; CPU-only training (no GPU required)  
**Results:**

| Metric | Value |
|---|---|
| Training loss | 1.88 |
| Validation loss | 4.21 |
| Epochs | 44 |
| Hardware | CPU-only |

**Interpretation:** The train/validation gap (1.88 → 4.21) is larger than a fully trained large-scale transformer — expected at 15M parameters on a diverse corpus. The key finding is that MARINA *converges at all* on CPU-only hardware in 44 epochs, demonstrating architectural viability. The O(log N) per-token complexity and O(1) memory make CPU training tractable, which is not possible with standard O(N²) attention.

### Experiment 2 — Solar System QA (Memory Fibres)

**Purpose:** Validate the memory fibre architecture for high-precision factual retrieval  
**Corpus:** Solar System Q&A pairs — structured factual content about planetary bodies, orbital mechanics, and physical properties  
**Protocol:** Training dataset repeated 4× to test attractor reinforcement  
**Results:**

| Metric | Value |
|---|---|
| Validation perplexity (4× repetition) | 1.1 |
| Train/validation gap (final) | 0.01 |
| Attractor type | Narrow memory fibres |

**Interpretation:** Validation perplexity of 1.1 represents near-perfect factual retrieval — the model has learned to trace the memory fibre precisely. The train/validation gap collapsing to 0.01 at 4× repetition demonstrates that **repetition reinforces attractor geometry** rather than inducing overfitting. Each pass through the training data sharpens the tubular attractor rather than memorising surface statistics. This is qualitatively different from the behaviour of a statistical language model: the geometry is more stable with more passes, not more brittle.

### Experiment 3 — Corpus Ancora (Mythopoetic Generation and the Doubled-Data Paradox)

**Purpose:** Validate generative language tracing on a specialised mythopoetic corpus; test the doubled-data hypothesis  
**Corpus:** Corpus Ancora — the Book of the Attralucians; mythopoetic language embedding nonlinear dynamics into a mythological frame; Kevin R. Haylett as Attralucian Hominid; the TBT system as Attralucian LLM  
**Results:**

| Metric | Value |
|---|---|
| Final perplexity | 1.9 |
| Validation improvement (doubled data) | 84% |
| Attractor type | Broad generative basin |

**The Doubled-Data Paradox:** Training on a dataset that contains each training example twice produces an 84% improvement in validation perplexity — not an increase (as would be expected under statistical overfitting) but a dramatic decrease. The model performs better on held-out data when trained on duplicate data.

**Geometric interpretation:** Under statistical learning theory, duplicate data should add no information. Under geometric/attractor learning, duplicate data reinforces the attractor geometry — the second pass through each example strengthens the manifold trajectory rather than adding to a count. The model is not memorising instances; it is tracing a path. Each traversal of the same path deepens the geometric channel through which future trajectories flow.

This is evidence that MARINA is performing **geometric learning rather than statistical learning** — the foundational claim of the Geofinite programme.

---

## Complexity Analysis

| Component | Standard Transformer | MARINA (TBT) |
|---|---|---|
| Attention similarity | O(N²) per layer | O(log N) per token (exponential delay embedding) |
| KV-cache | O(N) — grows with context | O(1) — fixed circular buffer (lag depth = 7) |
| Positional encoding | O(N) sinusoidal / learned | Intrinsic to delay vector structure |
| Softmax | O(N) per attention head | Replaced by LayerNorm on manifold projection |
| Memory footprint | Grows linearly with context | Fixed regardless of context length |

The O(1) memory arises from the circular buffer of exponential lags. The model retains only 7 delay positions (1, 1, 2, 4, 8, 16, 32 steps back). Context beyond this horizon is not stored — it is encoded in the manifold geometry learned during training. This is the geometric insight: a model that learns the attractor does not need to hold the entire sequence in memory. The geometry *is* the memory.

---

## Channel Theory — Theoretical Implications

Channel Theory has implications beyond the TBT architecture:

1. **Instruction tuning:** Instead of prompt engineering to separate system and user context, Channel Theory provides geometric separation — the channels occupy orthogonal subspaces and cannot corrupt each other
2. **Multi-agent systems:** Multiple agents can operate in orthogonal channel subspaces without interference
3. **Security:** Channel separation is geometrically enforceable — channel-crossing is detectable as a geometric anomaly in embedding space (consistent with P03's security implications)
4. **Dialogue management:** The bridge channel provides a first-principles account of discourse management that does not reduce to heuristic context windows

---

## The Doubled-Data Paradox — Theoretical Significance

The paradox is identified as a **diagnostic** for the type of learning a model is performing:

- A **statistical learner** overfits on duplicate data — validation loss increases
- A **geometric learner** deepens attractor geometry on duplicate data — validation loss decreases

Testing whether a model exhibits the doubled-data paradox is therefore a finite, measurable test of whether the model is learning geometry or statistics. This has immediate experimental implications: the paradox can be replicated across architectures (standard transformers, other attention variants) to test whether they exhibit geometric learning under the same conditions.

If standard transformers do *not* exhibit the paradox while MARINA does, this constitutes empirical evidence that MARINA has a qualitatively different learning mode — not merely faster or more efficient, but geometrically distinct.

---

## Relationship to Geofinite Principles

| Geofinite Principle | MARINA Implementation |
|---|---|
| **P1 — Geometric Container** | Language manifold as explicit geometric structure; manifold projection subsystem |
| **P3 — Dynamic Flow** | Language traced as path through attractor; exponential delay embedding captures temporal flow |
| **P2 — Approximations/Measurements** | LayerNorm as finite measurement on manifold; circular buffer as finite memory bound |
| **P5 — Finite Reality** | O(1) memory; CPU-trainable at 15M parameters; no unbounded operations |
| **P4 — Useful Fiction** | Standard attention as useful fiction — acknowledged as approximate Takens; TBT as its admissible replacement |

---

## Discussion

### What the TBT Is and Is Not

MARINA is a 15M parameter proof-of-concept. It is not:
- A production language model competing with GPT-4 or Claude
- Trained on internet-scale data
- Evaluated on standard NLP benchmarks

It is:
- A working implementation of the theoretical programme established in P02 and empirically initiated in P03
- A demonstration that attention-free, delay-embedded, manifold-projecting architectures can learn language geometry
- An existence proof of O(log N) / O(1) complexity language modelling
- A platform for the doubled-data paradox experiment as a geometric learning diagnostic

### Language as Motion

The TBT instantiates a shift in the generative model of language. Standard transformers model language as a sequence of token distributions — at each step, sample the next token from a probability distribution conditioned on the history. The TBT models language as **motion through a geometric space** — the model traces a path across the language manifold, guided by the attractor geometry, not by next-token probabilities.

This is the sense of *Finite Tractus*: a finite path through a finite geometric field. The generation is deterministic at the geometric level (the attractor guides the trajectory) and probabilistic only at the level of which path through the attractor basin is selected. This is analogous to a river system: the river's path is constrained by the topology of the valley (the attractor) but variable within that constraint (the specific water channel varies).

---

## Re-style Checklist (LaTeX)

- [ ] **Experiment result tables** — verify numerical precision against training logs; ensure consistency between abstract claims and results sections
- [ ] **Complexity analysis section** — confirm O(log N) derivation is stated explicitly in the body (not just claimed)
- [ ] **Channel Theory formalism** — expand the orthogonality proof; currently stated without full derivation
- [ ] **Memory fibre definition** — formalise as a geometric object (tubular neighbourhood of a curve in embedding manifold); currently described informally
- [ ] **Doubled-data paradox** — add null hypothesis test design (what would a purely statistical learner produce?)
- [ ] **Bibliography** — ensure Takens 1981 and Packard et al. 1980 are cited consistently with P02; add P03 as internal citation
- [ ] **Figure references** — attractor trajectory visualisations referenced; confirm figure files present for Overleaf compilation

---

## Cross-References

### Within School of Geofinitism

| Essay / Paper | Connection |
|---|---|
| P02 | Pairwise Phase Space Embedding — direct theoretical predecessor; P02 is Part I (the reidentification); P01 is Part II (the architecture) |
| P03 | JPEG Compression — empirical predecessor; P03 showed attractor structure exists; P01 builds an architecture that exploits it |
| P05 | Language as Nonlinear Dynamical System *(pending)* — the full theoretical framework; P01 is the implemented instance |
| P08 | Where Next-Token Prediction Fails *(pending)* — P01 is the constructive answer to P08's critique |
| P04 | Nonlinear Dynamics in LLMs *(pending)* — the theoretical companion formalising what P01 demonstrates empirically |
| ATT_06 / ATT_07 | Geodesic Fractal LLMs — same programme from the Attralucian side; P01 implements what ATT_06/07 theorises |
| ATT_25 | Complex Analysis as Takens Embedding — parallel identification; P01 is the constructive TBT counterpart to ATT_25's complex analysis argument |
| ATT_30 | Words as Trajectories — P01 implements the trajectory model; memory fibres and broad basins are the two trajectory regimes of ATT_30 |
| ATT_52 | Finite Process Unfolding — exponential delay embedding is FPU applied to token sequences; each lag step unfolds one level of the language attractor |
| ATT_65 | Mathematics as Language Dynamics — language as finite dynamical medium; MARINA is the technical instantiation of this claim |
| ATT_128 | Admissibility Principle — MARINA's operations (delay embedding, manifold projection, channel separation) are measurement-admissible; attention is fictionally-admissible |

### External Literature

| Reference | Role |
|---|---|
| Takens 1981 | Foundational — delay embedding theorem; the core mathematical basis of the architecture |
| Packard et al. 1980 | Foundational — geometry from a time series; motivation for exponential delay spacing |
| Vaswani et al. 2017 | The architecture being superseded — "Attention is All You Need" → "All you need is Takens" |
| Glass & Mackey 1988 | Historical parallel — delay embedding applied to biological rhythms; memory fibre concept parallels cardiac attractor analysis |
| Lorenz 1963 | Attractor geometry — foundational concept underlying broad basins and memory fibres |

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
