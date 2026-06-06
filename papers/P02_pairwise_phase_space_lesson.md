# P02 Lesson — Pairwise Phase Space Embedding in Transformer Architectures

**Lesson ID:** P02  
**Paper:** P02  
**College:** College of Machine Intelligence  
**Level:** Advanced  
**Prerequisites:** ATT_25 or ATT_52 (Takens / FPU — essential); familiarity with transformer architecture (Vaswani et al. 2017) helpful; ATT_06 or ATT_07 (LLM geometry) recommended  
**Pillars:** P3, P1 (primary); P2, P5, P4 (secondary)

---

## Overview

This lesson examines P02's central reidentification: the transformer's "attention" mechanism is not attention. It is **pairwise phase-space embedding** — a delay-coordinate reconstruction technique from nonlinear dynamical systems theory, independently developed by Takens, Packard, and Glass in the 1980s. The paper establishes a formal equivalence between Q/K dot products and delay-vector comparisons, then draws out three consequences: positional encodings are redundant, softmax is a computational crutch, and leaner geometry-aware architectures are possible. P02 is the theoretical bridge between the empirical finding of P03 (LLMs have attractor structure) and the constructive architecture of P01 (the Takens-Based Transformer).

---

## Learning Outcomes

By the end of this lesson, students will be able to:

1. State the central reidentification: explain why "attention" is a misnomer for the transformer's similarity mechanism, and what "pairwise phase-space embedding" captures more accurately.

2. Describe **Takens' delay embedding theorem**: how a 1D time series $x(t)$ is mapped to an $m$-dimensional trajectory via $\mathbf{x}(t) = [x(t), x(t-\tau), \ldots, x(t-(m-1)\tau)]$; what diffeomorphic attractor reconstruction means; and why the embedding reveals structure without adding information.

3. Apply delay embedding to a **language sequence**: explain how a sentence treated as a discrete token time series produces a phase-space trajectory encoding meaning in its geometric shape rather than in token values.

4. State the **standard transformer formulation** ($W_Q, W_K, W_V$ projections; similarity matrix $A_{ij}$; softmax normalisation; contextualised output $\mathbf{c}_i$) and identify each component's role.

5. Demonstrate the **formal equivalence** $\mathbf{q}_i \cdot \mathbf{k}_j \sim \langle \mathbf{x}(t_i), \mathbf{x}(t_j) \rangle$: explain how $W_Q$ and $W_K$ applying different transformations to the same embeddings are analogous to time-shifted delay coordinates, and how the similarity matrix $A$ represents a trajectory through the language attractor.

6. Explain why **positional encodings are redundant** under the phase-space interpretation: delay vectors encode temporal order intrinsically through relative positioning; sinusoidal or learned positional signals simulate artificially what delay structure provides naturally.

7. Explain why **softmax is a computational crutch** under the phase-space interpretation: introduced to stabilise unbounded dot products in variable-length sequences; the attractor's topology already bounds pairwise relationships; cosine similarity or geodesic metrics suffice.

8. Describe the **historical parallels**: how phase-space embedding was applied in cardiology (Glass/Mackey), seismology, neurophysiology, and audio processing before neural networks — and why these precedents constitute the transformer's unacknowledged intellectual ancestry.

9. Articulate the **three-level consequence** of the reidentification: terminological (retire "attention"), architectural (leaner delay-embedded models without positional encodings or softmax), and conceptual (language as traced manifold paths, not sampled token sequences).

10. Connect P02 to **P03** (empirical predecessor) and **P01** (constructive successor): explain the three-step programme — P03 diagnoses the geometry; P02 names it; P01 exploits it.

11. Connect P02 to the **Attralucian Essays**: identify ATT_25 (Complex Analysis as Takens Embedding), ATT_52 (FPU), ATT_30 (Words as Trajectories), and ATT_65 (Mathematics as Language Dynamics) as the theoretical companions within the School of Geofinitism.

12. Apply the **Geofinite philosophical alignment**: explain why renaming "attention" as pairwise phase-space embedding is not merely terminological — it is a shift from a cognitive/anthropomorphic frame to a geometric/finite-mechanics frame, consistent with Geofinitism's rejection of metaphor as primary description.

---

## Core Concepts

### Takens' Theorem — The Reconstruction Guarantee

Takens' theorem (1981) is one of the most powerful results in nonlinear dynamics. Its claim: given a time series of observations of *any* dynamical system, a suitable delay embedding of those observations is **diffeomorphic to the system's original attractor**. You do not need to measure the internal state directly — the geometry is recoverable from a single observable signal through delay coordinates.

This is why it matters for language: a sequence of token embeddings is a one-dimensional (per-dimension) time series. Delay embedding of those token sequences recovers the **latent language attractor** — the geometric structure that governs how meaning is organised across sentences.

### The Equivalence Argument

| Delay Embedding | Transformer |
|---|---|
| Time series $x(t)$ | Token embedding sequence $\mathbf{e}_i$ |
| Delay vectors $\mathbf{x}(t_i) = [\mathbf{e}_i, \mathbf{e}_{i-1}, \ldots]$ | Query $\mathbf{q}_i = W_Q \mathbf{e}_i$ (a different linear projection of $\mathbf{e}_i$) |
| Comparison $\langle \mathbf{x}(t_i), \mathbf{x}(t_j) \rangle$ | Dot product $A_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}$ |
| Reconstructed attractor trajectory | Similarity matrix $A \in \mathbb{R}^{n \times n}$ |
| Attractor geometry bounds relationships | Softmax attempts to bound what geometry already constrains |
| Temporal order encoded in vector positions | Positional encodings simulate what delay structure provides |

The formal statement: $\mathbf{q}_i \cdot \mathbf{k}_j \sim \langle \mathbf{x}(t_i), \mathbf{x}(t_j) \rangle$. Per Takens' theorem, for sufficiently large $d$, this pairwise comparison reconstructs a diffeomorphic image of the language attractor.

### The Two Redundancies

**Positional encodings** simulate delay structure that delay embedding provides intrinsically. Given delay vectors $[\mathbf{e}_1, \mathbf{e}_2]$ and $[\mathbf{e}_2, \mathbf{e}_3]$, temporal order is encoded in the structure of the vectors themselves. The transformer's sinusoidal positional signals are a corrective overlay introduced because the delay-structure interpretation was not recognised.

**Softmax** manages unbounded dot products in variable-length sequences. In delay embedding, the attractor's intrinsic topology constrains pairwise relationships — this is what Takens guarantees. The manifold's geometry is self-bounding. Softmax is therefore a workaround for a problem that does not exist once the geometric nature of the operation is explicit.

### The Historical Lineage

The transformer's intellectual ancestry belongs to nonlinear dynamical systems, not cognitive science:

- Takens (1981): delay-coordinate reconstruction of attractors from observations
- Packard et al. (1980): geometry from a time series — the experimental demonstration
- Crutchfield & Packard (1982): symbolic dynamics of noisy chaos
- Glass & Mackey (1988): delay embedding applied to biological rhythms (cardiac dynamics)

These were established, theorised methods for reconstructing dynamical systems from observations. The transformer rediscovered the core operation and added softmax and positional encodings to compensate for not recognising it.

### Language as Traced Path

Under the phase-space interpretation, a sentence is not a sequence of tokens to be predicted one-by-one. It is a **path through the language attractor manifold** — a geometric trajectory that the model traces. Meaning is not in the token values but in the **shape of the trajectory** through phase space.

This has generative consequences: the correct generative model for language is not next-token sampling but **manifold tracing** — following the attractor geometry from an initial position. The TBT (P01) is the architecture that does this explicitly.

---

## Five Pillars in P02

| Pillar | Role in P02 |
|--------|-------------|
| **P3 — Dynamic Flow** (primary) | Language as trajectory through attractor manifold; meaning encoded in geometric shape; sentences as paths not lists; the historical drift from signal to geometry |
| **P1 — Geometric Container** (primary) | The language manifold as a structured geometric container; delay-vector space as the phase-space container for language; diffeomorphic attractor reconstruction |
| **P2 — Approximations/Measurements** (secondary) | Dot products as finite geometric measurements; manifold-constrained similarity as measurement-admissible replacement for softmax; the delay embedding as finite probing of attractor geometry |
| **P5 — Finite Reality** (secondary) | Finite geometric principles over infinite parameterization; leaner architectures through geometry-awareness; the attractor as a finite object constraining an otherwise unbounded similarity space |
| **P4 — Useful Fiction** (secondary) | "Attention" as a useful fiction — the cognitive vocabulary has been practically effective, but it is admissible only as an approximation, not as a literal description; softmax as a useful fiction compensating for unrecognised geometry |

---

## Discussion Questions

1. Takens' theorem guarantees diffeomorphic attractor reconstruction for *sufficiently large* embedding dimension $m$. In the transformer, the embedding dimension $d$ is fixed by the model. Does this mean the transformer's phase-space reconstruction is *approximate* by construction? What are the implications for meaning representation?

2. The paper claims positional encodings are redundant. However, many experiments show that transformers perform significantly worse without them. How do you reconcile this with the delay-embedding argument? Does it suggest that current transformer training is optimised for the wrong inductive bias?

3. The paper proposes retiring the term "attention." Is this purely terminological, or does the vocabulary choice actively shape research directions? Can you think of research paths that the cognitive vocabulary has blocked, and research paths that the dynamical-systems vocabulary opens?

4. The historical parallels (cardiology, seismology) are striking. In those domains, delay embedding was used for **analysis** of existing systems, not for **generation**. Does the generative use of transformer-as-delay-embedding require additional theoretical justification beyond Takens' theorem?

5. Softmax is described as a "computational crutch." But it is also differentiable, which matters for gradient-based training. If you replaced softmax with cosine similarity or a geodesic metric, what changes in the training dynamics? Is this a practical barrier or a solvable engineering problem?

6. From a Geofinite perspective, the paper argues that "attention" is inadmissible as a description — it is fictionally-admissible at best, carrying unresolved assumptions about cognitive intentionality. Does renaming a mechanism change its admissibility? Or does the renaming need to be accompanied by an architectural change (as in P01)?

---

## Connections to Prior Work

- **ATT_25** (Complex Analysis as Takens Embedding): closest companion — ATT_25 argues complex analysis is already doing Takens; P02 argues transformers are already doing Takens; same move, different target
- **ATT_52** (FPU): the Q/K comparison sequence is Finite Process Unfolding applied to the token sequence — each pairwise comparison is a step of the unfolding
- **ATT_30** (Words as Trajectories): P02 provides the mechanism for ATT_30's conceptual claim; the language attractor and its phase-space paths are the "trajectories" of ATT_30
- **ATT_06 / ATT_07** (Geodesic Fractal LLMs): theoretical companion from the Attralucian series — same claim that LLMs are geometric dynamical systems, from a different analytical angle
- **ATT_65** (Mathematics as Language Dynamics): the philosophical framing — language as finite dynamical medium; P02 is the technical instantiation
- **ATT_09** (Ket Limit): parallel move — dissolving the dominant notation (quantum ket / attention) into a finite geometric framework
- **P03** (JPEG Compression): empirical predecessor — P03 found the attractor structure; P02 names the mechanism that generates it
- **P01** (TBT, pending): the constructive response — P02 is Part I (diagnosis); P01 is Part II (architecture)
- **P05** (Language as Nonlinear Dynamical System, pending): the full language dynamics framework paper; P02 contributes the transformer mechanism
- **P08** (Next-Token Prediction Fails, pending): P02 implies the critique — if language is traced not sampled, autoregression is the wrong generative model

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
