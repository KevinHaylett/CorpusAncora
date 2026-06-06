# P03 Lesson — JPEG Compression of Token Embeddings in Large Language Models

**Lesson ID:** P03  
**Paper:** P03  
**College:** College of Machine Intelligence  
**Level:** Advanced  
**Prerequisites:** ATT_06 or ATT_07 (familiarity with LLM dynamics); ATT_25 or ATT_52 (Takens / FPU); basic PyTorch familiarity helpful but not required  
**Pillars:** P3, P2 (primary); P1, P4, P5 (secondary)

---

## Overview

This lesson examines the JPEG embedding compression experiment — the empirical origin of the Geofinite language dynamics programme. The paper demonstrates, for the first time within the Geofinitism framework, that LLM cognition does not degrade randomly under information loss. It collapses into **structured attractor basins** — stable, reproducible states that mirror the underlying geometric skeleton of language. The lesson develops both the theoretical significance of this finding and its immediate practical consequence: a previously undocumented AI security vulnerability.

---

## Learning Outcomes

By the end of this lesson, students will be able to:

1. Explain the core experimental claim: LLM cognition under JPEG embedding compression does not degrade randomly but collapses into **structured linguistic attractor states** — and articulate why this is a significant finding about the geometry of language.

2. Describe the **methodology**: how a JPEG compression layer is inserted between the embedding lookup and the transformer blocks in a `ModifiedGPT2Model`, what the per-token processing pipeline does (reshape → normalise → JPEG encode/decode → inverse-normalise → flatten), and why this leaves model weights untouched.

3. Explain why **JPEG** was chosen as the probing instrument: its finite and geometric character (DCT + quantization on 2D blocks), its lossy-but-structured information loss, and its interpretability as a finite mechanics measurement tool.

4. Describe the **progressive collapse sequence** across compression levels (95% → 1%) and the characteristic attractor states at each level: coherence, categorical rigidity, paranoia/self-reference, violence/extreme recursion, Zen-like paradox.

5. State and apply the key empirical finding: **degradation is not random** — the same quality level produces the same attractor state regardless of prompt. Explain what this implies about the dimensionality and structure of language as a dynamical system.

6. Connect the finding to Geofinitism: explain why this result is "exactly as Finite Mechanics predicts" — i.e., how it confirms the claim that symbolic systems under finite constraint converge to admissible attractor regions rather than noise.

7. Explain the **security finding**: define embedding corruption as a stealth attack vector; explain why it bypasses prompt filtering, weight monitoring, and adversarial training; and describe the four domains of potential misuse (military, public opinion, corporate espionage, law enforcement).

8. Describe the **recommended countermeasures** for embedding integrity: cryptographic hashing, anomaly detection in embedding-space trajectories, redundant multi-encoder consistency checks.

9. Connect P03 to the broader research programme: explain its position as the empirical origin of the language dynamics sequence, and how it motivates the Takens-Based Transformer (P01) as a constructive response — P03 diagnoses the geometry; P01 exploits it.

10. Connect P03 to the **Attralucian Essays**: identify which essays provide the theoretical framework that P03 empirically supports (ATT_06/07, ATT_25, ATT_30) and which provide the formal apparatus that P03 instantiates (ATT_52 FPU, ATT_62 measurement constraints).

11. Apply the compression quality parameter as an analogue of the **resource bound** $B$ in Measured Kolmogorov Complexity (ATT_59): lower quality = tighter resource constraint = simpler representational regime; higher quality = closer to the full complexity of the embedding.

12. Evaluate the paper's methodological limitations: GPT-2 scale (small by current standards); the absence of systematic attractor mapping; the lack of replication across architectures — and identify which future directions would address these.

---

## Core Concepts

### JPEG as a Finite Measurement Instrument

JPEG compression is not being used here to compress images. It is being used as a **finite geometric probe** of embedding space. The key properties that make it a Geofinite instrument:

- **Finite:** operates on bounded 2D blocks via discrete cosine transform (not continuous Fourier analysis)
- **Geometric:** preserves structural relationships (low-frequency components, which dominate perceptual salience) while discarding high-frequency detail
- **Lossy:** information is irreversibly removed — this is the probe; the response to loss reveals structure
- **Parameterised:** the quality parameter is a single controllable dial from near-lossless (95%) to extreme loss (1%)

This is Geofinitism's measurement-first approach applied to an AI system: rather than theorising about the internal structure of an LLM, apply a finite measurement operation and observe the response.

### The Attractor Landscape of Language

The central finding is that the collapse under compression is **non-random and reproducible**. Different prompts at the same quality level produce the same attractor state. This means:

1. The attractor structure is a property of the **model**, not the prompt
2. Language (as encoded in GPT-2) has a **low-dimensional geometric skeleton** — high-dimensional embedding space collapses, under constraint, onto a small number of stable attractor regions
3. The attractor regions have **psychological and geometric character**: coherence, rigidity, paranoia, violence, paradox — these are the stable modes of the system

This is structurally identical to what Takens' theorem predicts for dynamical systems: a high-dimensional attractor can be reconstructed from a low-dimensional projection. P03 is, in effect, running the inverse of Takens — applying compression (dimension reduction) and watching the system fall into the attractors that Takens' theorem says must exist.

### The Progression

| Quality | Attractor | Geometric interpretation |
|---|---|---|
| 95% | Coherent + minor recursion | Near the full embedding manifold — slight deviation toward low-frequency basin |
| 75–50% | Categorical rigidity | Q&A attractor — the most structurally rigid mode of language; lowest-entropy category |
| 25–10% | Paranoia / self-reference | Fixation attractor — symbolic self-similarity; recursion dominates |
| 5% | Violence / extreme recursion | Deep fixation — primitive symbolic modes |
| 1% | Zen paradox | Profound disconnection — the embedding has collapsed to a near-zero-information region that maps paradoxically to the model's "deep prior" |

### Embedding Corruption as Attack Surface

The attack insight follows directly from the geometry. If embeddings have structured attractor basins, then:

- Controlled compression → controlled movement toward a target attractor
- The attack is **inside embedding space** — invisible to all token-level and weight-level defences
- The attacker does not need access to model weights, training data, or the prompt
- The effect is **predictable**: an attacker who knows the attractor landscape can target a specific state

This is a genuinely novel vulnerability class. It exists because the AI security community has focused on token-level and weight-level integrity while overlooking embedding-layer geometric integrity.

### Connection to FPU (ATT_52)

The quality sweep (95% → 50% → 25% → 5% → 1%) is a **Finite Process Unfolding** of the LLM's attractor landscape. Each quality level is one step in the unfolding — a new measured interaction state between the model and the compression constraint. The sequence of attractor states $\{A_q\}_{q \in Q}$ is a trajectory through the model's internal geometry, made visible by the finite probe.

### Connection to Measured Kolmogorov Complexity (ATT_59)

The quality parameter maps conceptually to the **resource bound** $B$ in $K^{\mathbb{M}}_{U,\tau,B}$:

- High quality (95%) ≈ large $B$ — the model can express its full complexity
- Low quality (1%) ≈ small $B$ — the model is forced into the shortest representational description available

The attractors visible at low quality are, in this reading, the model's **minimum description length attractors** — the simplest stable symbolic regimes the model can sustain under maximal resource constraint.

---

## Five Pillars in P03

| Pillar | Role in P03 |
|--------|-------------|
| **P3 — Dynamic Flow** (primary) | Attractor landscape; progressive collapse; linguistic attractors as stable dynamical modes; the finding that degradation is non-random and prompt-independent |
| **P2 — Approximations/Measurements** (primary) | JPEG as finite measurement instrument; quality parameter as resolution dial; the experiment as a probing measurement of the LLM's internal structure; embedding corruption as a measurement-level attack |
| **P1 — Geometric Container** (secondary) | Low-dimensional geometric skeleton of language; DCT as geometric transformation; the embedding manifold as the container within which attractor states organise |
| **P4 — Useful Fiction** (secondary) | JPEG-as-image-compression repurposed as a language measurement tool — the technique works as a useful fiction (image codec applied to non-image data); attractor states as stable useful fictions that the model falls into |
| **P5 — Finite Reality** (secondary) | Finite constraint reveals underlying structure; the principle that finite operations applied to symbolic systems expose their actual geometry; all measurement is finite |

---

## Discussion Questions

1. The paper claims that "degradation is not random." What would it look like if it *were* random? Design a null hypothesis test that would confirm or refute the attractor claim.

2. The attractor sequence (coherence → rigidity → paranoia → violence → paradox) has a psychological character. Is this a property of GPT-2 specifically, of transformer architectures generally, or of language itself? How would you test this?

3. The security finding depends on an adversary having access to the embedding layer between lookup and transformer. In what deployment scenarios is this access realistic? In what scenarios is it implausible?

4. The paper applies JPEG (a 2D image codec) to 1D embedding vectors by reshaping them into 2 rows. Is this the best possible geometric probe? What other finite, lossy, geometric transformations might reveal different aspects of the attractor landscape?

5. At quality 1%, the output is described as "Zen-like paradoxes — profound yet disconnected from original context." From a Geofinite perspective, what is happening here? Is this a "meaningful" attractor or a degenerate one? What does ∼Truth (ATT_65) say about outputs in this state?

6. The recommended countermeasure is cryptographic hashing of embeddings before transformer entry. What are the engineering costs of this approach? Does it fit within the Geofinite framework — is embedding integrity verification itself a Geofinite measurement operation?

---

## Connections to Prior Work

- **ATT_06, ATT_07** (Geodesic Fractal LLMs): P03 is the empirical confirmation of ATT_06/07's theoretical model
- **ATT_25** (Complex Analysis as Takens Embedding): the compression sweep is a finite Takens-type probe of the attractor structure
- **ATT_30** (Words as Trajectories): the attractor basins here are the trajectory attractors ATT_30 describes
- **ATT_52** (FPU): the quality sweep is FPU applied to the attractor landscape
- **ATT_59** (Geofinite Kolmogorov Complexity): quality parameter ↔ resource bound $B$
- **ATT_62** (Measurement Constraint Thesis): JPEG is a measurement instrument with a specific resolution bound
- **P01** (TBT, pending): the constructive response to P03's diagnostic — builds a transformer that exploits rather than merely probes the attractor geometry
- **P04** (Non-linear Dynamics in LLMs, pending): the theoretical formalisation of what P03 demonstrates empirically
- **P05** (Language as Nonlinear Dynamical System, pending): the framework paper; P03 is its experimental evidence base

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
