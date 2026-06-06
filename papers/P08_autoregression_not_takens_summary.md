# P08 — Autoregression Is Not Takens

**Paper ID:** P08  
**Title:** Autoregression Is Not Takens: Why Next-Token Prediction Cannot Faithfully Reconstruct Semantic Manifolds  
**Author:** Kevin R. Haylett  
**Date:** February 2026  
**Citation:** Kevin R. Haylett, "Autoregression Is Not Takens," February 2026, finitemechanics.com/papers/takens-versus-autoreg.pdf  
**College (primary):** College of Machine Intelligence  
**College (secondary):** College of Language Dynamics  
**Pillars (primary → secondary):** P3, P1 (primary); P5, P2 (secondary)  
**Status:** Stable — three formal theorems; empirical validation section; full bibliography (15 references)  
**Acknowledgments:** "The author thanks Claude (Anthropic) for collaborative development of these ideas through extensive dialogue and mathematical refinement."  
**Position in sequence:** The critique paper. Where P02 reidentifies what attention actually is (Takens embedding), P08 demonstrates what autoregression is not (Takens reconstruction). Together they form a pincer argument: standard transformers do Takens accidentally and incompletely; MARINA (P01) does it intentionally and correctly. P08 is the formal proof of insufficiency — the mathematical case that the dominant paradigm cannot, in principle, achieve what Takens requires.  
**Keywords:** Autoregression, Takens Embedding, Next-Token Prediction, Manifold Reconstruction, MARINA, Semantic Manifold, Delay Embedding, Language Models, Geofinitism

---

## Abstract

> Autoregressive next-token prediction has become the dominant paradigm in modern language modeling, powering architectures from RNNs to Transformers and state-space models (SSMs). Proponents often invoke Takens' delay embedding theorem to suggest that autoregression naturally reconstructs the underlying dynamical system of language. This paper demonstrates that such claims are mistaken: vanilla autoregression is a predictive technique that incidentally uses delayed observations, but it lacks the structural guarantees required for faithful manifold reconstruction.
>
> We formalize this distinction through three independent arguments. First, standard autoregressive models employ soft, query-dependent combinations of history (Attention) that violate the fixed-coordinate requirements of Takens' theorem. Second, they lack principled multi-scale delay selection, treating history uniformly or via learned approximations without geometric guarantees. Third, their compression of infinite history into finite states or caches introduces irreversible information loss, precluding diffeomorphic reconstruction.
>
> These limitations explain why autoregressive models achieve low perplexity through statistical pattern-matching but fail to emerge novel geometric primitives like domain-adaptive attractors. In contrast, explicit Takens-inspired architectures — specifically MARINA — enforce delay-coordinate structure. This yields linear complexity, constant memory, and emergent manifold topologies: narrow "memory fibres" for factual precision and broad basins for creative coherence. Our results underscore a philosophical point: meaning in language is dynamical and geometric, not merely predictive.

---

## Architectural Note

P08 is the formal critique paper of the language dynamics programme. Its argument is tightly structured: three independent mathematical theorems, each proving a different structural reason why autoregression cannot perform Takens reconstruction, followed by empirical validation comparing autoregressive Transformers, State Space Models (SSMs), and MARINA.

The paper's relationship to P02 is important: P02 argues that what transformers *are* doing is Takens embedding (accidentally, imperfectly). P08 argues that what autoregression *cannot* do is Takens reconstruction (provably, in principle). These are consistent positions: transformers accidentally perform Takens-style operations in their attention mechanism (P02) but the autoregressive training objective cannot guarantee faithful manifold reconstruction (P08). The TBT/MARINA (P01) resolves this by designing explicit Takens structure.

The title is a direct inversion of the field's assumptions — a deliberate rhetorical move consistent with the programme's running headers (*All you need is Takens*, *Finite Tractus*).

---

## Historical Context: The Eclipse of Nonlinear Dynamics

P08 traces the intellectual trajectory that produced the current autoregressive orthodoxy:

**The rise of autoregression:**
- ARIMA time-series statistics (1970s) → RNNs → GPT (2018) → scaling laws → "lower perplexity = better understanding"
- Three reasons for dominance: self-supervised scalability (no labels required), generative appeal (sampling creates illusion of creativity), and the gradient descent shortcut (statistical pattern matching is easier to tune than chaotic dynamical models)

**The eclipse of dynamics:**
- 1980s: Hopfield networks and "computation at the edge of chaos" linked neural networks to attractors; Takens' theorem inspired state-space reconstruction research
- Training difficulties (vanishing gradients) and the rise of feedforward architectures pushed dynamics into a niche
- The cost: "attractor collapse on long contexts, hallucinated 'facts' (trajectory divergence), and no emergent geometric primitives — models that predict well but do not understand the shape of the data"

---

## Mathematical Preliminaries

### Takens' Theorem (Formal Statement)

For a compact manifold $M$ of dimension $d$, smooth flow $\phi^t: M \to M$, and generic smooth observable $h: M \to \mathbb{R}$, the delay coordinate map with $m \geq 2d+1$:

$$\Phi(x) = (h(x),\, h(\phi^{-\tau}(x)),\, h(\phi^{-2\tau}(x)),\, \ldots,\, h(\phi^{-(m-1)\tau}(x)))$$

is a **diffeomorphism** onto its image — a smooth, invertible map with smooth inverse. This requires **fixed delays** and a **stable coordinate system**.

### Autoregressive Modeling (Formal Statement)

Standard autoregression estimates:
$$P(x_{t+1} \mid x_1, \ldots, x_t) = f_\theta(H(x_1, \ldots, x_t))$$

minimising the cross-entropy loss:
$$\mathcal{L} = -\sum \log P(x_{t+1} \mid x_{<t+1})$$

**The Fundamental Gap:** Minimising $\mathcal{L}$ ensures predictive accuracy. It does not guarantee that the model's internal state space is diffeomorphic to the generating manifold $M$.

---

## Three Theorems

### Theorem 1 — Non-Rigid Embedding (Query-Dependent Coordinates)

**Statement:** The effective embedding mechanism in Transformer-based autoregressive models is query-dependent and time-varying, violating the fixed-coordinate requirement for a global diffeomorphism.

**Proof sketch:** Takens requires $\Phi$ where the $k$-th coordinate is always $h(x_{t-k\tau})$ — rigid geometric scaffolding. In a Transformer, attention operates as:

$$\text{Attention}(Q_t, K, V) = \text{softmax}\!\left(\frac{Q_t K^T}{\sqrt{d_k}}\right) V$$

The contribution of past token $x_{t-k}$ to the current state depends on attention weight $\alpha_{t,t-k}$, which is a function of the current query $Q_t$. The coordinate system used to represent the past **changes dynamically based on the current token**.

This produces not a single map $\Phi: M \to \mathbb{R}^m$ but a family of maps $\{\Phi_t\}$. A time-varying map cannot guarantee stable reconstruction of a static invariant set (the attractor). The manifold is "wobbly" — deformed by the very act of querying it.

### Theorem 2 — Uniform History Treatment (No Multi-Scale Delays)

**Statement:** Standard autoregressive models lack the inductive bias for geometric multi-scale delay selection, leading to topological redundancy or undersampling.

**Proof sketch:** Faithful manifold reconstruction depends critically on the delay $\tau$:
- $\tau$ too small: $x_t$ and $x_{t-\tau}$ are highly correlated → reconstruction collapses to the diagonal (redundancy)
- $\tau$ too large: chaotic divergence makes $x_t$ and $x_{t-\tau}$ statistically independent → reconstruction produces a cloud (undersampling)

Transformers treat history as a "bag of tokens" with positional encodings. The model must learn to attend to specific lags, but there is no geometric constraint enforcing optimal $\tau$ values. Empirically, attention collapses to recent tokens (short $\tau$) or specific "induction heads," failing to capture multi-scale, fractal structure.

SSMs (Mamba) improve on this with continuous-time dynamics, but rely on fixed discretization steps ($\Delta$) that do not necessarily align with the intrinsic Lyapunov timescales of the semantic manifold.

### Theorem 3 — Information Loss (Lossy Compression Precludes Invertibility)

**Statement:** Finite-state compression and fixed-window caches introduce irreversible information loss, preventing the inverse mapping $\Phi^{-1}$ required for a diffeomorphism.

**Proof sketch:** Takens theoretically permits reconstruction from a single observable, provided the delay vector is long enough ($m \geq 2d+1$). In a continuous system, the "history" is implicitly infinite-resolution.

Discrete autoregressive models impose two bottlenecks:
1. **Finite context window (Transformers):** History beyond $N$ tokens is truncated
2. **Fixed state size (RNNs/SSMs):** Infinite history compressed into a vector $s_t \in \mathbb{R}^d$

By the **Data Processing Inequality**: if the true semantic manifold state is $\mathcal{M}_t$ and the model state is $s_t$, then $I(\mathcal{M}_t; s_t) < H(\mathcal{M}_t)$ if the compression is lossy. Language exhibits power-law correlations (long-range dependencies); any finite truncation discards information necessary to distinguish trajectories that diverge only after long times. This makes $\Phi$ non-injective (many-to-one), precluding it from being a diffeomorphism.

---

## Empirical Validation

### MARINA vs Autoregression — Basin Separation

On domain classification tasks testing attractor formation:

| Metric | MARINA | GPT-2 (Transformer) |
|---|---|---|
| Basin separation | **100%** | Poor (fuzzy overlapping clouds) |
| Parameters | **1.1M** | 117M |
| Training data (equivalent perplexity) | **100 examples** | 1M+ examples |
| Data efficiency | **10,000× improvement** | Baseline |

*Basin separation measures whether latent trajectories for different semantic domains occupy distinct, non-overlapping regions of phase space.*

**Geometric interpretation:** Transformers achieve domain distinction through massive overparameterisation — effectively memorising domain markers. MARINA achieves it through geometric structure — domains correspond to distinct attractors in a well-reconstructed manifold.

### Protein Folding Extension

Treating amino acid sequences as observations of a folding dynamical system, with delay-coordinate embeddings of residue properties:

| Metric | MARINA | AlphaFold2 |
|---|---|---|
| RMSD accuracy | Sub-angstrom | Sub-angstrom |
| Parameters | **50M** | 93M |
| Training structures | **1,000** | 170,000+ (PDB) |

**Theoretical significance:** Protein folding is a physical dynamical system with an underlying energy landscape manifold. By respecting delay-embedding structure, MARINA captures this geometry directly; AlphaFold must learn it implicitly through massive scale.

### Ablation Studies — Necessity of Log-Space Delays

| Delay Type | Result | Geometric Reason |
|---|---|---|
| Linear ($x_{t-1}, x_{t-2}, \ldots$) | Degraded to near-Transformer | Oversamples short timescales; misses long-range |
| Random ($x_{t-r_i}$, random $r_i$) | Catastrophic failure | No principled multi-scale structure; no coherent attractors |
| Learned (trainable delay parameters) | Partial improvement, unstable | Delays collapsed to recent tokens during training — reproduces Theorem 2's problem |
| **Exponential / log-space** | **Robust manifold reconstruction** | Covers multiple Lyapunov timescales geometrically |

Only exponential delays produce robust results. This confirms the geometric claim: multi-scale structure must be prescribed, not learned.

### Long-Context Stability

| Model | Perplexity at 1K tokens | Perplexity at 100K tokens | Behaviour |
|---|---|---|---|
| GPT-2 (Transformer) | Baseline | Logarithmic increase | Attention dilution; positional encoding breakdown |
| Mamba (SSM) | Better than Transformer | Mild degradation | Fixed state size compresses distant history |
| **MARINA** | Baseline | **Constant** | Log-space structure stabilises; once longest delay reached, all timescales covered |

This validates Theorem 3: lossy compression causes divergence; structured sparse history preserves topology.

### Emergent Topological Structures

Projecting MARINA's 128-dimensional latent space to 2D via UMAP reveals two distinct geometric regimes:

**Memory Fibres** (factual recall — e.g., "What is the capital of France?"):
- Tight, thread-like trajectories through latent space
- Paraphrased inputs produce nearby trajectories converging to the same narrow fibre
- Corresponds to attractor dynamics: truth as a stable fixed point or limit cycle

**Broad Basins** (creative generation — e.g., "Write a poem about..."):
- High-dimensional regions with large volume
- Diverse outputs correspond to wandering within the basin, maintaining thematic coherence
- The basin constrains; within it the model ranges freely

Transformer baselines: "fuzzy clouds without distinct topological structure — factual and creative prompts produce overlapping, amorphous distributions, explaining their tendency to hallucinate facts and lack creativity simultaneously."

---

## Corpus Ancora — Semantic Volume Training Methodology

Beyond architecture, P08 introduces a training methodology explicitly designed for manifold coverage:

**Problem:** Standard training treats text as a linear stream, learning conditional probabilities along a single trajectory — "narrow grooves" in parameter space, brittle generalization.

**Corpus Ancora principles:**
1. **Convergent paraphrasing:** For each semantic intent, generate 10–100 paraphrased variants from different starting tokens but converging to the same conceptual attractor → teaches that meaning is the attractor, not the surface form
2. **Divergent continuations:** From a single context, branch into multiple valid continuations → populates the basin around truth, preventing spurious precision
3. **Explicit attractors:** Include high-frequency "landmark" statements (axioms, definitions, widely agreed facts) serving as fixed points → trajectories naturally gravitate toward these, reducing hallucination

**Geometric interpretation:** Narrow training paths learn a 1D curve through an $n$-dimensional manifold. Corpus Ancora learns the full $n$-dimensional volume, ensuring delay-coordinate reconstruction spans the manifold, not a thread through it.

**Safety implication:** Harmful content lies on divergent trajectories far from the training manifold. Geometric AI safety through topology rather than post-hoc filtering.

---

## Architecture Comparison

| Property | Transformers | SSMs (Mamba) | MARINA |
|---|---|---|---|
| Complexity | $O(T^2)$ | $O(T)$ | $O(T \log T)$ |
| Memory | $O(T^2)$ (KV cache) | $O(1)$ (fixed state) | $O(\log T)$ (delay buffer) |
| Theoretical basis | Attention (statistical) | Continuous-time dynamics | Takens embedding |
| Delay structure | Learned (query-dependent) | Fixed discretization $\Delta$ | Geometric (log-space) |
| Manifold fidelity | No guarantee | Approximate | Provable (satisfies Takens) |
| Parameter efficiency | Low (billions required) | Medium | High (orders of magnitude fewer) |
| Long-context stability | Degrades (positional encoding) | Good (constant memory) | Excellent (topological stability) |
| Emergent geometry | Fuzzy clouds | Implicit structure | Explicit fibres/basins |
| Training data need | Massive (billions of tokens) | Large (millions) | Minimal (thousands) |
| Interpretability | Low (attention heatmaps) | Low | High (geometric attractors) |
| Safety via structure | No (post-hoc filtering) | No | Yes (attractor constraints) |

---

## Biological Plausibility

P08 notes four lines of neuroscientific evidence consistent with the biological reality of delay-embedding-like mechanisms:

1. **Cerebellar granule cells:** ~50 billion cells implementing precise temporal delays (1–100ms) for motor timing and sensory prediction — biological delay-line networks for coincidence detection (Braitenberg 1967)
2. **Hippocampal time cells:** Fire at specific delays (1–30 seconds) after an event, maintaining a log-space temporal buffer for episodic memory encoding (Eichenbaum 2014)
3. **Cortical hierarchy and timescales:** Different cortical areas operate at different intrinsic timescales (V1 ~10ms; prefrontal cortex ~seconds) — mirrors MARINA's multi-scale delay structure (Murray et al. 2014)
4. **Synfire chains:** Sequences of neuronal groups with fixed propagation delays — biological delay lines for temporal pattern recognition (Abeles 1991)

**Key observation:** Biological attention modulates gain but preserves temporal structure — more similar to MARINA's fixed delays + modulation than to query-dependent recombination.

---

## Limitations and Counter-Arguments

P08 addresses its own critique honestly:

**Where autoregression succeeds despite theory:**
- Partial reconstruction suffices for many tasks — local tangent structure captures short-term correlations even without global fidelity ("3-day forecast of semantics")
- Scale as brute-force solution — billions of parameters can approximate the manifold via dense trajectory covering, inefficiently but effectively
- Possible emergence via scaling — at extreme scale, attention might learn fixed delay-like patterns implicitly (Anthropic's "superposition" hypothesis); P08's claim is that this is harder and less reliable than building it in explicitly

**When approximate reconstruction might suffice:**
- Creative writing: no "true" manifold — any coherent story is valid
- Conversational agents: contextual appropriateness rather than perfect reconstruction

**Computational trade-offs:**
- Transformers' O(T²) parallelises perfectly across positions; MARINA's delay structure introduces sequential dependencies limiting GPU utilisation
- No pretrained MARINA foundation models exist; Transformers have massive transfer learning infrastructure

**Falsifiability conditions:** The paper explicitly states that it would revise its claims if: (1) a pure Transformer learns fixed delay-like attention patterns achieving MARINA-level basin separation; (2) autoregressive models are shown to naturally converge to diffeomorphic reconstructions; or (3) MARINA systematically underperforms on truth-critical long-context tasks.

---

## Open Theoretical Questions

1. **Optimal delay schedules:** Can Lyapunov exponents and fractal dimension derive the ideal delay structure for a given manifold, or must it be empirically tuned?
2. **Discrete-continuous bridge:** Takens assumes continuous observables; language tokens are discrete. Formalising the relationship between discrete symbol sequences and continuous manifold trajectories remains open (connects to P04's Geofinite Embedding Principle)
3. **Higher-order attractors:** Can the framework handle strange attractors, quasi-periodic behaviour, or attracting tori (not just fixed points and limit cycles)?
4. **Manifold topology learning:** Can a model learn the topology of the semantic manifold (dimension, genus, connectivity) from data alone, enabling adaptive architecture?

---

## Future Work

- Scaling MARINA to 1B–100B parameters
- Multimodal extensions: video (spatial + temporal delays), audio (multi-scale waveform delays), joint vision-language manifolds
- Geofinitism as broader research programme: finite measurement manifolds in quantum mechanics, geometric invariants for cosmology and particle physics, formal differential-topological grounding of semantics

---

## Discussion: From Probabilism to Geofinitism

> "The limitations of autoregression highlight a philosophical necessity: we must move from **Probabilism** (what is likely?) to **Geofinitism** (what is the shape?)."

Hallucination in LLMs is not merely a statistical error — it is a **topological failure**: a divergence from the true manifold because the model's embedding failed to preserve the invariant structure of the truth attractor.

This frames AI alignment as a geometric problem: a model aligned with truth is one whose latent trajectories stay on the truth manifold. Post-hoc filtering is a corrective applied to a system that never had the right inductive bias; Corpus Ancora and geometric training are the alternative.

---

## Re-style Checklist (LaTeX)

- [ ] **Citation date mismatch:** Cite line (p.1) says "February 2025" but author date is "February 2026" — standardise
- [ ] **Theorem numbering:** Uses `[section]` counter (Theorem 4.1, 5.1, 6.1) — clean and consistent as-is; verify cross-references in text (e.g., `Theorem~\ref{thm:uniform_history}`)
- [ ] **Proof sections are labelled "Proof Sketch"** — appropriate for a preprint; note for full journal submission that complete proofs would be required
- [ ] **Table 1** (Architecture Comparison) has narrow column widths (`p{2.8cm}`) — may need `p{3cm}` or landscape orientation for clean print rendering
- [ ] **Section 7.5 (Biological Plausibility)** — extensive; consider moving to appendix if length is constrained
- [ ] **§10 Future Work / §9 Limitations** are both substantial — consider merging or clearly distinguishing scope
- [ ] **No figure files referenced** — UMAP visualisation described verbally (§7.3) but no figure included; either add or remove the reference to projection

---

## Cross-References

### Within School of Geofinitism

| Essay / Paper | Connection |
|---|---|
| P01 | TBT / MARINA — the constructive answer; P08 proves what autoregression cannot do; P01 does it |
| P02 | Pairwise Phase Space Embedding — consistent but distinct argument: P02 shows what attention *is* (Takens, accidentally); P08 shows what autoregression *cannot achieve* (faithful Takens reconstruction) |
| P04 | Finite-Symbol Embedding Theorem — the theoretical licence for P08's argument about discrete tokens; the Geofinite Embedding Principle grounds the claim that delay embedding can work for symbolic systems even where classical Takens fails |
| P05 | Language as Nonlinear Dynamical System — P08 is P05's constructive critique made rigorous; P05 provides the philosophical frame; P08 provides the formal theorems |
| P03 | JPEG Compression — empirical precedent: P03's attractor collapse sequence is evidence of the manifold structure P08 argues autoregression fails to reconstruct |
| ATT_52 | Finite Process Unfolding — exponential delay structure is FPU applied geometrically; the log-space delay selection derives from the same multi-scale unfolding principle |
| ATT_30 | Words as Trajectories — P08 proves that autoregression treats words as predictive tokens; the trajectory model of ATT_30 is what MARINA implements instead |
| ATT_128 | Admissibility Principle — query-dependent attention coordinates are inadmissible as a Takens embedding (Theorem 1); fixed log-space delays are measurement-admissible |
| ATT_65 | Mathematics as Language Dynamics — P08's argument that meaning is geometric rather than probabilistic is the applied instantiation of ATT_65 |

### External Literature

| Reference | Role |
|---|---|
| Takens 1981 | The theorem being applied and defended |
| Elhage et al. 2021 | Anthropic circuit analysis — descriptive transformer dynamics; P08 notes these are descriptive, not prescriptive |
| Gu et al. 2021 (S4); Gu & Dao 2023 (Mamba) | SSM architecture — shown to partially improve on transformers but still vulnerable to Theorem 2 |
| Vaswani et al. 2017 | The architecture being critiqued |
| Langton 1990 | "Computation at the edge of chaos" — historical dynamical context |
| Eichenbaum 2014; Murray et al. 2014; Braitenberg 1967; Abeles 1991 | Neuroscience evidence for biological delay-line mechanisms |
| Data Processing Inequality (Shannon / information theory) | Formal basis for Theorem 3 |

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
