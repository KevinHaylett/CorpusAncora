# P08 Lesson — Autoregression Is Not Takens

**Lesson ID:** P08  
**Paper:** P08  
**College:** College of Machine Intelligence  
**Level:** Advanced  
**Prerequisites:** P02 (Pairwise Phase Space Embedding — essential); P04 (Finite-Symbol Embedding Theorem — strongly recommended); P01 (TBT / MARINA — for the constructive counterpart); ATT_25 or P05 for accessible Takens background; undergraduate familiarity with probability and information theory helpful  
**Pillars:** P3, P1 (primary); P5, P2 (secondary)

---

## Overview

This lesson examines P08 — the formal critique paper of the language dynamics programme. While P02 shows what transformer attention actually *is* (Takens embedding, accidentally), P08 proves what autoregressive training *cannot* do (faithful Takens reconstruction, provably). The paper delivers three independent theorems — Non-Rigid Embedding, Uniform History Treatment, Information Loss — each establishing a different structural reason why the standard autoregressive paradigm cannot satisfy Takens' conditions. These are followed by empirical validation on domain classification tasks (100% basin separation with 1.1M parameters vs GPT-2's 117M), protein folding (sub-angstrom RMSD with 1,000 training structures vs AlphaFold's 170,000+), ablation studies confirming the necessity of log-space delays, and long-context stability tests. The lesson concludes with P08's philosophical pivot: from Probabilism (what is likely?) to Geofinitism (what is the shape?).

---

## Learning Outcomes

By the end of this lesson, students will be able to:

1. State the core claim of P08 precisely: distinguish between (a) the claim that transformers *accidentally perform* Takens-style operations (P02's claim) and (b) the claim that autoregressive training *cannot guarantee* faithful Takens reconstruction (P08's claim) — and explain why these two claims are consistent, not contradictory.

2. Identify and explain the **Fundamental Gap** between autoregression and Takens reconstruction: why minimising cross-entropy loss $\mathcal{L} = -\sum \log P(x_{t+1} | x_{<t+1})$ ensures predictive accuracy but does not guarantee that the model's internal state space is diffeomorphic to the generating manifold $M$.

3. State **Theorem 1 (Non-Rigid Embedding)** and explain the proof: why query-dependent attention weights $\alpha_{t,t-k} = f(Q_t)$ produce a time-varying family $\{\Phi_t\}$ rather than a single map $\Phi: M \to \mathbb{R}^m$; why a time-varying map cannot guarantee stable attractor reconstruction; and what the metaphor of a "wobbly manifold" captures.

4. State **Theorem 2 (Uniform History Treatment)** and explain the proof: the two failure modes of delay selection (redundancy when $\tau$ too small; undersampling when $\tau$ too large); why transformer positional encodings and SSM discretization steps $\Delta$ both fail to enforce geometrically motivated multi-scale delays; and what the empirical evidence of attention collapsing to recent tokens shows about inductive bias.

5. State **Theorem 3 (Information Loss)** and explain the proof: how the Data Processing Inequality ($I(\mathcal{M}_t; s_t) < H(\mathcal{M}_t)$ for lossy compression) applies to finite context windows and fixed state sizes; why language's power-law correlations mean finite truncation always discards trajectory-distinguishing information; and why this makes $\Phi$ non-injective, precluding a diffeomorphism.

6. Analyse the **ablation study results**: explain why linear delays fail (oversampling short timescales), random delays catastrophically fail (no coherent attractors), and learned delays partially fail (collapsing toward recent tokens during training); and articulate what these results confirm about the geometric necessity of log-space delays.

7. Interpret the **empirical results** geometrically: explain what "100% basin separation with 1.1M parameters" means in attractor-theoretic terms; why 10,000× data efficiency is predicted by manifold reconstruction theory; and why the protein folding result (sub-angstrom RMSD with 1,000 training structures) is "theoretically significant" — what property of protein folding makes it amenable to delay-embedding treatment.

8. Distinguish **Memory Fibres from Broad Basins** as they appear in MARINA's UMAP projections: what each looks like geometrically; what tasks produce each; why Transformer latent spaces show "fuzzy clouds" instead; and what this difference implies about hallucination and creativity simultaneously.

9. Explain the **Corpus Ancora methodology**: define convergent paraphrasing, divergent continuations, and explicit attractors; explain the geometric distinction between learning a 1D curve through an $n$-dimensional manifold (narrow training paths) versus learning the full $n$-dimensional volume (Corpus Ancora); and explain the geometric AI safety implication (harmful content on divergent trajectories far from the training manifold).

10. Evaluate P08's **limitations section honestly**: identify where autoregression succeeds despite the theorems (partial reconstruction, brute-force scale, possible emergence at extreme scale); assess when approximate reconstruction suffices; and apply the falsifiability conditions the paper itself proposes.

11. Assess the **biological plausibility argument**: identify and interpret the four neuroscientific findings P08 cites (cerebellar granule cells, hippocampal time cells, cortical hierarchy, synfire chains); explain what distinguishes biological attention from Transformer attention in terms of temporal structure preservation; and evaluate the strength of the evolutionary argument for delay-embedding mechanisms.

12. Articulate the **philosophical conclusion**: explain the shift from Probabilism to Geofinitism in the context of AI; connect hallucination to topological failure (trajectory divergence from the truth attractor) rather than statistical error; and assess whether geometric AI safety (safety through topology) is a stronger foundation than post-hoc filtering.

---

## Core Concepts

### The Pincer Argument — P02 and P08 Together

Understanding P08 requires holding its relationship to P02 clearly:

| Paper | Argument | Target |
|---|---|---|
| **P02** | Transformer attention IS Takens pairwise phase-space embedding — accidentally, imperfectly | What attention *is* |
| **P08** | Autoregressive training CANNOT guarantee faithful Takens reconstruction — provably, in principle | What autoregression *cannot achieve* |

These are not contradictory. Attention accidentally performs Takens-style operations (P02); the autoregressive training objective cannot guarantee this will produce faithful manifold reconstruction (P08). MARINA (P01) resolves the tension by making the Takens structure explicit and intentional.

### The Fundamental Gap

$$\underbrace{\min_\theta \mathcal{L}}_{\text{predictive accuracy}} \quad\neq\quad \underbrace{\Phi: M \overset{\sim}{\to} \mathbb{R}^m}_{\text{diffeomorphic reconstruction}}$$

Training to predict the next token optimises a statistical loss. The geometric guarantee — that the model's internal representation is topologically equivalent to the generating manifold — is a separate requirement that cross-entropy minimisation does not impose.

### Three Theorems — One Table

| Theorem | What It Shows | The Violated Requirement |
|---|---|---|
| **T1: Non-Rigid Embedding** | Attention is query-dependent → time-varying map family $\{\Phi_t\}$, not a single $\Phi$ | Fixed coordinate system |
| **T2: Uniform History Treatment** | No geometric constraint on delay selection → redundancy ($\tau$ too small) or undersampling ($\tau$ too large) | Principled multi-scale delays covering Lyapunov timescales |
| **T3: Information Loss** | Finite context/state compresses history lossily → Data Processing Inequality → $\Phi$ non-injective | Injectivity required for diffeomorphism |

Each theorem is independent. Together they are exhaustive: they cover the coordinate structure, the delay structure, and the information structure of autoregression.

### Delay Structure Ablation — What the Results Show

| Delay Type | Performance | Geometric Reason |
|---|---|---|
| Linear | Near-Transformer | Short timescales oversampled; long-range dependencies missed |
| Random | Catastrophic | No multi-scale structure; no attractor formation |
| Learned | Partially improved, unstable | Training gradient pushes delays toward recency — Theorem 2 reproduced |
| **Log-space (geometric)** | **Robust manifold reconstruction** | Covers multiple Lyapunov timescales by construction |

The ablation is a controlled experiment specifically targeting Theorem 2. If the result were that all delay types worked equally well, Theorem 2 would be undermined. The catastrophic failure of random delays and the instability of learned delays both confirm the theorem.

### Manifold Topology vs Statistical Distribution

| Standard Transformer (Autoregressive) | MARINA (Takens-Based) |
|---|---|
| Latent space: "fuzzy clouds" | Latent space: distinct fibres and basins |
| Factual and creative prompts overlap | Factual prompts → fibres; creative prompts → broad basins |
| Hallucination as statistical misstep | Hallucination as trajectory divergence from truth attractor |
| Safety via post-hoc filtering | Safety via geometric topology — harmful content on far-attractor trajectories |
| Data need: massive (billions of tokens) | Data need: minimal (thousands, once manifold covered) |

### Probabilism vs Geofinitism — The Philosophical Pivot

| Framework | Question | Unit of Analysis | Failure Mode |
|---|---|---|---|
| **Probabilism** | What is likely? | Token probability distribution | Hallucination as wrong probability |
| **Geofinitism** | What is the shape? | Geometric trajectory through manifold | Hallucination as topological failure |

The difference matters practically: a Probabilist approach to AI safety improves statistical calibration. A Geofinite approach builds systems whose architecture geometrically constrains trajectories to stay on the truth manifold. The first is a corrective; the second is a design principle.

---

## Five Pillars in P08

| Pillar | Role in P08 |
|--------|-------------|
| **P3 — Dynamic Flow** (primary) | Autoregression as trajectory in semantic space; attractor collapse in long contexts as dynamical failure; memory fibres and broad basins as two attractor regimes; hallucination as trajectory divergence; Corpus Ancora as manifold-volume training |
| **P1 — Geometric Container** (primary) | The semantic manifold as the container autoregression fails to reconstruct; basin separation as the geometric diagnostic; UMAP projection revealing fibre/basin topology; diffeomorphism as the geometric requirement |
| **P5 — Finite Reality** (secondary) | Three theorems about finite systems: finite context windows (T3), finite state size (T3), finite attention coordinates (T1); MARINA's O(log T) memory and O(1) delay buffer as finite geometric alternatives; all results finite and bounded |
| **P2 — Approximations/Measurements** (secondary) | Data Processing Inequality as information-theoretic measurement bound; basin separation percentage as a geometric measurement; sub-angstrom RMSD as a precision measurement; Lyapunov timescales as the right measurement scale for delay selection |

---

## Discussion Questions

1. Theorem 1 proves that query-dependent attention violates the fixed-coordinate requirement. But P02 argues that attention *is* pairwise phase-space embedding. Reconcile these two claims precisely: what does attention accomplish that is Takens-like, and what does the autoregressive training objective fail to guarantee?

2. Theorem 3 uses the Data Processing Inequality to prove information loss. This is an information-theoretic argument. Does it depend on any assumption about the nature of the semantic manifold $M$? Specifically, what would have to be true about language for finite context windows to be *sufficient* for diffeomorphic reconstruction?

3. The ablation study shows that learned delays collapse toward recent tokens. From an optimization perspective, why might gradient descent produce this result? Is there a training modification (e.g., a delay regularization term) that could prevent collapse without committing to pre-specified log-space delays?

4. The protein folding result is described as "theoretically significant" because protein folding is a physical dynamical system with an underlying energy landscape manifold. Is language analogously a physical dynamical system? What physical substrate would the language manifold correspond to? Does this matter for the validity of applying Takens to language?

5. The biological plausibility section notes that hippocampal time cells maintain a "log-space temporal buffer for episodic memory encoding." If the brain independently evolved log-space delay-like structures for memory, what does this suggest about the universality of the MARINA delay schedule? Could the log-space structure be derived from first principles (e.g., from information-theoretic arguments about optimal multi-scale coverage)?

6. P08 proposes a falsifiability condition: "If a pure Transformer learns fixed delay-like attention patterns at scale, achieving MARINA-level basin separation with no architectural bias, we would revise our claims." Design an experiment that could test this condition. What measurements would constitute strong evidence one way or the other?

---

## Connections to Prior Work

- **P01** (TBT / MARINA): the constructive answer to P08's critique; where P08 proves what autoregression cannot do, P01 shows what a Takens-compliant architecture does instead
- **P02** (Pairwise Phase Space Embedding): consistent but distinct argument; P02 names what attention is (Takens, accidentally); P08 proves what autoregressive training cannot guarantee (faithful reconstruction)
- **P04** (Finite-Symbol Embedding Theorem): provides the theoretical licence for P08's use of delay embedding on discrete tokens; the Geofinite Embedding Principle grounds the claim that geometric reconstruction is possible even for symbolic systems
- **P05** (Language as Nonlinear Dynamical System): the philosophical frame; P08 makes P05's dynamical linguistics thesis rigorous through formal theorems
- **P03** (JPEG Compression): empirical precedent for the attractor structure P08 argues autoregression fails to reconstruct; JPEG compression collapse sequences = trajectory divergence from the manifold
- **ATT_30** (Words as Trajectories): P08 proves formally that autoregression misidentifies the unit (treating words as tokens to predict rather than perturbations of a trajectory)
- **ATT_52** (Finite Process Unfolding): log-space delay structure is FPU applied geometrically; multi-scale unfolding is the right inductive bias
- **ATT_128** (Admissibility Principle): query-dependent attention coordinates are inadmissible as a Takens embedding (Theorem 1 demonstrates the inadmissibility); fixed log-space delays are measurement-admissible

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
