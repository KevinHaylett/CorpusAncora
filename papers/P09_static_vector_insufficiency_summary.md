# P09 — Static Vector Insufficiency for Natural Language Meaning

**Paper ID:** P09  
**Title:** Static Vector Insufficiency for Natural Language Meaning: A Multi-Vector Proof — The Reconstruction of Meaning  
**Author:** Kevin R. Haylett  
**Date:** February 2026  
**Citation:** Haylett, K.R. (2026). *Static Vector Insufficiency for Natural Language Meaning: A Multi-Vector Proof.* Geofinitism. https://finitemechanics.com/papers/static-vector-insufficiency.pdf  
**License:** Creative Commons CC BY-ND 4.0  
**College (primary):** College of Machine Intelligence  
**College (secondary):** College of Language Dynamics  
**Pillars (primary → secondary):** P1, P3 (primary); P2, P5, P4 (secondary)  
**Status:** Stable — three full theorems with detailed proofs; empirical validation; unified synthesis theorem; extensive plain-language commentary throughout  
**Position in sequence:** The foundational critique paper for the entire language dynamics programme. While P08 proves that autoregression cannot achieve Takens reconstruction, P09 goes deeper: it proves that the *input layer* itself — the static vector embedding — is fundamentally incapable of representing meaning, regardless of the architecture built on top of it. This is the diagnosis that motivates the full programme: not just a better architecture (P01), but a different theory of what semantic representation must be.  
**Keywords:** Static Embeddings, Word2Vec, GloVe, Polysemy, Word-Sense Disambiguation, Information Theory, Takens Embedding, Transduction Chain, Contextual Embeddings, Semantic Manifold

---

## Abstract

> We prove that static type-level vector embeddings — the foundation of classical word representation methods like Word2Vec, GloVe, and fastText — are fundamentally insufficient for representing natural language meaning. Through three independent mathematical arguments, we demonstrate that assigning a single fixed vector to each word type constitutes an irreversible compression that destroys essential information.
>
> First, from information theory, we prove that static vectors carry zero mutual information about specific word senses, making perfect disambiguation impossible even in principle. Second, from dynamical systems theory, we show that static embeddings represent a degenerate case (embedding dimension $m=1$) that violates Takens' theorem requirements for faithful manifold reconstruction, destroying the trajectory structure essential to meaning. Third, from the perspective of transduction chains, we demonstrate that static vectors perform a second lossy compression on language that has already been transduced from continuous acoustic dynamics, breaking the geometric chain that connects meaning to its physical origins.
>
> These three proofs converge on a single conclusion: meaning is not a static property of word types but emerges through dynamic trajectories in semantic space. We validate these theoretical predictions with empirical evidence from compression experiments showing systematic attractor collapse, word-sense disambiguation performance ceilings, and the dramatic improvements achieved by contextual embeddings.

---

## Architectural Note

P09 addresses a question that precedes architecture: before asking *how* to model language, we must ask *what* a word representation must capture. The answer the programme gives is: trajectories, not points. P09 provides the formal proof that this answer is not a preference but a mathematical necessity.

The paper has an unusual structure — every proof section includes a "Plain language" summary after each formal step, making the arguments accessible to readers without advanced mathematics. This reflects the paper's dual role: a formal mathematical proof for technical readers, and a philosophical-scientific argument for the broader audience of the School.

The three proofs are logically independent: if any two were undermined, the third would still establish insufficiency. Their convergence on the same conclusion — and on the same alternative — gives the result its force.

---

## Mathematical Framework and Key Definitions

### Core Definitions (formally stated)

**Static Embedding:** A function $\phi_{\text{static}}: \mathcal{V} \to \mathbb{R}^d$ mapping each word *type* to a single fixed vector, context-independent. For all occurrences $t_1, t_2$ of word type $w$, $\phi_{\text{static}}(w)$ is identical regardless of context.

**Word Type vs Token:** A word type $w \in \mathcal{V}$ is the dictionary entry (abstract). A token is a specific occurrence of that type in context. "Bank" (type) may appear a million times (tokens) across a corpus.

**Word Sense and Polysemy:** A word type $w$ is polysemous if it has multiple distinct senses $S = \{s_1, s_2, \ldots, s_k\}$. Senses are *distinct* if context distributions differ: $P(C|s_i) \neq P(C|s_j)$ and $I(C; s) > 0$.

**Semantic Manifold and Trajectories:** Language is modelled as a dynamical system on a semantic manifold $\mathcal{M}$. Meaning is represented as trajectories $\gamma: [0,T] \to \mathcal{M}$. Attractors are stable regions; different senses correspond to different attractor basins.

### Foundational Assumption (Geofinite)

The paper works within classical mathematical analysis for accessibility, but explicitly acknowledges: all measurements have finite precision; numerical values represent fuzzy boundaries; infinite-precision $\mathbb{R}$ is a useful idealisation. The paper notes that all three proofs hold — and "are strengthened" — under a fully finite mathematical framework.

---

## Theorem 1 — Information-Theoretic Impossibility

### Statement

**Polysemy Separation Impossibility:** Let $w$ be a polysemous word with at least two distinct senses. Let $\phi_{\text{static}}$ assign $w \mapsto v_w$. Then for any token occurrence $t$ of $w$ in context $c$:

$$I(v_w;\, s(t)) = 0$$

The static vector carries *zero* mutual information about the specific sense of any token occurrence.

**Corollary 1:** Perfect word-sense disambiguation using only $v_w$ is impossible.  
**Corollary 2:** The compression is lossy and irreversible — sense distinctions cannot be recovered by any subsequent processing.

### Proof Structure (five steps)

**Step 1:** $v_w$ is determined by training over all occurrences of $w$ — it is constant with respect to both the specific context $c$ and the sense $s(t)$. It does not vary across different uses of $w$.

**Step 2:** Since $v_w$ is constant, $P(s(t) | v_w) = P(s(t))$ — the posterior equals the prior. Knowing $v_w$ provides no additional information beyond knowing the word type. Therefore $I(v_w; s(t)) = H(s(t)) - H(s(t)) = 0$.

**Step 3:** $v_w \approx \sum_i p_i c_i$ (a weighted centroid over sense clusters). This is many-to-one: all occurrences, regardless of sense, map to the same point. Information is irreversibly destroyed.

**Step 4:** Data Processing Inequality — since $I(v_w; s(t)) = 0$, it follows that $I(g(v_w); s(t)) = 0$ for *any* downstream function $g$. No neural network, no dimensionality reduction, no clustering can recover sense information from $v_w$ alone.

**Step 5:** Disambiguation requires $P(s | v_w, c) = P(s | c)$ — all discriminative power comes from context. The word vector contributes nothing. Performance ceiling determined entirely by context informativeness.

### Empirical Confirmation

Static embeddings achieve ~60–70% accuracy on standard WSD benchmarks (SemEval, Senseval). Performance degrades monotonically with number of senses ($r \approx -0.85$ between sense count and accuracy). Controlled ablation:

| Condition | WSD Accuracy |
|---|---|
| Static vector $v_w$ only (no context) | Random baseline (frequency-weighted) |
| Context only (zero vector for $v_w$) | ~65% — barely changes |
| Contextual embedding (BERT) | ~95% |

*The word vector contributes essentially nothing. Removing it barely changes performance. Replacing it with a contextual vector produces a 30-point improvement.*

---

## Theorem 2 — Dynamical Systems Insufficiency

### Statement

**Dynamical Reconstruction Insufficiency:** Static embedding $\phi_{\text{static}}: \mathcal{V} \to \mathbb{R}^D$ is equivalent to a delay-coordinate embedding with dimension $m = 1$ and delay $\tau = \infty$ (no temporal structure). For any semantic attractor of dimension $d \geq 1$:

1. **Dimensional insufficiency:** $m = 1 < 2d + 1$, violating Takens' requirement
2. **Non-diffeomorphism:** The mapping is many-to-one and not invertible
3. **Topological destruction:** Curvature, recurrence, and basin boundaries are destroyed
4. **Trajectory collapse:** Different semantic trajectories for the same word type all map to the same single point

### Proof Structure (six steps)

**Step 1:** A proper delay-coordinate embedding for token sequence $w_1, w_2, \ldots, w_n$ would be $\Phi(w_i) = (h(w_i), h(w_{i-\tau}), h(w_{i-2\tau}), \ldots)$ — depending on preceding tokens. A static embedding is the degenerate case $m=1$, $\tau=\infty$: zero-order embedding with no temporal structure.

**Step 2:** Takens requires $m \geq 2d+1$. For $d=1$ (simplest non-trivial attractor): requirement is $m \geq 3$; static gives $m=1$. For any $d \geq 1$: $2d+1 \geq 3 > 1$. Static embeddings always violate this for any non-trivial semantic manifold.

**Step 3:** Static embedding fails injectivity — both financial-bank and river-bank trajectories through $\mathcal{M}$ map to the same $v_\text{bank}$. Many-to-one mapping cannot be a diffeomorphism.

**Step 4:** Topological properties destroyed: *curvature* (trajectory bending indicating semantic shift → zero curvature in a static point); *recurrence* (loops and cycles → cannot be represented in a single point); *basin boundaries* (separatrices between senses → eliminated); *neighbourhoods* (nearby contexts → identical representation).

**Step 5:** Polysemy as trajectory divergence — different senses start from the same word type but flow in different directions, settling in different attractor basins. A proper delay embedding captures this divergence (different preceding/following tokens → different coordinates). Static embeddings make the divergence invisible.

**Step 6:** Contextual embeddings (BERT, GPT) are functionally closer to delay-coordinate embedding — $v_i = f(w_i, w_{i-1}, \ldots, w_{i+1}, \ldots)$ — varying with position and context. They partially restore the trajectory structure static embeddings destroy.

---

## Theorem 3 — Transduction Chain Violation

### Statement

**Transduction Chain Violation:** Natural language meaning originates in continuous acoustic dynamics: pressure wave $p(t)$ produced by vocal system dynamics. The representational chain is:

$$p(t) \xrightarrow{T_1} \text{phonemes} \xrightarrow{T_2} \text{graphemes} \xrightarrow{T_3} \text{tokens} \xrightarrow{T_4} \text{embeddings}$$

Each $T_i$ is a controlled, partially-reversible lossy compression designed to preserve semantic structure. A static embedding function performs:

$$\phi_{\text{static}}(w) \approx \sum_i p_i c_i$$

This constitutes a **second irreversible discretization** — an uncontrolled compression that breaks the transduction chain by eliminating sequential relationships.

### The Transduction Chain (detailed)

| Stage | Transformation | Character |
|---|---|---|
| $T_1$: Acoustic → Phonemes | Continuous waveform $p(t)$ → discrete phoneme categories | Controlled lossy: maps attractor basins of acoustic phase space to symbols; bidirectionally recoverable |
| $T_2$: Phonemes → Graphemes | Phoneme sequences → written symbols (alphabetic, logographic) | Controlled lossy: orthographic neighbours preserve phonetic relationships; writing evolved to maintain reconstructability |
| $T_3$: Graphemes → Tokens | Written text → token segmentation | Minimal loss: mostly segmentation, sequence order fully preserved |
| $T_4a$ (contextual): Tokens → Contextual vectors | Each token → vector *in context* | Chain integrity maintained: sequential relationships preserved; different positions → different vectors |
| $T_4b$ (static): Tokens → Static vectors | Each word *type* → fixed vector | **Chain broken**: sequential information destroyed; all senses → one centroid; unrecoverable |

**Key distinction:** The first three stages preserve distinctions between *different* meanings (different words → different symbols). Static embedding ($T_{4b}$) destroys distinctions between different meanings of the *same* word.

**Evidence of chain integrity through $T_1$–$T_3$:** Bidirectional transduction works — humans can read text aloud (text → speech) and transcribe speech (speech → text). Reading comprehension approaches listening comprehension. Cross-modal priming (hearing a word primes recognition of its written form) proves representations are compatible across stages.

**Static embeddings cannot be inverted:** Given $v_w$, one cannot determine which sense was intended, which context it appeared in, or which position in a trajectory. Unlike writing (designed to be recoverable), static averaging is a one-way operation.

### Why Takens Applies Despite Discretization (four responses)

The paper directly addresses the objection "Takens requires continuous systems; text is discrete":

1. Takens applies rigorously to the acoustic signal $p(t)$, which is genuinely continuous
2. Orthography is an *engineered* finite transduction — writing systems evolved to preserve enough structure for successful communication (analogous to sampling at the Nyquist rate)
3. Empirical proof: the success of transformers in performing approximate delay-coordinate embedding on text sequences demonstrates that sufficient geometric residue survives discretization
4. Phonemes are attractor basins in acoustic phase space — representing attractor regions by discrete labels is standard practice in dynamical systems analysis

---

## Unified Theorem — Convergence of All Three Proofs

**Fundamental Insufficiency of Static Vector Embeddings:** Let meaning be characterised as (a) contextually dependent — multiple senses per word type; (b) dynamically generated — emerges through temporal sequences and trajectories; (c) physically grounded — originates in continuous acoustic dynamical systems.

Then $\phi_{\text{static}}$ is fundamentally insufficient because:

| Criterion | Condition | Consequence |
|---|---|---|
| **Information** | $I(\phi_{\text{static}}(w); s_{\text{token}}) = 0$ | Irreversible sense loss; insurmountable disambiguation ceiling |
| **Dynamical** | $m = 1 < 2d+1$ (for $d \geq 1$) | Cannot faithfully reconstruct semantic attractor manifolds |
| **Transduction** | Performs second uncontrolled compression | Breaks geometric chain from continuous acoustic origins |

These criteria are independent yet mutually reinforcing: each alone proves insufficiency; together they demonstrate the problem is architectural, not empirical — not fixable by more data, better optimisation, or larger dimensions.

**Corollary:** Any representation achieving (a) $I > 0$ for sense information, (b) $m \geq 3$ delay coordinates, and (c) chain integrity, will necessarily be context-dependent and cannot assign single fixed vectors to word types.

### What Static Embeddings Can and Cannot Capture

| Can capture | Cannot capture |
|---|---|
| Coarse semantic similarity (synonyms, category members) | Fine-grained sense distinctions (bank₁ vs bank₂ vs bank₃) |
| Broad categorical relationships | Context-dependent meaning modulation |
| Some relational structure (king–queen, gender, number) | Temporal and sequential structure |
| Distributional statistics (co-occurrence patterns) | Trajectory dynamics and attractor basin transitions |

**Analogy:** Static embeddings are to language what photographs are to dance. Photographs capture spatial relationships and appearance but cannot show motion, timing, or flow. "Useful for some purposes (identifying dancers, positions); fundamentally inadequate for others (learning choreography, understanding rhythm)."

---

## Empirical Validation

### JPEG Compression Experiments (P03 Connection)

The paper cites the JPEG compression experiments (P03) as direct empirical validation of Theorem 2. JPEG compression applied to GPT-2's embedding matrix at decreasing quality levels produces *systematic attractor collapse* — not random degradation:

| Quality | Attractor State | Interpretation |
|---|---|---|
| 95–75% | Categorical rigidity, tautological responses | Fine-scale trajectory curvature lost; system falls into nearest stable patterns |
| 50–25% | Philosophical self-reference, recursive loops | "Philosophical attractor" — abstract self-referential basin |
| 10–5% | Aggressive, paranoid, violent language | Primitive aggression attractor — deeper, more primal basin |
| 1% | Zen-like paradox, Koan-like statements | Deepest attractor — maximally simple, maximally stable |

Critical observation: "The progression is *not random*. Each compression level reliably produces the same type of response. Different prompts at the same compression level fall into the same attractor." This confirms meaning has geometric attractor structure. Static embeddings, as $m=1$ representations, perform this collapse *permanently*.

### Word-Sense Disambiguation Benchmarks

Empirical ceiling for static embeddings: ~60–70% accuracy on SemEval/Senseval benchmarks. The ceiling degrades predictably:

- 2 senses: ~75–80% accuracy
- 3–4 senses: ~65–70% accuracy  
- 5+ senses: ~55–60% accuracy
- Correlation: $r \approx -0.85$ between sense count and accuracy

For words with dominant sense frequency >80%: majority sense accuracy ~90% (trivially achieved by guessing); minority senses ~30–40% (systematic failure).

### Contextual Embedding Improvements (All Benchmarks)

| Task | Static | Contextual (BERT/GPT) | Improvement |
|---|---|---|---|
| Word-sense disambiguation | ~65% | ~95% | +30 pp |
| Named entity recognition | ~85% | ~95%+ | +10 pp |
| Question answering | ~70% | ~90%+ | +20 pp |
| Sentiment analysis | ~80% | ~95%+ | +15 pp |
| Coreference resolution | ~65% | ~85%+ | +20 pp |

Each improvement aligns with a specific theorem: WSD improvements confirm Theorem 1 ($I > 0$ restored); scaling with context window size confirms Theorem 2 (more delays → better manifold reconstruction); word-order scrambling degrading performance confirms Theorem 3 (sequence = trajectory).

---

## Implications

**For linguistics:** Static vectors are not merely suboptimal — they violate fundamental mathematical requirements from three independent fields. The shift from Word2Vec to BERT is not just empirical progress — it is a move toward mathematical correctness.

**For AI architecture:** Static embeddings are the input layer that destroys the trajectory information before any architecture can reconstruct it. Any architecture receiving static vectors must rebuild sequential relationships from impoverished input, re-inferring structure that was already present.

**For the Geofinite programme:** The paper situates the TBT (P01) and Corpus Ancora as the constructive response. Fully geometric, trajectory-based methods do not merely improve on static embeddings — they are what the mathematics demands.

**Performance ceiling as architectural fact:** The ~70% WSD ceiling is not a training problem. It is a consequence of the $I(v_w; s) = 0$ result. No amount of additional data or better optimisation changes it.

---

## Re-style Checklist (LaTeX)

- [ ] **Bibliography:** Uses `biber` backend with `apa` style; `\begin{filecontents}{refs.bib}` block has a syntax error on line 30 (stray text inside the bib entry) — fix before compilation
- [ ] **Section `\ref{sec:thm1}` etc.** — cross-references use `\ref`; verify all `\label{}` tags are present on each theorem
- [ ] **"Plain language" notes** — formatted as `\noindent\textbf{Plain language:}` throughout; consistent and correct; consider a custom `\plainlanguage{}` environment for visual distinction
- [ ] **Theorem 2 Step 2** — uses concrete cases ($d=1$, $d=2$) with tabular presentation in itemize; clean but could be a formal table for print
- [ ] **`% Due to length constraints, I'll continue in the next part...`** comment at line 851 — an in-document author note, should be removed before submission
- [ ] **License block** (lines 72–73) — "This work is shared under the Creative Commons Licence.Creative Commons CC BY-ND 4.0 License" has a missing space and line break; format as a footnote or separate block

---

## Cross-References

### Within School of Geofinitism

| Essay / Paper | Connection |
|---|---|
| P03 | JPEG Compression — the JPEG compression experiments cited in §6.1 are the experiments reported in P03; P09 reframes P03's results as empirical validation of Theorem 2 (attractor collapse = $m=1$ static vectors losing geometric structure) |
| P05 | Language as Nonlinear Dynamical System — P05's Transfictor concept is the philosophical companion to P09's transduction chain proof; a Transfictor is precisely the output of the chain that static vectors break; P09 provides the formal proof |
| P08 | Autoregression Is Not Takens — complementary critique: P08 proves autoregressive training cannot achieve Takens reconstruction; P09 proves the *input layer* cannot represent meaning; together they are a complete critique of the standard paradigm |
| P02 | Pairwise Phase Space Embedding — P09 provides the proof that static vectors are degenerate ($m=1$) while P02 identifies what the transformer actually does (approximate Takens); P09 is the motivation; P02 is the reidentification |
| P01 | TBT / MARINA — the constructive answer to P09's critique; MARINA's exponential delay embedding is precisely the non-degenerate ($m \geq 3$) representation P09 proves is necessary |
| P04 | Finite-Symbol Embedding Theorem — P04 provides the formal licence for applying Takens to symbolic systems; P09's Theorem 2 invokes Takens for language sequences; P04 justifies this for discrete symbolic systems |
| ATT_30 | Words as Trajectories — P09 provides the three formal proofs that ATT_30's trajectory model is mathematically necessary, not merely a conceptual preference |
| ATT_52 | Finite Process Unfolding — P09's delay-coordinate perspective: temporal structure must be unfolded from the sequence; static vectors collapse this unfolding to a single step ($m=1$) |
| ATT_65 | Mathematics as Language Dynamics — P09's transduction chain (acoustic → phonetic → orthographic → token → meaning) is the applied instantiation of ATT_65's thesis that language is a finite dynamical medium |
| ATT_128 | Admissibility Principle — static embeddings are inadmissible as a complete representation of meaning: they fail the information criterion, the dynamical criterion, and the transduction chain criterion simultaneously |

### External Literature

| Reference | Role |
|---|---|
| Word2Vec (Mikolov et al. 2013) | The paradigm being critiqued — type-level static embeddings |
| GloVe (Pennington et al. 2014) | Same paradigm: co-occurrence statistics → single vectors |
| FastText (Joulin et al. 2016) | Same paradigm: character n-gram extension, still type-level |
| ELMo (Peters et al. 2018) | First major contextual embedding — beginning of the move toward trajectory representation |
| BERT (Devlin et al. 2018) | Contextual embeddings achieving $I > 0$; validates the three theorems |
| Takens 1981 | Foundation for Theorem 2 — dimensional requirement $m \geq 2d+1$ |
| Shannon 1948 | Basis for Theorem 1 — entropy, mutual information, Data Processing Inequality |
| SemEval / Senseval benchmarks | Empirical data for WSD performance ceilings |

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
