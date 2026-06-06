# P09 Lesson — Static Vector Insufficiency for Natural Language Meaning

**Lesson ID:** P09  
**Paper:** P09  
**College:** College of Machine Intelligence  
**Level:** Intermediate–Advanced  
**Prerequisites:** ATT_30 (Words as Trajectories — strongly recommended); P05 (Language as Nonlinear Dynamical System — philosophical context); P02 (Pairwise Phase Space Embedding — for the Takens connection); basic familiarity with information theory (entropy, mutual information) helpful but not required — the paper provides introductions  
**Pillars:** P1, P3 (primary); P2, P5, P4 (secondary)

---

## Overview

This lesson examines P09 — the foundational critique paper of the language dynamics programme. Where P08 proves that autoregressive *training* cannot achieve Takens reconstruction, P09 goes deeper: it proves that the *representation layer itself* — the static vector embedding of words (Word2Vec, GloVe, fastText) — is mathematically incapable of representing meaning, regardless of what architecture is built on top of it. Three independent proofs establish this: information-theoretic (zero mutual information about word senses), dynamical systems (violates Takens' dimensional requirement), and transduction chain (breaks the geometric chain from acoustic origins). All three converge on the same conclusion and the same alternative: meaning is not a static property of word types but a trajectory in semantic space — and any representation that succeeds must be contextual, sequential, and geometrically structured.

---

## Learning Outcomes

By the end of this lesson, students will be able to:

1. Define **static embedding** precisely — a function $\phi_{\text{static}}: \mathcal{V} \to \mathbb{R}^d$ assigning a single fixed, context-independent vector to each word type — and explain the architectural consequence: for all occurrences of word type $w$, the vector is identical regardless of sense or context.

2. State and explain **Theorem 1 (Polysemy Separation Impossibility)**: prove step-by-step why $I(v_w; s(t)) = 0$ for any static embedding; explain what it means that the posterior $P(s|v_w)$ equals the prior $P(s)$; and apply the Data Processing Inequality to show that no downstream processing can recover sense information destroyed by static averaging.

3. Apply **Theorem 1 to empirical predictions**: explain why static embeddings have a WSD accuracy ceiling of ~60–70%; why performance degrades monotonically with number of senses; why minority senses fail systematically; and interpret the ablation study results (removing $v_w$ barely changes accuracy; replacing it with a contextual vector produces +30pp improvement).

4. State and explain **Theorem 2 (Dynamical Reconstruction Insufficiency)**: identify that static embedding is the degenerate case $m=1$, $\tau=\infty$ of a delay-coordinate embedding; apply Takens' dimensional requirement $m \geq 2d+1$ to show this fails for any non-trivial semantic manifold ($d \geq 1$, so $m$ must be $\geq 3$); and explain the four consequences: dimensional insufficiency, non-diffeomorphism, topological destruction, trajectory collapse.

5. Explain why static embeddings destroy each of these topological properties: **curvature** (semantic shift encoded in trajectory bending → zero in a static point), **recurrence** (loops and cycles → unrepresentable), **basin boundaries** (separatrices between senses → eliminated), **neighbourhoods** (nearby contexts → identical representation).

6. Describe and interpret the **transduction chain** $p(t) \to \text{phonemes} \to \text{graphemes} \to \text{tokens} \to \text{embeddings}$: explain why each of the first three stages is a *controlled* lossy compression (bidirectionally recoverable, preserving geometric structure), and why static embeddings ($T_{4b}$) constitute an *uncontrolled* second compression that breaks chain integrity.

7. Respond to the objection "Takens requires continuous systems but text is discrete": state all four responses P09 gives (the original acoustic signal is continuous; orthography is engineered to preserve geometric structure; transformer success is empirical proof of geometric residue surviving discretization; phonemes are attractor basins in acoustic phase space).

8. State the **Unified Theorem** and its three criteria (information, dynamical, transduction), explain why the three proofs are logically independent, and articulate why their convergence on the same conclusion and the same alternative strengthens the case beyond what any single proof would establish.

9. Distinguish precisely **what static embeddings can and cannot capture**: articulate the photograph-vs-dance analogy; identify tasks for which static embeddings are adequate and tasks for which they are architecturally incapable; and connect this to the concept of "useful fiction" — static embeddings are admissible approximations for some tasks but inadmissible as a complete theory of meaning.

10. Connect the **JPEG compression experiments** (P03) to P09's theoretical framework: explain why systematic attractor collapse under compression (rather than random degradation) is empirical confirmation of Theorem 2; and explain what it means that static embeddings permanently impose the same attractor collapse that maximum compression ($m=1$) produces experimentally.

11. Apply the **Transfictor concept** (P05) to P09's transduction chain: explain how the Transfictor — word as measurement output — is precisely the kind of representation that results from the chain stages $T_1$–$T_3$; and explain why static embeddings ($T_{4b}$) violate the Transfictor's provenance by destroying the sequential and contextual information that the chain preserved.

12. Situate P09 within the full language dynamics programme: explain how it establishes the foundational diagnosis — the *input layer* is broken — that motivates P02 (what attention actually does), P08 (why training objectives fail), and P01 (what an explicitly geometric architecture looks like).

---

## Core Concepts

### The Architecture of the Critique

P09 addresses the representation layer — logically prior to architecture:

```
Static paradigm:    [v_w (fixed)] → Architecture → Output
                         ↑
                    P09 proves: this is broken regardless of architecture

Contextual paradigm: [v_{w,context} (dynamic)] → Architecture → Output
                         ↑
                    This is necessary, not just better
```

### Three Theorems — Independent but Convergent

| Theorem | Perspective | Core Result | Empirical Signature |
|---|---|---|---|
| **T1: Polysemy Separation Impossibility** | Information theory | $I(v_w; s(t)) = 0$ — zero sense information | WSD ceiling ~65%; removing $v_w$ doesn't hurt |
| **T2: Dynamical Reconstruction Insufficiency** | Dynamical systems | $m = 1 < 2d+1$ — degenerate embedding | JPEG collapse is structured, not random; attractor hierarchy |
| **T3: Transduction Chain Violation** | Signal origins | Second uncontrolled compression breaks chain | Order scrambling kills performance; TTS works; $v_w$→context not invertible |

The "multi-vector proof" of the title: three completely independent vectors of attack, all hitting the same target.

### Theorem 1 — The Information Argument

**Setup:** $v_w$ is fixed — the same for every token of type $w$, regardless of sense or context.

**Proof in one step:** If $v_w$ is constant across all tokens of $w$, then knowing $v_w$ tells you nothing about which sense any particular token carries — you already knew the word was $w$. Therefore:

$$P(s(t) | v_w) = P(s(t)) \implies I(v_w; s(t)) = 0$$

**What this means geometrically:** $v_w$ is a weighted centroid of all sense clusters:
$$v_w \approx p_1 c_1 + p_2 c_2 + \cdots + p_k c_k$$

This collapses $k$ clusters to 1 point, pulled toward the most frequent sense. The sense distinctions are averaged away permanently. No algorithm can reconstruct which cluster any given token came from.

**Analogy:** Mixing red and blue paint to get purple. The information about which colour was which is gone. More mixing (downstream processing) cannot unmix.

### Theorem 2 — The Dimensional Argument

A static embedding is the degenerate limit of delay-coordinate embedding:
- Embedding dimension: $m = 1$ (only the current token — no delays)
- Delay: $\tau = \infty$ (delayed terms vanish — no temporal structure)

Takens requires $m \geq 2d+1$ for any manifold of dimension $d \geq 1$. The simplest possible non-trivial attractor ($d=1$, e.g., meanings on a circle) requires $m \geq 3$. Static embeddings always violate this — by at least a factor of 3.

**What this destroys:** Four topological properties that encode meaning are lost when a trajectory is collapsed to a point:

| Property | In a trajectory | In a static point |
|---|---|---|
| Curvature | Direction and rate of semantic shift | Zero — no bending, no turning |
| Recurrence | Loops, cyclic patterns, repeated themes | Cannot be represented |
| Basin boundaries | Separatrices between sense attractors | Eliminated — all senses collapse |
| Neighbourhoods | Slightly different contexts → slightly different vectors | Zero distance — identical regardless of context |

**Analogy:** "Trying to understand a dance by taking one photograph per dancer. You've lost the motion, the rhythm, the paths they trace."

### Theorem 3 — The Transduction Chain

Language originates in continuous acoustic flow. The chain from flow to text is a sequence of *controlled* lossy compressions designed to preserve geometric structure:

```
Continuous acoustic p(t)   — genuinely continuous; Takens applies directly
        ↓ T₁ [controlled]
Phonemes                   — attractor basins of acoustic phase space; recoverable
        ↓ T₂ [controlled]
Graphemes                  — engineered to preserve reconstructability; TTS proves this
        ↓ T₃ [minimal]
Sequential tokens           — segmentation only; order fully preserved
        ↓ T₄a (contextual) — chain maintained: sequence → context-aware vectors ✓
        ↓ T₄b (static)     — chain BROKEN: sequence → single centroid per type ✗
```

**Proof that the first chain is recoverable:** Text-to-speech (reading aloud) works. Speech-to-text (transcription) works. Cross-modal priming (hearing a word primes recognition of its written form) works. These bidirectional transductions prove geometric structure survives $T_1$–$T_3$.

**Proof that static embedding is unrecoverable:** Given $v_w$, you cannot determine which sense was intended, which context it appeared in, or where in a trajectory it sat. The information was destroyed, not compressed.

### The Unified Theorem — Why Convergence Matters

When three completely independent mathematical frameworks (information theory, dynamical systems, signal origins) all deliver the same verdict from different directions:

1. The conclusion is robust — any one framework could in principle be wrong; having all three agree makes it far harder to undermine
2. The guidance is clear — all three point toward the same alternative: context-dependent, sequentially-aware, geometrically structured representations
3. The problem is architectural, not empirical — it cannot be fixed by more data, better training, or higher dimensions

"Like proving a building is unsafe by showing (1) the foundation is cracked, (2) the support beams are too weak, and (3) the materials are faulty."

### What Static Embeddings Are Good For

The paper is careful to be precise. Static embeddings are not useless:

**They can represent:** coarse semantic similarity; broad categorical relationships; some relational structure (gender, number); distributional statistics.

**They cannot represent:** fine-grained sense distinctions; context-dependent modulation; temporal and sequential structure; trajectory dynamics and basin transitions.

**The admissibility assessment:** Static embeddings are *fictionally-admissible* for tasks where coarse similarity suffices. They are *inadmissible* as a complete theory of meaning.

---

## Five Pillars in P09

| Pillar | Role in P09 |
|--------|-------------|
| **P1 — Geometric Container** (primary) | The semantic manifold $\mathcal{M}$ as the container that static embeddings fail to reconstruct; attractor basins as geometric structures; the transduction chain as preserving geometric structure through successive stages; Theorem 2 as a proof about geometric representation requirements |
| **P3 — Dynamic Flow** (primary) | Meaning as trajectory, not location; polysemy as trajectory divergence from same word type; attractor collapse under JPEG compression as geometric flow disrupted; contextual embeddings restoring trajectory structure |
| **P2 — Approximations/Measurements** (secondary) | Every word as a measurement ($v_w$ as a measurement with zero sense information); WSD accuracy as empirical measurement of the information ceiling; the transduction chain as a sequence of measurements; mutual information as the precision metric |
| **P5 — Finite Reality** (secondary) | $m=1$ as a finite degenerate embedding; the requirement $m \geq 3$ as a finite bound on minimum representational structure; the Geofinite acknowledgment that all three proofs hold and are "strengthened" under fully finite mathematical framework |
| **P4 — Useful Fiction** (secondary) | Static embeddings as admissible useful fictions for coarse semantic similarity tasks; inadmissible as complete meaning representations; the photograph analogy — useful for some purposes, fundamentally inadequate for others |

---

## Discussion Questions

1. Theorem 1 proves $I(v_w; s(t)) = 0$ for static embeddings. But static embeddings clearly capture *something* — "king − man + woman ≈ queen" works. What information does a static embedding carry, and how does it carry it if it carries zero sense information? Is there a measure of information content for which static embeddings score well, while still failing the sense-mutual-information test?

2. Theorem 2 characterises static embedding as $m=1$, $\tau=\infty$. Contextual embeddings like BERT are closer to $m > 1$ (multiple attention heads, multiple layers). But they don't use fixed delays — they use learned, query-dependent attention (which P08's Theorem 1 says violates Takens' fixed-coordinate requirement). On the Takens compliance spectrum from $m=1$ (static) to fully compliant (MARINA), where would you place BERT? What is BERT's effective $m$?

3. Theorem 3 argues that writing systems evolved to preserve geometric structure through controlled lossy compression. This is an evolutionary/historical claim embedded in a mathematical proof. How strong is this argument? Can you think of writing systems or encoding schemes where the geometric structure is *not* well preserved — and what would P09's theory predict about disambiguation performance in those systems?

4. The JPEG compression experiments (P03/P09) show that attractor collapse under compression is systematic, not random. P09 interprets this as confirmation that meaning has geometric attractor structure. A sceptic might argue: "The systematic patterns you observe are artifacts of the model's training distribution, not evidence of a semantic manifold." How would you respond? What would the null hypothesis predict, and does the data rule it out?

5. P09 argues that static embeddings are the input layer that "destroys trajectory information before any architecture can reconstruct it." But BERT and GPT — which use static input embeddings followed by contextual processing — achieve very high performance. Does this undermine P09's claim? Or does it support it, and why?

6. P09 claims the inadequacy of static vectors is "architectural, not empirical — not fixable by more data, better optimisation, or larger dimensions." Design an experiment that would test this claim: specifically, what would you observe if you trained Word2Vec on 100× more data, and what would the result (positive or negative) tell you about whether the ceiling is architectural or empirical?

---

## Connections to Prior Work

- **P03** (JPEG Compression): the JPEG attractor collapse experiments cited in §6.1 are from P03; P09 reframes them as empirical validation of Theorem 2 — structured collapse confirms geometric attractor structure that static embeddings permanently destroy
- **P05** (Language as Nonlinear Dynamical System): P05's Transfictor concept is the philosophical companion to P09's transduction chain proof; both argue the word-as-measurement carries the provenance of continuous acoustic flow
- **P08** (Autoregression Is Not Takens): complementary at a different level — P09 critiques the representation layer ($m=1$ input embeddings); P08 critiques the training objective (cross-entropy cannot guarantee manifold reconstruction)
- **P02** (Pairwise Phase Space Embedding): P09 provides the proof that static embeddings are degenerate ($m=1$) while P02 identifies what the transformer actually does (approximate Takens, accidentally); together they frame the problem and the partial solution
- **P01** (TBT / MARINA): the constructive answer — MARINA's exponential delay embedding is the non-degenerate ($m \geq 3$) representation P09 proves is necessary
- **P04** (Finite-Symbol Embedding Theorem): provides the formal licence for applying Takens to discrete symbolic sequences; P09's Theorem 2 invokes Takens for language; P04 justifies this formally
- **ATT_30** (Words as Trajectories): P09 is the formal proof that ATT_30's trajectory model is mathematically necessary; P09's three theorems are the rigorous backing for ATT_30's conceptual argument
- **ATT_52** (Finite Process Unfolding): P09's $m=1$ critique is FPU's inverse — static embeddings collapse the temporal unfolding to a single step; FPU demands the full unfolding
- **ATT_128** (Admissibility Principle): static embeddings are fictionally-admissible for coarse tasks; inadmissible as complete meaning representations — P09 provides the formal criteria for this assessment

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
