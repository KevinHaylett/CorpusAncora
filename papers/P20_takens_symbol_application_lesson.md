# Lesson P20-L — Takens' Theorem Applies to Discrete Symbol Sequences

**Lesson ID:** P20-L  
**Source paper:** P20  
**Title:** *Takens' Theorem Applies to Discrete Symbol Sequences: A Formal Note on Language as a Dynamical System*  
**Series:** Selected Communications  
**Difficulty:** Intermediate  
**Prerequisites:** P04-L (Takens-Haylett Theorem — essential); P05-L (Language as NDS — strongly recommended); P11-L (Takens Applies to Symbol Sequences — strongly recommended)  
**Estimated study time:** 45 minutes

---

## Learning Objectives

After completing this lesson you will be able to:

1. State Takens' theorem precisely and identify the three components to which smoothness applies
2. Explain why discrete or symbolic data does not violate the theorem's conditions
3. Define the symbolic measurement function h_s = κ ∘ h_r and trace the compositional structure
4. Articulate the interpretive model: words as symbolic measurements of a continuous underlying dynamical system
5. Construct a symbolic delay embedding vector W_t and explain what it reconstructs
6. State the generating partition condition and explain its role in the formal proof
7. Explain the claim that quantization commutes with delay embedding up to topological equivalence
8. Respond to the "smooth signals" objection using the theorem's own conditions

---

## Key Idea 1 — What the Theorem Actually Requires

### The Theorem

Takens' theorem states: for a generic smooth dynamical system φ : M → M on a compact manifold M of dimension d, and a smooth measurement function h : M → ℝ, the delay embedding map

$$\Phi_{h,\phi}(x) = \bigl(h(x),\, h(\phi(x)),\, h(\phi^2(x)),\, \ldots,\, h(\phi^{2d}(x))\bigr)$$

is an embedding of M into ℝ^{2d+1}.

### The Misreading

A common but mistaken reading: since h maps to ℝ and every textbook example uses real-valued data (temperature, voltage, mechanical position), the data sequence {h(φ^t(x₀))} must itself be smooth. On this reading, discrete symbols appear to violate the theorem.

### The Correct Reading

P20 locates the smoothness requirements exactly:

| Component | Smoothness required? |
|---|---|
| State space M | Yes — differentiable manifold |
| Dynamics φ | Yes — smooth map |
| Measurement function h | Yes — smooth |
| Data sequence {h(φ^t(x₀))} | **No** |

The data sequence is a set of samples taken at discrete times. It can be real-valued, integer-valued, or categorical. The proof uses the smoothness of φ and h to guarantee that nearby states produce nearby measurement sequences — but says nothing about the smoothness of those sequences themselves.

### Exercise 1.1

(a) Takens' theorem guarantees an *embedding* — an injective map with continuous inverse. The proof uses smoothness of φ and h. Why does injectivity require smoothness of the dynamics and measurement function, but not smoothness of the data sequence? What would break if φ were not smooth?

(b) A thermometer reading is recorded every hour, producing a sequence {23.1, 23.4, 22.8, ...}. This is a real-valued discrete-time sequence. Does Takens' theorem apply to it? Is the data smooth? Is the underlying process smooth? How does this case resemble the language case?

(c) The misreading arises from textbook examples. Name two fields outside language where Takens-style delay embedding is applied to obviously non-smooth or discrete data. What does the success of those applications tell you about where the smoothness requirement actually lies?

---

## Key Idea 2 — Symbolic Measurements as Compositions

### The Compositional Structure

Let Σ be a finite set of symbols. P20 defines a symbolic measurement function h_s : M → Σ as:

$$h_s = \kappa \circ h_r$$

where:
- h_r : M → ℝ^k is a smooth real-valued measurement function
- κ : ℝ^k → Σ is a quantization or symbol-assignment function (measurable; generally discontinuous)

The symbolic sequence s_t = h_s(φ^t(x₀)) = κ(h_r(φ^t(x₀))).

The structure matters: the smoothness comes from h_r and φ. The quantization κ then discretises the output. Takens' theorem applies to the real-valued sequence h_r(φ^t(x₀)); the symbolic sequence s_t is a coarsened version of the same trajectory.

### What Coarsening Does

Quantization collapses nearby states into the same symbol. Two states x and y that are close in M may both map to the same word — κ cannot distinguish them. This loss of injectivity in h_s is recovered in the delay embedding space: by stacking enough delayed copies, the sequence S_t = (s_t, s_{t-τ}, ..., s_{t-(E-1)τ}) can again distinguish states that any single symbol cannot.

### Exercise 2.1

(a) "Injectivity is lost in h_s but recovered in the delay embedding space." Explain this geometrically. Draw a simple manifold (a circle, for example), partition it into four regions labelled A, B, C, D, and show how a single symbol reading is ambiguous but a delay vector (two readings) may be injective.

(b) The quantization function κ is discontinuous at partition boundaries. Why does this not prevent the delay embedding from working? Under what condition on the partition does the formal proof guarantee that Ψ is continuous on a dense open set?

(c) h_s = κ ∘ h_r is one composition. Could you add further layers — say, a second quantization κ₂ applied after the delay embedding? What would this produce, and would Takens' theorem still apply? Where does the compositional chain terminate?

---

## Key Idea 3 — Words as Measurements of a Continuous System

### The Interpretive Model

P20 proposes a concrete model for how words arise as symbolic measurements:

- A continuous dynamical system (neural, articulatory, or semantic) evolves in a low-dimensional manifold M
- Each word w corresponds to a region R_w ⊂ M in the state manifold — different words tile the manifold into labelled regions
- The word produced at time t is w_t = f(x_t), where f returns the symbol whose region contains the current state x_t

This is exactly the structure h_s = κ ∘ h_r: f is the measurement-then-quantization composition.

### The Delay Embedding for Language

Given this model, a sufficiently long word sequence {w_t} allows reconstruction of a space equivalent to M (up to diffeomorphism) via:

$$W_t = (w_t,\, w_{t-\tau},\, w_{t-2\tau},\, \ldots,\, w_{t-(E-1)\tau})$$

This is the foundation of the Takens-based language model. The delay vector W_t is not a collection of independent word co-occurrences — it is a coordinate system in the reconstructed state space of the underlying continuous system.

### Connection to the Corpus

This interpretive model connects directly to:
- **P04**: the Takens-Haylett Theorem — applies delay embedding to biological signal sequences
- **P05**: language is a nonlinear dynamical system — the manifold M is the linguistic attractor
- **P08**: autoregression is not Takens — the next token is not a stochastic prediction but a measurement of a deterministic trajectory
- **P11**: the earlier formal treatment of the same claim, extended here with the appendix proof

### Exercise 3.1

(a) "Each word w corresponds to a region R_w ⊂ M." What shapes might these regions have? Are they required to be convex? Equal-sized? Simply connected? What happens to the delay embedding if the regions are poorly shaped — very thin, or with long narrow protrusions?

(b) The model posits "a continuous dynamical system (neural, articulatory, or semantic)." These are three different candidate systems. Do they make different predictions about the delay embedding? If M is a neural manifold, what would the reconstructed attractor represent? If M is semantic, what would it represent?

(c) P08 argued that autoregression treats the next word as a stochastic prediction from context, whereas Takens treats it as a deterministic measurement of an underlying trajectory. In light of P20's formal clarification, restate this distinction. What exactly is being predicted in each case, and what does P20's proof change about the status of the Takens claim?

---

## Key Idea 4 — The Formal Proof: Quantization Commutes with Delay Embedding

### The Setup

The appendix formalises the core claim. Given:
1. Smooth compact system (M, φ) with M ⊂ ℝ^m
2. Smooth measurement h_r : M → ℝ^k
3. Finite partition P = {P₁,...,P_N} of ℝ^k

The quantized symbolic measurement is h_s(x) = i if h_r(x) ∈ P_i. The symbolic delay embedding is S_t = (s_t, s_{t-τ}, ..., s_{t-(E-1)τ}) ∈ {1,...,N}^E.

### The Key Relationship

The quantized embedding satisfies:

$$\Psi(x) = \kappa^E \circ \Phi(x)$$

where Φ is the Takens delay embedding and κ^E applies the quantization componentwise. The diagram commutes: quantizing then embedding gives the same result as embedding then quantizing (up to measure-zero boundary effects).

### Continuity and Injectivity

**Continuity**: κ is constant on open subsets of ℝ^k (the interiors of partition cells). The boundaries have measure zero. Therefore Ψ is continuous on a dense open subset of M — the measure-zero set where the trajectory hits partition boundaries is the only exception.

**Injectivity**: requires the **generating partition condition** — the symbolic sequence {s_t} uniquely determines the asymptotic state in M up to measure zero. Under this condition, if Ψ(x) = Ψ(y), the entire symbolic sequence is identical, which forces x = y up to measure zero for embedding dimension E ≥ 2 dim(M) + 1.

**Result**: Ψ provides a topological conjugacy between the original system and the symbolically reconstructed system on a full-measure subset of M. The attractor's topological structure is preserved.

### The Generating Partition Condition

This is the strong assumption. A generating partition is one where the symbolic sequence uniquely identifies the state. For a generic system and fine enough partition, generating partitions exist (by the theory of symbolic dynamics). In practice, for language data, exact generating-partition status is not required — only that the embedding preserves enough structure for the downstream task.

### Exercise 4.1

(a) "The diagram commutes: quantizing then embedding gives the same result as embedding then quantizing." Draw the commutative diagram explicitly. What are the four objects in the diagram, and what are the four arrows? What does commutativity mean geometrically?

(b) The generating partition condition is called "strong." Give an example of a partition that is generating and one that is not. What goes wrong with a non-generating partition in the delay embedding?

(c) "For language data, exact injectivity is not required." P19 (Clay stress test) made a similar move: exact solutions are not required, only measured separability. Is this a pattern? What is the FSM justification for relaxing exact formal conditions to measured/practical ones?

---

## Synthesis — P20's Role in the School

P20 is short — five pages — but its role is structural. The entire Takens-based language modelling programme (P04, P05, P08, P11, and the TBT protein papers P15–P17) depends on the move of applying delay embedding to symbolic data. If that move is mathematically illegitimate, the programme collapses. P20 shows the move is legitimate and provides the proof.

The paper is labelled *Selected Communications* — a category that signals precision and directness: it responds to a specific objection with a specific argument. It is not a speculative exploration; it is a formal defence.

The refutation table in §3 is worth memorising:

| Claim | Reality |
|---|---|
| "Takens requires smooth signals" | No — smooth dynamics and measurement function, not smooth data |
| "Discrete symbols violate the theorem" | No — symbols are valid measurements from a continuous system |
| "You can't apply Takens to text" | Yes you can — text is a discrete-time symbolic sequence from a continuous process |
| "The reconstruction won't work with discrete data" | It does — confirmed across symbolic dynamics, Boolean networks, cellular automata, and the TBT language model |

The burden of proof has been reversed: those who claim symbolic measurements violate Takens' theorem must show where in Takens' original proof the violation occurs. They will not find it.

---

## Consolidation Questions

1. P20 says "injectivity is recovered in the delay embedding space" after being lost through quantization. What does this mean for the information content of a single word vs. a delay vector of words? Does this support or challenge the idea that context is essential to meaning?

2. The appendix proof assumes the partition is generating. The generating partition condition comes from symbolic dynamics — the study of discrete symbols arising from continuous maps. Does the existence of a generating partition for language imply that language is deterministic? What would determinism mean in this context?

3. P05 argued that language is a nonlinear dynamical system. P20 provides a mathematical justification for measuring that system through word sequences. Together, do they establish that language has an attractor? What would the attractor of language look like?

4. The paper acknowledges that "the burden of proof now rests on those who claim otherwise." Is this an appropriate rhetorical move in a formal mathematical note? Under what conditions is reversing the burden of proof legitimate — and when is it premature?

5. P20 is a short targeted paper in the "Selected Communications" series rather than a full paper. Does the length affect its evidential weight? Can a five-page note make a decisive mathematical contribution?

---

## Further Reading

- **P04-L** (Takens-Haylett Theorem) — the foundational application of delay embedding to biological sequences; P20 provides its formal justification
- **P05-L** (Language as NDS) — the interpretive frame: M as the linguistic manifold that P20 shows is reconstructable
- **P08-L** (Autoregression Is Not Takens) — the negative argument that P20 supports: autoregression is not a delay embedding of a deterministic trajectory
- **P11-L** (Takens Applies to Symbol Sequences) — the earlier treatment of the same claim; P20 is the formal note with appendix proof
- **P12-L** (Trajectory-Based Computation) — computation as trajectory; P20's formal model makes this precise for language
- **ATT_81-L** (Functional Symbolic Trajectory) — the FSM framework within which word sequences are FSTs; P20 grounds the FST formalism mathematically

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
