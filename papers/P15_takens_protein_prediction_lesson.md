# Lesson P15-L — Takens-Based Transformer for Protein Structure Prediction

**Lesson ID:** P15-L  
**Source paper:** P15  
**Title:** *Takens-Based Transformer for Protein Structure Prediction: A Proof-of-Concept Implementation with Open-Source Code*  
**Difficulty:** Intermediate  
**Prerequisites:** P01-L (TBT introduction — essential); M02-L (FSET — strongly recommended); ATT_08-L (Geofinitism — recommended)  
**Estimated study time:** 50 minutes

---

## Learning Objectives

After completing this lesson you will be able to:

1. Explain why protein folding can be reframed as attractor reconstruction rather than sequence-to-structure mapping
2. State Takens' delay embedding theorem and apply it to a discrete biological sequence
3. Describe the MARINA architecture: residue encoding, exponential delay embedding, manifold projection, coordinate prediction
4. Explain why exponential delay spacing captures multi-scale protein organisation
5. Explain why triplication of training data improves a TBT model but not a statistical model
6. Interpret the in-training RMSD results correctly and state the open research questions
7. Connect MARINA to the broader TBT programme (P01, P02, M02, M07)

---

## Key Idea 1 — Two Paradigms for the Same Problem

### The Statistical Paradigm

AlphaFold and its successors treat protein structure prediction as a statistical mapping: given a sequence of amino acids, find the 3D structure most consistent with the patterns seen in the training data. The attention mechanism identifies which residues are likely to be in contact; the structure is assembled to be consistent with these predicted contacts.

This approach works extraordinarily well. But P15 asks a prior question: **what kind of thing is protein folding, physically?**

### The Dynamical Paradigm

A newly synthesised polypeptide chain does not jump directly from sequence to structure. It explores conformational space over time. The chain moves, flexes, and folds, eventually converging to a stable 3D geometry. **This is a temporal dynamical process** — a trajectory in conformational space converging to an attractor.

The statistical model treats the folded structure as a static target. The dynamical model treats folding as a trajectory and the folded structure as the attractor that trajectory reaches.

**The TBT insight:** if folding is an attractor in a dynamical system, and the amino acid sequence is an observable of that system, then Takens' theorem applies — the attractor can be reconstructed from delayed observations of the observable.

### Exercise 1.1

(a) AlphaFold uses a multiple sequence alignment (comparing the protein to evolutionarily related sequences across many species) as its primary input. Why does this help statistically? Does this information enter the dynamical reframing in the same way?

(b) P15 says protein folding is "a temporal dynamical process." What evidence from biology supports this? (Consider chaperone proteins, folding kinetics, misfolding diseases like Alzheimer's.)

(c) A protein takes microseconds to milliseconds to fold, but MARINA processes the sequence position-by-position in a fixed order. In what sense is this a faithful representation of the temporal process?

---

## Key Idea 2 — Takens Embedding for Amino Acid Sequences

### The Theorem

Takens' delay embedding theorem states: under mild conditions, the state space of a deterministic dynamical system can be reconstructed from delayed observations of a single scalar observable. The reconstruction is diffeomorphic to the original — it has the same geometric structure.

For proteins:
- **Observable:** the amino acid sequence, processed position-by-position
- **Hidden state:** the evolving 3D conformation at each sequence position
- **Goal:** reconstruct the attractor geometry (the folded 3D structure)

### The Delay Coordinate Vector

At each sequence position t, the delay-coordinate vector is:

$$z(t) = \bigl(e(t), e(t-1), e(t-2), e(t-4), e(t-8), e(t-16), e(t-32), e(t-64), e(t-128)\bigr)$$

where e(·) is a learned embedding of the amino acid at that position. Delays are chosen **exponentially**: [1, 2, 4, 8, 16, 32, 64, 128].

### Why Exponential Spacing?

Proteins organise simultaneously across multiple structural scales:

| Scale | Structural feature | Delay range |
|---|---|---|
| Local | Backbone geometry, dihedral angles | 1–4 |
| Secondary | α-helices (3.6 residues/turn), β-sheets | 8–32 |
| Tertiary | Long-range contacts, domain topology | 64–128 |

Linear delay spacing would oversample the local scale and undersample the long-range scale. Exponential spacing captures all three levels with a fixed number of delays.

**Crucially:** the delays are not learnable — they are fixed before training. What is learned is the embedding e(·) and the projection matrix W_p, which discover which temporal scales are informationally relevant for the geometry.

### Exercise 2.1

(a) An α-helix repeats approximately every 3.6 residues. What delay would most directly encode a helix repeat? Why is this within the range captured by exponential delays [1, 2, 4, 8, 16, 32, 64, 128]?

(b) Takens' theorem applies to deterministic dynamical systems. Protein folding has stochastic elements (thermal fluctuations, solvent effects). Does this invalidate the Takens approach? What assumption does MARINA implicitly make?

(c) The embedding vector has dimension 9 × 128 = 1,152. This vector is then projected down. Why is dimensionality reduction necessary? What information is being discarded, and what is being preserved?

---

## Key Idea 3 — The MARINA Architecture

### Residue Encoding

The 20 standard amino acids are mapped to learned embedding vectors of dimension 128. No positional encodings are used. This is a deliberate choice: **positional order is encoded through the delay structure, not through an explicit position signal.**

This is the same approach as in P01 (TBT for language): positional information is implicit in which delays are active, not explicit in a separate positional embedding. The model must discover the sequential structure through the geometry of the delay-coordinate manifold.

### Adaptive Manifold Projection

The 1,152-dimensional delay vector is projected to a lower-dimensional manifold:

$$h(t) = \text{LayerNorm}(W_p \cdot z(t) + b_p)$$

The projection matrix $W_p$ is the geometric heart of the model. Its rows can be analysed to reveal which temporal scales the model learned to use. This is a direct window into the model's geometric structure — unlike attention weights, which require careful interpretation, the rows of W_p are directly interpretable as learned delay filters.

### Coordinate Prediction

Three independent linear heads map h(t) to the (x, y, z) coordinates of the C_α carbon atom at position t. The model predicts the 3D coordinates directly in Ångström space. Training minimises mean-squared-error loss.

**No cross-position attention.** The model processes each position using only its local delay neighbourhood, not global pairwise interactions. This gives O(N) complexity and O(1) memory with respect to sequence length.

### Exercise 3.1

(a) AlphaFold uses attention over all pairs of residues — O(N²) complexity. MARINA uses a local delay window — O(N log N) complexity (the log N comes from the exponential delay span). What does this trade-off imply about very long proteins? What structural information might MARINA miss?

(b) The projection matrix W_p is learned from data. After training, how could you inspect it to understand what the model learned? What pattern would you expect to see if the model successfully learned multi-scale protein structure?

(c) MARINA predicts C_α coordinates only (the backbone carbon). Full protein structure includes side chains and backbone oxygens/nitrogens. What would be required to extend MARINA to full-atom prediction?

---

## Key Idea 4 — Triplication and Attractor Geometry

### The Methodological Choice

P15 triples all training proteins in the preprocessing pipeline — each protein is presented to the model three times per epoch, as three copies.

In a standard statistical model, this is useless: the model simply memorises the examples faster, with no benefit to generalisation.

### Why Triplication Helps in TBT

In a Takens-based model, each pass through a protein sequence is a traversal of the conformational trajectory. Repeated traversal:
1. **Deepens the learned attractor basins** — the manifold geometry around stable conformations becomes more sharply defined
2. **Thickens the trajectory filaments** — the dense paths through phase space become thicker and more reliably reconstructable
3. **Improves generalisation to structurally similar proteins** — proteins sharing the same fold family share the same attractor geometry, so a more precisely learned attractor generalises better

This is the same principle as the FSM Triplication Principle from M02: finite symbolic sequences benefit from repeated embedding because the geometry, not the statistics, is what matters.

**The triplication strategy is a diagnostic.** If it improves performance in a model, that model is doing genuine geometric reconstruction. If it does not, the model is doing statistical memorisation.

### Exercise 4.1

(a) Design an experiment to test whether triplication genuinely improves geometric generalisation (testing on structurally similar but sequence-dissimilar proteins) versus merely improving in-distribution accuracy.

(b) The paper says triplication "thickens conformational trajectory filaments in phase space." Draw or describe what this means geometrically. What does a thin trajectory filament look like, and why is a thicker one more useful?

(c) What would happen if you triplicated training data in a standard transformer language model? Would you expect the same benefit? Why or why not?

---

## Key Idea 5 — Results, Limits, and the Open Research Programme

### The In-Training Results

On protein 1A7S (227 residues, included in training):
- Overall RMSD: **1.01 Å** — the predicted backbone falls on average 1 Ångström from the true structure
- Mean per-residue RMSD: **0.62 Å**
- N-terminal region: elevated error (~1.5–2 Å), expected due to greater conformational freedom

**Scale reference:** A hydrogen atom is ~0.5 Å radius. A C–C bond is ~1.5 Å. An RMSD of 1 Å corresponds to sub-bond-length accuracy on average. For context, AlphaFold achieves sub-Ångström RMSD on many benchmark proteins.

### The Honest Caveat

P15 is explicit: these are in-training results. The model was trained on 300–400 proteins — an extremely small dataset by modern standards. Systematic out-of-distribution generalisation testing requires thousands to hundreds of thousands of proteins.

**What the paper claims:**
- The TBT approach can reconstruct coherent protein geometry from residue sequences
- This can be done on consumer hardware with modest compute
- The code is fully open-source and reproducible

**What the paper does not claim:**
- That MARINA outperforms AlphaFold
- That generalisation is established
- That the approach is ready for practical use without scaling

This is the correct scientific posture for a proof-of-concept paper: demonstrate architectural viability, release the tools, invite the community to scale.

### Exercise 5.1

(a) An RMSD of 1.01 Å is reported on an in-training protein. A naive baseline that always predicts the mean atomic position would have a much higher RMSD. What additional baselines should be reported to properly assess the result?

(b) P15 acknowledges its results are in-training. Why does this not make the paper worthless? What specific claims are validly supported by in-training results?

(c) Design a research programme to rigorously evaluate MARINA's generalisation. What datasets would you use? What metrics beyond RMSD? What protein families would be most informative test cases?

---

## Synthesis — P15 in the School of Geofinitism

P15 demonstrates the reach of the TBT programme across radically different domains. The same architectural principle — Takens delay embedding of a finite symbolic sequence to reconstruct attractor geometry — applies to:
- Text sequences (P01, P02): attractor = semantic coherence
- Protein sequences (P15): attractor = 3D fold

This domain-agnostic applicability is a strong argument for the underlying FSM claim: **finite symbolic sequences in any domain can be approached as dynamical systems, and their structure can be reconstructed geometrically rather than statistically.**

The practical significance extends to interpretability. MARINA's projection matrix W_p is directly inspectable; its rows reveal which temporal scales matter for structure. This is a model whose geometry can be read, not merely whose outputs can be evaluated.

**The hardest open question:** can the attractor-reconstruction approach scale to the generalisation challenges that AlphaFold solved through evolutionary information and enormous training data? P15 makes no claim here. But the question is now open experimentally, with fully reproducible code.

---

## Consolidation Questions

1. P15 reframes protein structure prediction as "attractor reconstruction" rather than "sequence-to-structure mapping." Using the formal FSM framework (from M07), what is the relationship between the protein's conformational attractor and the measurement that MARINA is making? Is the predicted 3D structure an admissible claim about the conformational space, or a model-conditioned inference?

2. The triplication strategy improves MARINA but not a standard statistical model. What does this tell us about what kind of model MARINA is? If triplication had no effect, what would that imply?

3. MARINA uses no attention and no positional encodings. AlphaFold uses both extensively. Construct an argument that AlphaFold's positional encodings are a form of model-conditioned inference (in the sense of P14) about the protein's sequential structure, rather than a primitive geometric feature.

4. The paper notes that MARINA's projection matrix W_p "reveals learned temporal scales." How would you inspect W_p to determine whether the model learned short-range (backbone), medium-range (secondary structure), and long-range (tertiary) contributions separately? What would a well-structured W_p look like?

5. P15 acknowledges the results are in-training and that generalisation requires much larger datasets. From an FSM perspective (M02, M07): what is the theoretical condition for Takens embedding to generalise across protein fold families? What structural property would proteins in the same fold family share that would make this generalisation possible?

---

## Further Reading

- **P01** (Introducing the TBT) — the parent architecture; essential prerequisite
- **P02** (Pairwise Phase Space Embedding) — the pairwise delay-coordinate generalisation
- **M02** (FSET Theorem) — theoretical foundation for Takens-style embedding of finite symbolic sequences
- **M03** (Collatz Reconstruction) — another discrete-domain attractor reconstruction; same structural logic
- **M07** (Principia Geometrica) — the Takens-Cauchy-Riemann Theorem; formal conditions for geometric preservation in delay embedding
- **ATT_76** (Semantic Coupling to Observation) — protein binding as dynamic measurement → fixed symbol; the same physical system approached from the measurement direction
- **P12** (Trajectory-Based Computation) — computation as trajectory; P15 is trajectory reconstruction in a biochemical domain
- **P14** (Admissibility and Limits of Measurement) — for careful reading of the in-training RMSD results: what is admissible to claim, and what requires additional model assumptions?

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
