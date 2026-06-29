# Lesson P16-L — Protein-Ligand Affinity as Multiscale Correspondence

**Lesson ID:** P16-L  
**Source paper:** P16  
**Title:** *Protein-Ligand Affinity as Multiscale Correspondence: A Takens-Based Programme for Sequence-to-Structure and Affinity Modelling — Part 1*  
**Difficulty:** Intermediate  
**Prerequisites:** P15-L (MARINA protein structure — strongly recommended); P14-L (Admissibility — recommended); ATT_08-L (Geofinitism — recommended)  
**Estimated study time:** 55 minutes

---

## Learning Objectives

After completing this lesson you will be able to:

1. State the data construct actually underlying modern protein-ligand affinity models, distinguishing what is measured from what is inferred
2. Explain why binding affinity is a multiscale property, not merely a local property of the binding pocket
3. Interpret SMILES notation as a compressed construction signal and identify its known failure modes
4. Apply the multiscale correspondence framework A = C(S(P), M(L), E, Q) + ε to a concrete example
5. Describe the five-level protein scale hierarchy and explain why each level contributes to the affinity label
6. Explain how Takens delay embeddings provide multiscale coverage of a protein sequence
7. Outline the six-step research programme and the scientific reason for each step

---

## Key Idea 1 — Unpacking the Data Construct

### What Is Actually in the Dataset?

Modern protein-ligand affinity datasets typically contain:
- A protein identifier or amino-acid sequence
- A ligand SMILES string
- A measured activity value (IC₅₀, K_d, K_i, or similar)
- Assay metadata (often incomplete)

What they do **not** typically contain:
- Full three-dimensional atomic coordinates for the bound complex
- Dynamic atom-position histories (the protein and ligand moving over time)
- Environmental degrees of freedom (water, ions, temperature, pH)

This compression is unavoidable — collecting full dynamic structures for millions of protein-ligand pairs is not currently feasible. But the compression must not be forgotten. P16's opening move is to make this compression explicit:

$$\text{complete atom trajectories} \;\to\; \text{protein code + ligand code + activity label}$$

### What Is a Model Doing?

A neural affinity model learns:

$$(P, L) \;\longrightarrow\; Z(P,L) \;\longrightarrow\; \hat{a}$$

It maps compressed symbolic inputs to a latent representation, then predicts the affinity label. The latent representation Z is learned, not derived from first principles. The output â is useful when Z captures physically relevant structure — and misleading when Z captures dataset artefacts.

**The scientific risk:** a model can produce confident numerical outputs that correlate with activity in retrospective datasets while failing on new chemical scaffolds. This failure is not detectable from training metrics alone.

### Exercise 1.1

(a) A drug company trains an affinity model on ChEMBL data. The model achieves R² = 0.82 on the test set. A medicinal chemist then uses the model to rank 200 macrocyclic compounds. The model performs no better than random. Using P16's framework, explain what likely happened.

(b) The metadata term q in the dataset D = {(P_n, L_n, a_n, q_n)} is described as "often incomplete." What specific assay variables would be contained in q? Why does ignoring them create systematic bias in the learned model?

(c) P16 says "data curation is not a secondary detail — it is part of the model." What does this mean formally? In FSM terms (P13, P14), what kind of drag does assay heterogeneity introduce?

---

## Key Idea 2 — Affinity Is Multiscale

### The Binding Pocket Is Not Self-Contained

Consider a drug binding in the active site of a protein. It is tempting to think that only the handful of residues lining the pocket matter. But those residues are where they are because the whole protein has folded into a particular geometry.

**Examples where long-range structure determines local affinity:**
- A mutation 50 residues away from the binding pocket changes the dynamics of a helix that borders the pocket, altering pocket shape and affinity by two orders of magnitude (allosteric effect)
- A quaternary assembly (two protein chains forming a dimer) creates a binding interface that does not exist in either chain alone
- A disordered region far from the binding site undergoes ordering upon ligand binding, contributing an entropic cost to the affinity

In each case, the affinity label a is attached to the full multiscale correspondence — not merely the local contact.

### The Formal Statement

$$A = C\bigl(S(P),\; M(L),\; E,\; Q\bigr) + \epsilon$$

where:
- S(P) = {S₁(P), ..., S_K(P)}: the multiscale structural realisation of the protein (residues → motifs → secondary structures → domains → tertiary fold → quaternary assembly)
- M(L) = {M₁(L), ..., M_J(L)}: the multiscale molecular realisation of the ligand (atoms → functional groups → ring systems → conformers → charge distributions → molecular shape)
- E: environmental conditions
- Q: measurement protocol
- ε: measurement noise and unmodelled factors

### The Language Scale Metaphor

| Language scale | Protein scale | Structural meaning |
|---|---|---|
| Letter / word | Residue | Local symbolic unit |
| Short phrase | Motif | Local pattern |
| Sentence | Secondary structure | Local folded grammar |
| Paragraph | Domain | Coherent substructure |
| Whole document | Tertiary fold | Full single-chain meaning |
| Library / dialogue | Quaternary assembly | Multi-chain context |

Scoring a binding affinity from only the binding-pocket residues is like judging the meaning of a document from a single paragraph. The paragraph's meaning depends on the whole document.

### Exercise 2.1

(a) Allosteric drugs bind at sites *away* from the active site and yet modulate activity. Using the multiscale correspondence framework, explain why an allosteric drug's affinity cannot be predicted from local binding-pocket features alone. What scale does allostery operate at?

(b) The paper states: "An affinity label a attached to (P, L) is formally a label over the whole pair, even if the immediate physical contact is local." Does this mean a model must always use the full protein sequence? Or can local crops sometimes be sufficient? Under what conditions?

(c) "The binding pocket exists because the whole protein folds into a particular geometry." Use this principle to explain why proteins in the same fold family (similar tertiary structure, different sequence) often bind similar classes of ligands.

---

## Key Idea 3 — SMILES as a Compressed Construction Signal

### What SMILES Is and Is Not

SMILES (Simplified Molecular Input Line Entry System) encodes: atoms, bonds, branches, ring closures, charges, sometimes stereochemistry. It is compact, machine-readable, and widely used in bioactivity databases.

**What SMILES is:** a traversal of a molecular graph. A symbolic code from which the connectivity of a molecule can be reconstructed.

**What SMILES is not:** the molecule itself, a unique representation (multiple valid SMILES strings per molecule unless canonicalised), a specification of 3D geometry, a description of the dynamic conformational ensemble.

**Formal status (in FSM/P14 terms):** a SMILES string is an admissible compression of molecular connectivity — but it is not an admissible claim about 3D structure. A model predicting affinity from SMILES is operating from a symbolically compressed signal, not from a measured atomic configuration. The admissibility boundary (P14) sits between "the molecule has this connectivity" (admissible from SMILES) and "the molecule has this 3D shape" (not admissible from SMILES without further reconstruction).

### Known SMILES Failure Modes

1. **Notation artefacts:** a model may learn shortcuts specific to a SMILES traversal order rather than the molecular structure itself. Solution: use randomised SMILES (multiple random traversals of the same molecule) or molecular graphs.

2. **Conformational silence:** a SMILES string does not encode which conformation the ligand adopts when bound. For rigid, planar molecules, this may not matter. For flexible molecules (especially macrocycles), the bound conformation may be very different from the lowest-energy solution conformation.

3. **Scaffold bias:** if the training set contains mainly one chemical scaffold, a model may learn the scaffold rather than the underlying physical interaction. The failure mode does not appear until a new scaffold is tested.

### Exercise 3.1

(a) The SMILES string for benzene is `c1ccccc1`. Write an alternative valid SMILES for benzene. Why do these represent the same molecule, and what does this imply about a model trained on raw (non-canonical) SMILES?

(b) Macrocyclic drugs (large ring systems with many rotatable bonds) are particularly difficult for SMILES-based models. Using P16's failure-mode analysis, explain *why* specifically. What is the admissibility claim a SMILES-based model makes about macrocycle binding?

(c) P16 mentions conformer ensembles as one ligand representation option. A conformer ensemble is a set of possible 3D shapes computed by a molecular mechanics method. Is this an admissible measurement (in P14 terms) or a model-conditioned inference? What does this imply for how it should be labelled?

---

## Key Idea 4 — Takens Delay Embeddings for Multiscale Coverage

### Why Delay Embeddings?

A residue at position i influences the 3D structure not just through its immediate neighbours (positions i-1, i+1) but through distant residues that become spatially adjacent after folding. A model processing only local windows misses these long-range constraints.

Delay embeddings at position i:

$$\Phi^P_{\tau,d}(i) = \bigl(e_i,\; e_{i+\tau},\; e_{i+2\tau},\; \ldots,\; e_{i+(d-1)\tau}\bigr)$$

A family of delays T = {τ₁, τ₂, ..., τ_R} gives the model explicit access to multiple scales simultaneously. No single delay can capture all levels; the family captures all of them.

**The connection to protein scales:**

| Delay range | Structural scale captured |
|---|---|
| τ = 1–4 | Local residue interactions, backbone angles |
| τ = 8–32 | Secondary structure (α-helix period ≈ 3.6 residues, β-strand spacing) |
| τ = 32–128 | Domain-level and long-range contacts |
| τ > 128 | Very long-range contacts, inter-domain, quaternary context |

### The Epistemic Difference

A standard Transformer processes (i, j) residue pairs with learned pairwise attention weights. The model learns *which* residue pairs to attend to — but it learns this from data, without being told that different spatial scales require different distance regimes.

A Takens-based Transformer is given explicit access to specific delay regimes. The model does not need to discover from scratch that secondary structure operates at the 3.6-residue scale; this is encoded in the delay choice. This is a form of **prior knowledge injection** that is interpretable: you can examine which delays the model uses after training (Step 2 of the research programme: delay-scale ablation).

### Exercise 4.1

(a) An α-helix repeats approximately every 3.6 residues. In the delay family [1, 2, 4, 8, 16, 32, 64, 128], which delay most directly captures a single helix repeat? Which combination of delays would be needed to capture a complete helix of 10 residues?

(b) A standard self-attention layer has O(N²) complexity (all pairwise interactions). A Takens delay embedding with a fixed delay family has O(N log N) complexity (each position attends to log N delayed neighbours). For a protein of 300 residues, compute the difference in number of interactions. What is the memory and computational cost of adding a longer delay?

(c) The delay-scale ablation (Step 2 of the research programme) removes individual delay families and measures the effect on prediction accuracy. What result would you expect if short delays dominate (model is mostly using local context)? What result would you expect if long delays dominate (model is mostly using global context)? What result would suggest the model is genuinely using multiscale structure?

---

## Key Idea 5 — The Research Programme

### Why a Six-Step Programme?

P16 is explicitly Part 1 — a theoretical and programmatic document, not a results paper. The six steps are ordered from most basic (does the architecture work at all?) to most demanding (does it make useful predictions before the experiment?):

**Step 1 — Sequence-to-structure validation:** Establish that the Takens architecture can reconstruct protein geometry with held-out test proteins. Without this, nothing downstream can be trusted.

**Step 2 — Delay-scale ablation:** Verify that the delay structure is doing interpretable work — not merely adding parameters. If removing all delays beyond τ = 4 does not hurt performance, the model is not using long-range structure.

**Step 3 — Ligand-code representation tests:** Find the best ligand encoding. This is a non-trivial choice: SMILES, canonical SMILES, randomised SMILES, molecular graph, conformer ensemble. The right answer may be different for different chemical series.

**Step 4 — Structure-plus-affinity training:** Combine structure supervision with affinity supervision. The hypothesis is that structure-aware representations generalise better across chemical series than purely affinity-trained models.

**Step 5 — Multiscale correspondence diagnostics:** After training, examine *what* the model learned. Does it use long-range sequence features, local pocket features, ligand substructures, assay conditions? A model that only uses scaffold similarity is not doing physics — it is doing memorisation dressed as chemistry.

**Step 6 — Prospective validation:** Make predictions *before* experiments. Compare with results. This is the only test that genuinely establishes causal rather than retrospective correspondence.

### Exercise 5.1

(a) Step 6 (prospective validation) is described as "the strongest test." Why is retrospective performance on a held-out test set not sufficient to establish that the model is learning physical correspondence rather than dataset correlation?

(b) A pharmaceutical company runs Step 4 and finds that adding structure supervision (training simultaneously on structure RMSD and affinity) improves affinity prediction on unseen scaffolds but slightly reduces performance on the in-distribution test set. How should this result be interpreted? What does it suggest about what the structure supervision is providing?

(c) Step 5 asks: "Does performance degrade outside the training basin?" This is a reference to the FSM concept of an attractor basin. Translate this question into FSM language (M07, M02). What is the "training basin" geometrically? What does it mean for performance to degrade "outside" it?

---

## Synthesis — P16's Place in the School

P16 is the most programmatic paper in the proteins cluster (P15–P17). Where P15 demonstrates architectural viability, P16 provides the theoretical framework that makes the approach principled rather than ad hoc.

The paper's most important contribution is conceptual clarity: distinguishing what is measured, what is compressed, what is inferred, and what is narrated as though it were measured but is not. This is the admissibility framework (P14) applied to an entire research field.

The six-step research programme gives the programme a clear path from proof-of-concept to prospective validation. Each step is falsifiable. The delay-scale ablation in particular is a crucial diagnostic: if long-range delays do not contribute to affinity prediction, the Takens-based approach offers no advantage over a standard local-attention model. The programme cannot succeed on its central claim without this step showing genuine multiscale contribution.

**The deepest claim:** affinity is not a local property that happens to be influenced by global structure. It is constitutively multiscale. A model that predicts affinity by looking only at the local binding pocket is making a category error — like predicting the meaning of a document from a single sentence.

---

## Consolidation Questions

1. P16 draws a distinction between "structure-free affinity prediction" (as used by AQAffinity/OpenFold3) and "prediction without any internal structural representation." What is the distinction? Why does it matter for how we interpret the model's outputs?

2. The formal affinity equation is A = C(S(P), M(L), E, Q) + ε. The term Q (assay protocol) is often not provided to the model during training. Using P14's admissibility framework, what is the formal problem with reporting an affinity prediction that ignores Q? What kind of claim is it making?

3. A model trained on ChEMBL (diverse chemical series, many assay types) is compared to a model trained on proprietary data for a single protein target (one assay type, dense chemical series coverage). Under what conditions would each model generalise better? Use P16's failure-mode analysis.

4. The six-step research programme proceeds from Step 1 (structure) to Step 6 (prospective validation). Why is it essential to complete Step 1 before Step 4? Could a model skip structure supervision and go directly to affinity training? What would be lost?

5. P16 says "the scientific risk is not that such models exist; the risk is narrating them as though the compression necessarily carries full physical understanding." Write the admissible and inadmissible version of the following claim: "Our model accurately predicts the binding affinity of this drug candidate for EGFR."

---

## Further Reading

- **P15** (MARINA) — the proof-of-concept protein structure implementation that P16 extends to affinity
- **P14** (Admissibility and Measurement) — the formal framework for P16's data construct table and failure modes
- **P13** (FSI Drag) — the formal cost of symbolic instantiation; SMILES compression and affinity label compression are FSI drag in biochemistry
- **ATT_76** (Semantic Coupling to Observation) — protein binding as the conversion of dynamic measurement into fixed symbols; P16's theoretical companion in essay form
- **M02** (FSET) — theoretical foundation for delay embedding of finite symbolic sequences
- **M07** (Principia Geometrica) — Takens-Cauchy-Riemann Theorem; formal conditions for geometric preservation in delay embedding
- **P12** (Trajectory-Based Computation) — computation as trajectory; the general FSM framework that P16 applies to biochemistry

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
