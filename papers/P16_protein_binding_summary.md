# P16 — Protein-Ligand Affinity as Multiscale Correspondence

**Full title:** *Protein-Ligand Affinity as Multiscale Correspondence: A Takens-Based Programme for Sequence-to-Structure and Affinity Modelling — Part 1*  
**Paper ID:** P16  
**Author:** Kevin R. Haylett  
**Location:** Manchester, UK  
**Series:** Selected Communications  
**Date:** May 24, 2026  
**Pages:** 18  
**Primary College:** College of Attralucian Studies  
**Secondary Colleges:** College of Machine Intelligence; College of Finite Symbolic Mechanics; College of Finite Measurements  
**Primary Pillars:** P3 (Dynamic Flow), P2 (Approximations/Measurements)  
**Secondary Pillars:** P1 (Geometric Container), P5 (Finite Reality)  
**Status:** Stable (programmatic — Part 1 of a series)  
**Source:** `P16_protein_binding.pdf`

---

## Overview

P16 is a programmatic paper — less a results report and more a systematic unpacking of the data construct underlying protein-ligand affinity modelling, followed by a six-step research programme for a Takens-based approach. The central contribution is theoretical: the paper argues that binding affinity is a **multiscale correspondence** between two compressed construction signals (protein sequence, ligand code), and that Takens-style delay embeddings provide a principled method for reconstructing the hidden geometric structure those signals carry.

Written for readers from molecular biology, chemistry, and computational machine learning. Acronyms are minimised; the goal is to recover what is actually being measured and computed beneath the technical language of current affinity models.

---

## The Core Argument in Plain Language

Modern protein-ligand affinity models take a protein sequence and a ligand SMILES string as input and output a binding score. The paper asks: **what is this actually doing?**

The practical data construct beneath the language:

> *Complete measured atomic trajectories are replaced by: protein code + ligand code + activity label.*

This compression is unavoidable given what data exist. But it must not be narrated as though it is equivalent to first-principles physical measurement. P16 provides the vocabulary for saying precisely what each layer is.

The key theoretical refinement: **affinity is not merely a local property** of a ligand touching a binding pocket. The binding pocket is a consequence of the whole multiscale protein construction. A residue far from the pocket in sequence can stabilise a domain orientation that creates or destroys the pocket. Allosteric mutations outside the pocket can alter binding. Quaternary assembly can create binding sites between chains.

Therefore: an affinity label attached to a protein-ligand pair is formally a label over the entire multiscale relationship, even if immediate physical contact is local.

---

## Molecules as Construction Signals

**The protein as construction signal:**  
The amino acid sequence P = (p₁, p₂, ..., p_N) is a one-dimensional symbolic code produced from genetic information (DNA → RNA → protein). It is a biological construction signal: from the sequence, a three-dimensional molecular object is built. The sequence is not arbitrary — evolution, chemistry, and physical viability have already shaped the distribution.

Protein structure organises hierarchically:
- Primary: residue sequence
- Secondary: α-helices, β-sheets (local folded grammar)
- Tertiary: full 3D fold of one chain
- Quaternary: multi-chain assembly and higher-order context

**The ligand as construction signal:**  
A ligand code such as a SMILES string is a compact symbolic representation: atoms, bonds, branches, rings, charges, stereochemistry. It is not the molecule — it is a traversal of a molecular graph. Two SMILES strings can represent the same molecule. A model trained on raw SMILES may learn notation artefacts rather than molecular geometry.

**The multiscale correspondence:**

$$S(P) = \{S_1(P), S_2(P), \ldots, S_K(P)\}$$

where S₁ = local residue patterns, S₂ = secondary structures, S₃ = domains, S₄ = tertiary fold, S₅ = quaternary assembly.

$$M(L) = \{M_1(L), M_2(L), \ldots, M_J(L)\}$$

where levels include atoms, functional groups, ring systems, conformers, charge distributions, molecular shape.

**Measured affinity:**

$$A = C(S(P),\, M(L),\, E,\, Q) + \epsilon$$

where C is an unknown correspondence/locking function, E denotes environmental conditions, Q denotes the measurement protocol, and ε represents measurement noise and unmodelled factors.

The affinity label a is therefore not: *binding pocket + ligand → score.* It is a measurement over the full nested structure encoded by the protein sequence and the molecular object encoded by the ligand.

---

## The Sentence Metaphor

The paper introduces a scale-structure analogy to language (not to be overextended — proteins are physical molecules, not texts):

| Language scale | Protein scale | Structural meaning |
|---|---|---|
| Letter or word | Residue | Local symbolic unit |
| Short phrase | Motif | Local pattern |
| Sentence | Secondary structure | Local folded grammar |
| Paragraph | Domain | Coherent substructure |
| Whole document | Tertiary fold | Full single-chain meaning |
| Library or dialogue | Quaternary assembly | Multi-chain context |

A binding-affinity measurement is therefore not like scoring the match between one word and one other word. It is like scoring how one structured text relates to another structured object across letters, phrases, sentences, paragraphs, and the whole document.

---

## What Is Actually Measured: Affinity Values

Experimental affinity can be reported as dissociation constants (K_d), inhibition constants (K_i), half-maximal inhibitory concentrations (IC₅₀), percentage inhibition, or other assay-derived readouts. These are not identical measurements. A common transformation:

$$\text{pIC}_{50} = -\log_{10}(\text{IC}_{50})$$

A binding-affinity dataset:

$$\mathcal{D} = \{(P_n, L_n, a_n, q_n)\}_{n=1}^{M}$$

where q_n is assay metadata — the same protein-ligand pair can appear under different assay types, laboratories, buffer conditions, temperature, and constructs. **Data curation is not a secondary detail — it is part of the model.**

---

## Current Modelling Landscape

**Physics-based approaches:** docking (place ligand, score pose), molecular dynamics (simulate atomic motion under force field), free-energy perturbation (estimate binding free energy changes alchemically). Physically motivated, powerful, computationally expensive.

**Machine-learning approaches:** learn patterns from sequence, ligand strings/graphs, predicted structures, and activity labels. Modern systems (AlphaFold 3, AQAffinity on OpenFold3) predict structures of biomolecular complexes. "Structure-free" in this context means: no experimentally measured input structure required at inference time — not that the model has no internal structural representation.

**What an affinity head is:** the final task-specific neural module attached to a trunk/backbone. The trunk converts (P, L) → internal representation Z. The head reads Z and predicts:
- Continuous affinity: â = h_θ(Z(P,L))
- Binder classification: b̂ = σ(h_θ(Z(P,L)))

The model learns: (P, L) → Z(P,L) → â. The scientific risk: the output can be useful without carrying full physical understanding; the risk is narrating it as though the compression necessarily carries that understanding. P16 provides the language to avoid this.

---

## Takens Embedding as the Modelling Principle

For a protein sequence, the delay-coordinate representation at position i:

$$\Phi^P_{\tau,d}(i) = (e_i,\, e_{i+\tau},\, e_{i+2\tau},\, \ldots,\, e_{i+(d-1)\tau})$$

where e(·) = E(p_i) is a learned residue embedding.

A multi-delay family T = {τ₁, τ₂, ..., τ_R} produces a multiscale representation:

$$\mathcal{E}(P) = \{\Phi^P_{\tau_r, d_r}(i) : r = 1,\ldots,R;\; i = 1,\ldots,N\}$$

- **Short delays:** capture local motifs, backbone geometry
- **Intermediate delays:** capture relations across helices, sheets, loops, domains
- **Long delays:** expose distant sequence positions that become spatially close after folding (the core biological difficulty: residues far apart in sequence can be close in 3D space)

A Takens-based Transformer uses attention guided by multiple delayed views of the sequence, rather than raw token positions. The model learns which delays matter for reconstructing hidden geometry.

**The distinction is epistemic, not merely architectural.** The model explicitly treats input codes as observed signals from which hidden geometry and correspondence must be reconstructed — rather than as a direct representation of physical state.

---

## Illustrated Example: 1E2F

A selected in-training result from a prototype Takens-based Transformer run on a home computer: Protein Data Bank entry 1E2F, human thymidylate kinase complexed with thymidine monophosphate, adenosine diphosphate, and a magnesium ion. Reported aligned RMSD: **1.39 Å**.

Careful interpretation: this is a selected good example from the training set. It does not establish generalisation. Its scientific value is **methodological** — demonstrating that Takens-based delay-coordinate modelling can be implemented and can reproduce known structure under favourable conditions.

---

## The Extended Affinity Architecture

For protein-ligand affinity prediction:

$$\hat{a} = H_\theta\bigl(\mathcal{E}^P(P),\; \mathcal{E}^L(L),\; C^{PL}(P,L),\; q\bigr)$$

where:
- E^P(P): multiscale delayed representation of the protein
- E^L(L): ligand representation (SMILES token sequence, graph, conformer ensemble, or hybrid)
- C^PL(P,L): cross-correspondence representation between protein and ligand tokens
- q: assay metadata when available
- H_θ: the predictive model

**Ligand representation options:** raw SMILES tokens, canonical SMILES, randomised SMILES (to reduce notation artefacts), molecular graph, conformer ensembles, hybrid. Each has different failure modes; ligand representation is a major unresolved choice.

**Combined training objective:**

$$\mathcal{L} = \lambda_S \mathcal{L}_{\text{structure}} + \lambda_A \mathcal{L}_{\text{affinity}} + \lambda_B \mathcal{L}_{\text{binding}}$$

---

## Why the Affinity Label Acts Across All Scales

Even when a ligand binds locally, the local pocket is not an independent object — it exists because the whole protein folds into a particular geometry. Therefore the affinity label a attached to (P, L) is formally a label over the whole pair:

$$a \sim C\bigl(\{S_k(P)\}_{k=1}^K,\; \{M_j(L)\}_{j=1}^J,\; E,\; Q\bigr)$$

This is the core reason a multiscale delay method is attractive: it gives the model explicit routes to learn short, medium, and long separation dependencies.

---

## Knowledge Limits and Failure Modes

Five known failure modes:
1. **SMILES compression insufficiency** — different valid strings for the same molecule; string-pattern shortcuts may fail on new chemical scaffolds
2. **Dataset noise and heterogeneity** — assay type, conditions, laboratory, construct, temperature, pH, detection technology; curation is part of the model
3. **Local fine-tuning does not imply broad physical understanding** — a model can succeed in one chemical series (e.g., JAK2 macrocycles) and fail on others
4. **Structure and affinity require different validation** — structural RMSD and affinity ranking are different skills; a model can produce plausible-looking structures while failing to rank affinity correctly
5. **Dataset correlation vs. physical causation** — internal representations difficult to inspect; the risk is amplified relative to classical methods

---

## The Six-Step Research Programme

**Step 1 — Sequence-to-structure validation:** Train and test Takens-based Transformers with strict training/test separation. Evaluate RMSD, local distance errors, residue-level errors, secondary-structure consistency, and performance on unseen folds.

**Step 2 — Delay-scale ablation:** Remove or vary different delay families to identify which sequence separations matter for local motifs, secondary structures, domains, and long-range contacts. Essential to demonstrate that the delay construction adds interpretable value beyond ordinary attention.

**Step 3 — Ligand-code representation tests:** Compare raw SMILES, canonical SMILES, randomised SMILES, molecular graphs, and conformer ensembles. Determine where each fails.

**Step 4 — Structure-plus-affinity training:** Combine structure reconstruction with affinity prediction. Test both with known protein-ligand complexes and in structure-free settings.

**Step 5 — Multiscale correspondence diagnostics:** Examine whether affinity predictions depend on local pocket features, long-range sequence features, ligand substructures, assay metadata, or scaffold similarity. Diagnose the learned correspondence rather than simply report a score.

**Step 6 — Prospective validation:** Use the model to propose predictions before experiments, then compare with new measured results. The strongest test of whether the learned correspondence is useful outside retrospective datasets.

---

## Data Construct Table

| Layer | Data object | Role in modelling |
|---|---|---|
| Genetic coding | DNA / RNA sequence | Upstream biological code used by the cell to produce the amino-acid sequence |
| Protein input | Amino-acid sequence | Main symbolic construction signal for the protein chain |
| Protein structure target | Atom or residue coordinates | Experimental or curated target geometry for structure training |
| Ligand input | SMILES or graph | Symbolic construction signal for the drug-like molecule |
| Ligand geometry | Conformers or bound pose | Optional computed or measured geometry — often unavailable |
| Affinity target | Binding/activity value | Experimental scalar label for regression or classification |
| Metadata | Assay conditions | Critical context for interpreting activity values — often incomplete |
| Model representation | Latent vectors/tensors | Learned compression where protein-ligand correspondence is represented |
| Prediction | Structure and/or affinity | Model output requiring validation |

---

## Connections to Other Work

- **P15** (MARINA) — the proof-of-concept Takens protein structure implementation; P16 extends this to the affinity problem and provides the full programmatic framework that P15 demonstrates in miniature
- **P01** (TBT) — the parent Takens-Based Transformer architecture; P16 proposes its extension to protein-ligand cross-correspondence
- **P12** (Trajectory-Based Computation) — the general FSM framework for computation as trajectory; P16 is its application in biochemistry
- **P13** (FSI Drag) — the formal framework for the cost of symbolic instantiation; P16's SMILES compression insufficiency and RMSD limitations are FSI drag in a biochemical setting
- **P14** (Admissibility and Measurement) — P16's data construct table and failure modes analysis is the admissibility framework applied to molecular biology: SMILES is not the molecule; the affinity label is model-conditioned, not a primitive measurement of binding
- **ATT_76** (Semantic Coupling to Observation) — protein binding is explicitly discussed in ATT_76 as a case study of dynamic measurement converted to fixed symbols; P16 develops the prediction direction
- **ATT_38** (The Generonic Boundary) — the observation pipeline in ATT_38 (interaction → capture → projection → corpus integration → correspondence modelling) maps directly onto P16's data construct table
- **M02** (FSET) — the theoretical foundation for Takens-style delay embedding of finite symbolic sequences; protein sequences are finite symbolic sequences in an alphabet of 20 residues
- **M07** (Principia Geometrica) — the Takens-Cauchy-Riemann Theorem; the formal conditions for geometric preservation in delay embedding that P16 relies on

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
