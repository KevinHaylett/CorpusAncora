# P15 — Takens-Based Transformer for Protein Structure Prediction

**Full title:** *Takens-Based Transformer for Protein Structure Prediction: A Proof-of-Concept Implementation with Open-Source Code*  
**Paper ID:** P15  
**Author:** Kevin R. Haylett  
**Location:** Manchester, UK  
**Series:** Selected Communications  
**Date:** May 24, 2026  
**Pages:** 6  
**Licence:** Mozilla Public License 2.0 (code); open access (paper)  
**Primary College:** College of Attralucian Studies  
**Secondary Colleges:** College of Machine Intelligence; College of Finite Symbolic Mechanics  
**Primary Pillars:** P3 (Dynamic Flow), P1 (Geometric Container)  
**Secondary Pillars:** P2 (Approximations/Measurements), P5 (Finite Reality)  
**Status:** Stable  
**Source:** `P15_takens_protein_prediction.pdf`  
**Code:** https://github.com/KevinHaylett/takens-protein-folding

---

## Overview

P15 applies the Takens-Based Transformer (TBT) architecture to protein structure prediction, introducing **MARINA** (Manifold-Aware Reconstruction and Inference Network Architecture). The central claim is that protein folding is a nonlinear dynamical process — not a sequence-to-structure statistical mapping — and that the 3D fold can be reconstructed from exponential delay coordinates of the amino acid sequence using Takens' embedding theorem.

The paper is a proof-of-concept with full open-source code release. The model runs on consumer CPU hardware (~15M parameters), requires no attention mechanism and no positional encodings, and achieves 1.01Å overall RMSD on an in-training example (protein 1A7S, 227 residues).

The work belongs to the domain-agnostic Takens-Based Transformer programme (P01, P02) applied to a new domain: biochemistry.

---

## The Dynamical-Systems Reframing

Most current protein structure prediction (e.g., AlphaFold) treats the problem as a statistical mapping: **sequence → structure**. P15 argues this engineering framing, while successful, obscures the underlying physics.

**The Geofinite reframing:**
- The amino acid sequence is the **observable time series** of a nonlinear dynamical system
- The hidden state of that system is the **evolving conformation** in 3D space
- The system converges to a stable attractor — the **folded geometry**
- The task is therefore **attractor reconstruction from delayed observations** — precisely the setting of Takens' theorem

This is the same structural move made throughout the TBT programme: replace statistical pattern-matching with geometric trajectory reconstruction.

---

## Takens' Theorem in the Protein Context

Takens' delay embedding theorem (1981) states that, under mild conditions, the state space of a deterministic dynamical system can be reconstructed from delayed observations of a single scalar (or vector) time series. Applied to proteins:

- **Observable:** the amino acid sequence processed position-by-position; each residue yields a learned embedding vector
- **Hidden state:** the evolving 3D conformation
- **Delay coordinates:**

$$z(t) = \bigl(e(t),\, e(t-\tau_1),\, e(t-\tau_2),\, \ldots,\, e(t-\tau_m)\bigr)$$

where e(·) is a learned residue embedding and delays are chosen exponentially:

$$\text{delays} = [1, 2, 4, 8, 16, 32, 64, 128]$$

**Why exponential spacing?** Proteins organise across multiple scales simultaneously:
- Short delays (1–4): local backbone geometry, immediate residue interactions
- Medium delays (8–32): secondary structure (helices, sheets)
- Long delays (64–128): tertiary topology, long-range contact patterns

Exponential spacing captures this multi-scale organisation with a fixed embedding dimension.

---

## The MARINA Architecture

MARINA consists of four core components:

### 1. Residue Encoding
Each of the 20 standard amino acids (plus a small extension vocabulary for non-standard residues) is mapped to a learned embedding vector of dimension `embed_dim = 128`. No positional encodings are used — temporal order is encoded implicitly through the delay structure.

### 2. Exponential Takens Embedding
At each position t, a delay-coordinate vector is constructed using a **circular buffer** of size 2k+1 (k=7 for the longest delay of 128). This yields:
- **Memory:** O(1) — fixed circular buffer, independent of sequence length
- **Embedding dimension:** (8+1) × 128 = **1,152**

### 3. Adaptive Manifold Projection
The high-dimensional, sparsely populated delay vector is projected onto a lower-dimensional manifold:

$$h(t) = \text{LayerNorm}(W_p \cdot z(t) + b_p)$$

where $W_p \in \mathbb{R}^{d_{\text{out}} \times 1152}$ is a learned projection matrix. This matrix is the **geometric core** of the architecture — its rows reveal learned temporal scales; phase-space analysis of manifold states can probe attractor stability and mutation effects.

### 4. Coordinate Prediction
Three independent linear heads predict the x, y, z coordinates of the C_α atom. Training uses mean-squared-error loss in Ångström space. No cross-position attention is used.

**Full computational profile:**

| Property | MARINA (TBT) |
|---|---|
| Complexity per position | O(log N) |
| Memory footprint | O(1) (fixed circular buffer) |
| Attention | None |
| Positional encodings | None |
| Hardware requirement | CPU sufficient |
| Parameters | ~15M |

---

## Triplication Training Strategy

A deliberate methodological choice: training proteins are **triplicated** (repeated three times) in the preprocessing pipeline.

**Why this works in TBT but not in statistical models:**
- In a statistical pattern-matching model, duplication provides no new information — the model simply memorises the example
- In a Takens-based architecture, repeated exposure to the same protein **deepens the learned attractor basins** and **thickens the conformational trajectory filaments** in phase space
- This strengthens the geometric structure of the manifold and improves prediction accuracy on structurally similar proteins

The triplication strategy illustrates the conceptual difference between the two paradigms: it is only meaningful if the model is genuinely doing geometric reconstruction, not statistical memorisation.

---

## Proof-of-Concept Results

The model was trained on approximately 300–400 proteins (triplicated), including 1A7S (227 residues, a well-studied globular protein).

**Results on 1A7S (in-training example):**

| Metric | Value |
|---|---|
| Overall RMSD | 1.01 Å |
| Mean per-residue RMSD | 0.62 Å |
| N-terminal region | Elevated (~1.5–2 Å, higher conformational freedom) |
| Remainder of chain | ~0.5 Å |

**Important caveat:** These are in-training results. The paper is explicit that systematic out-of-distribution generalisation testing requires substantially larger datasets and compute resources. Some preliminary evidence of generalisation on structurally similar proteins has been observed, but this is not yet rigorously established.

**The paper's contribution is architectural viability, not benchmark-beating performance.** The goal is to demonstrate that the Takens-based approach can reconstruct coherent protein geometry on modest hardware, and to release fully reproducible code as a foundation for community scaling experiments.

---

## Code Release and Reproducibility

Full open-source release under Mozilla Public License 2.0:  
**Repository:** https://github.com/KevinHaylett/takens-protein-folding

Key files:
- `core/takens_embedding.py` — core delay embedding module
- `protein/protein_tbt.py` — MARINA model definition
- `train.py`, `inference.py` — training and prediction scripts
- `pipeline/pdb_to_training.py` — PDB preprocessing and duplication (triplication)
- `results/` — 1A7S example outputs

All training and inference commands are documented in the README and can be run on consumer hardware (Intel i7 CPU, 32 GB RAM demonstrated).

---

## Interpretability

The architecture offers geometric interpretability not available in attention-based models:
- **Rows of W_p** reveal learned temporal scales — which delay combinations the model uses to construct its manifold
- **Phase-space analysis** of manifold states can probe attractor stability
- **Mutation effects** can be studied by examining how a single residue change perturbs the manifold trajectory
- **No attention weights to interpret** — the model's geometric structure is directly accessible

---

## Position Within the TBT Programme

P15 is one of three applications of the domain-agnostic TBT architecture:

1. **P01** (TBT for language) — the original Takens-Based Transformer applied to text sequences
2. **P02** (Pairwise Phase Space Embedding) — the pairwise generalisation of the embedding approach
3. **P15** (MARINA for proteins) — TBT specialised for amino acid sequences → 3D structure

The common thread: replace attention-based statistical pattern matching with attractor reconstruction from delay coordinates. Each domain provides a new test of whether the Takens-geometric perspective captures structure that purely statistical approaches treat as opaque.

---

## Connections to Other Work

- **P01** (Introducing the TBT) — the parent architecture; P15 is P01 applied to proteins
- **P02** (Pairwise Phase Space Embedding) — the pairwise delay-coordinate framework; MARINA uses the simpler exponential delay form
- **M02** (FSET Theorem) — the theoretical foundation for applying Takens-style embedding to finite symbolic sequences; proteins are a physical instantiation of a finite symbolic sequence (alphabet of 20 amino acids)
- **M03** (Collatz Reconstruction) — another application of delay-coordinate reconstruction to a discrete process; same structural logic as P15
- **M07** (Principia Geometrica) — the Takens-Cauchy-Riemann Theorem formalises the conditions under which delay embedding preserves geometry; P15 relies on this result implicitly
- **ATT_76** (Semantic Coupling to Observation) — analyses protein binding as a case study of dynamic measurement converted into fixed symbols; P15 approaches the same physical system from the prediction direction
- **P12** (Trajectory-Based Computation) — classifies computation as trajectory reconstruction; P15 is trajectory reconstruction in a biochemical system
- **P13** (FSI Drag) — the cost of symbolic instantiation; MARINA's projection matrix W_p performs the admissible compression of the conformational trajectory

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
