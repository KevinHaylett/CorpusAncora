# P17 — Protein-Ligand Affinity as Multiscale Correspondence: Part 2

**Full title:** *Protein-Ligand Affinity as Multiscale Correspondence: A Takens-Based Programme for Sequence-to-Structure and Affinity Modelling — Part 2: The Construction Signal as a Literal Dynamical Time Series — Transcription, Introns, and Co-Translational Dynamics*  
**Paper ID:** P17  
**Author:** Kevin R. Haylett  
**Location:** Manchester, UK  
**Series:** Selected Communications  
**Date:** May 24, 2026  
**Pages:** 3  
**Primary College:** College of Attralucian Studies  
**Secondary Colleges:** College of Machine Intelligence; College of Finite Symbolic Mechanics  
**Primary Pillars:** P3 (Dynamic Flow), P5 (Finite Reality)  
**Secondary Pillars:** P2 (Approximations/Measurements), P1 (Geometric Container)  
**Status:** Stable (addendum to P16)  
**Source:** `P17_protein_binding_part2.pdf`

---

## Overview

P17 is a three-page addendum to P16, sharpening one key claim: the phrase "ordered symbolic trace" used in P16 to describe the amino acid sequence is not merely a metaphor. The biological construction process — DNA transcription, pre-mRNA splicing, co-translational folding — is a **literal dynamical time series**. This strengthens the Takens-based framing considerably: Takens' theorem was developed precisely for dynamical time series, and the protein biosynthesis pathway is one.

The addendum extends the mathematical framework, identifies four immediate modelling implications, and poses five open research questions for future investigators. It closes the protein cluster (P15–P17) while leaving the door open for the TBT programme to be applied to new domains.

---

## The Central Sharpening

In P16, the amino acid sequence P = (p₁, p₂, ..., p_N) was treated as a construction signal from which hidden geometry could be reconstructed. P17 makes the time-series nature literal, not figurative:

**Transcription** — RNA polymerase advances base-by-base in real time, producing the pre-mRNA signal sequentially.

**Introns** — retained in the raw transcript; not extraneous noise but an integral part of the temporal signal carrying essential regulatory information: splice-site signals, transcriptional pausing elements, chromatin-looping anchors, and evolutionary modules for exon shuffling.

**Splicing** — operates on the temporal signal T, not on a static bag of exons; may be co-transcriptional or post-transcriptional but is sequential.

**Translation** — strictly vectorial; the ribosome reads codons one-by-one; the nascent polypeptide begins folding inside the exit tunnel while later residues are still being synthesised.

**Consequence:** long-range sequence dependencies in the final protein are shaped by temporal constraints already present in the primary transcript. The affinity label a supervises a correspondence spanning the entire dynamical history: genomic DNA → full transcript time series → spliced protein → folded multiscale object.

---

## Extended Mathematical Framework

Let the observable construction signal be the full primary transcript (pre-mRNA):

$$T = (t_1, t_2, \ldots, t_M)$$

where each t_j is a nucleotide (or codon-level token), M ≫ N, including both exons and introns.

The mature protein sequence P is obtained via the splicing map:

$$P = \mathcal{S}(T)$$

Delay-coordinate vectors on the transcript signal:

$$\Phi^T_{\tau,d}(j) = \bigl(e^T_j,\; e^T_{j+\tau},\; \ldots,\; e^T_{j+(d-1)\tau}\bigr)$$

Multi-delay family for the full transcript:

$$\mathcal{E}(T) = \bigl\{\Phi^T_{\tau_r,d_r}(j) : r = 1,\ldots,R;\; j = 1,\ldots,M\bigr\}$$

**Delay semantics on the transcript:**
- Short delays: local splicing signals, codon usage patterns
- Intermediate delays: intron-mediated exon pairing, co-transcriptional RNA secondary structure
- Long delays: distant genomic positions that become spatially or functionally related after splicing and folding

**Extended affinity architecture** (extending Eq. 29 in P16):

$$\hat{a} = H_\theta\bigl(\mathcal{E}(P),\; \mathcal{E}(T),\; \mathcal{E}^L(L),\; C^{PL}(P,L),\; q\bigr)$$

The splicing map S can itself be learned or treated as a differentiable operation within the model.

---

## Four Practical Extensions

**1. Transcript-aware input layer:** Replace or augment the mature protein sequence input with paired genomic/transcript data (available from RefSeq, Ensembl, GTEx). Delay embeddings are built on the longer T signal before splicing.

**2. Intron-aware delay ablation:** Systematically vary delay families on intron-containing transcripts to quantify which separations (splice-junction proximity, intron length) contribute most to structure and affinity prediction. Extends Step 2 of the P16 research programme with an explicit biological interpretation via intron positions.

**3. Co-translational regulariser:** Add a causal (autoregressive) component that simulates the growing chain, using delay coordinates to enforce that early residues influence later folding while the chain is still incomplete. Mirrors the actual biological process where co-translational folding occurs in the ribosome exit tunnel.

**4. Joint structure–affinity–splicing objective:** Extends the P16 training objective to include a splicing reconstruction term:

$$\mathcal{L} = \lambda_S \mathcal{L}_{\text{structure}} + \lambda_A \mathcal{L}_{\text{affinity}} + \lambda_B \mathcal{L}_{\text{binding}} + \lambda_{\text{splice}} \mathcal{L}_{\text{splice}}$$

---

## Five Open Research Questions

1. How much additional predictive power does the full primary transcript provide over the mature protein sequence alone, especially for proteins with complex alternative splicing?

2. Can learned delay families recover known biological periodicities (exon length distributions, intron-mediated pausing) without explicit supervision?

3. To what extent do intron-containing delay embeddings improve generalisation to novel folds or ligand scaffolds outside the training distribution?

4. Can the model be trained end-to-end to predict both splicing patterns and binding affinity from genomic sequence, thereby closing the loop from DNA to phenotype?

5. What new diagnostics (e.g., delay-attention maps) become possible once the construction signal is treated as a true dynamical trace?

---

## Relation to the Core Programme

P17 does not alter the core programme of P16; it makes its biological grounding more precise. The six-step research programme in P16 remains intact:
- Step 1 (sequence-to-structure validation) can now be performed on both spliced and unspliced signals
- Step 2 (delay-scale ablation) gains an explicit biological interpretation via intron positions

The addendum is complete as stated: it closes the current line of protein-structure/affinity work given available compute and time constraints. Future work in the TBT programme will explore **alternative domains** — showing the Takens-based approach as a general nonlinear dynamical prediction instrument applicable wherever a sequential symbolic construction signal can be identified.

---

## Connections to Other Work

- **P16** (Multiscale Correspondence, Part 1) — the parent paper; P17 is a direct addendum, sharpening the time-series claim
- **P15** (MARINA) — the proof-of-concept architecture; P17 points toward a richer input signal (transcript rather than mature sequence) for future implementations
- **P01** (TBT for language) — language sequences are also, in a weaker sense, "produced" sequentially; the co-translational analogy connects to the left-to-right generation process in language models
- **ATT_38** (The Generonic Boundary) — the five-stage observation pipeline; transcription-splicing-translation is the most literal biological instance of the generonic pipeline: interaction → capture → projection → corpus integration → correspondence modelling
- **M02** (FSET) — the formal theorem that finite symbolic sequences admit delay-coordinate reconstruction; intron-containing transcripts extend the alphabet and the sequence length but remain within the FSET framework
- **ATT_81** (Functional Symbolic Trajectory) — the FST as a framework for understanding how symbols carry history through context; the pre-mRNA transcript is the most literal available example of a symbol (the codon) carrying its full historical context (the surrounding intron/exon structure)

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
