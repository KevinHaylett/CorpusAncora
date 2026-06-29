# Lesson P17-L — The Construction Signal as a Literal Dynamical Time Series

**Lesson ID:** P17-L  
**Source paper:** P17  
**Title:** *Protein-Ligand Affinity as Multiscale Correspondence — Part 2: The Construction Signal as a Literal Dynamical Time Series*  
**Difficulty:** Intermediate  
**Prerequisites:** P16-L (Multiscale Correspondence — essential); P15-L (MARINA — recommended)  
**Estimated study time:** 30 minutes

---

## Learning Objectives

After completing this lesson you will be able to:

1. Explain why protein biosynthesis (transcription → splicing → translation) is a literal dynamical time series, not a metaphor
2. State the extended mathematical framework: transcript T, splicing map S, and the extended delay-coordinate representation E(T)
3. Describe the four modelling extensions P17 proposes and the biological rationale for each
4. Explain why introns are part of the construction signal, not extraneous noise
5. Articulate the five open research questions and identify which are most tractable given current tools
6. Connect P17 to the broader TBT programme across domains

---

## Key Idea 1 — Metaphor Becomes Literal

### From "Treated As" to "Is"

P16 treated the amino acid sequence as a "construction signal" — a useful framing that motivated Takens-style delay embeddings. P17 makes a stronger claim: this framing is not a metaphor. The biological process that produces a protein sequence is, in a precise technical sense, a dynamical time series.

**The three stages of protein biosynthesis as a time series:**

**Transcription:** RNA polymerase moves along the DNA template base-by-base in physical time, producing a pre-mRNA strand nucleotide-by-nucleotide. The output is ordered, sequential, and governed by dynamics: polymerase speed, pausing, backtracking, termination. This is a scalar observable evolving in time.

**Splicing:** The pre-mRNA (which includes both coding regions — exons — and non-coding regions — introns) is processed by the spliceosome. Introns are removed; exons are joined. Splicing can be co-transcriptional (while transcription is still occurring) or post-transcriptional. It operates on the temporal signal T, not on a static set of exon sequences.

**Translation:** The ribosome reads codons one at a time, strictly left-to-right. Crucially, the growing polypeptide chain begins folding inside the ribosome exit tunnel while later residues are still being synthesised. Co-translational folding means that the final 3D structure is shaped by *which residues were available at each moment* during synthesis — a temporal constraint encoded in the sequence order.

### What This Means for Takens

Takens' theorem applies to dynamical systems — systems that evolve in time. If protein biosynthesis is a literal temporal process, then the amino acid sequence is not just a convenient signal to apply delay embeddings to. It is an actual observable of an actual dynamical system. The Takens framing is not an analogy; it is a correct description.

### Exercise 1.1

(a) A protein takes ~30–60 seconds to synthesise (for a 300-residue protein at ~4 residues/second on a human ribosome). Co-translational folding means folding begins during this time. What does this imply about the relationship between sequence order and fold topology? Can a model that ignores sequence order recover the same structure?

(b) Alternative splicing allows the same gene to produce different protein isoforms by including or excluding different exons. In P17's framework, this means the same T (primary transcript) maps to different P (protein sequences) via different splicing maps S. What does this imply about a model that uses only the mature protein sequence as input?

(c) P17 says introns carry "essential regulatory information: splice-site signals, pausing elements, chromatin-looping anchors, and evolutionary modules for exon shuffling." Which of these would appear in the delay-coordinate structure of the transcript? At what delay scale?

---

## Key Idea 2 — The Extended Framework

### From P to T

P16's framework used the mature protein sequence P = (p₁, ..., p_N). P17's extended framework uses the full primary transcript:

$$T = (t_1, t_2, \ldots, t_M), \quad M \gg N$$

where each t_j is a nucleotide or codon-level token, and T includes both exons and introns.

The connection: P = S(T), where S is the splicing map (possibly alternative).

### Extended Delay Embeddings

Delay vectors on the transcript:

$$\Phi^T_{\tau,d}(j) = \bigl(e^T_j,\; e^T_{j+\tau},\; \ldots,\; e^T_{j+(d-1)\tau}\bigr)$$

**What different delays capture:**

| Delay range | Biological content |
|---|---|
| Short (τ = 1–4) | Local codon usage, splice-site signals, immediate nucleotide context |
| Intermediate (τ = 10–50) | Intron-mediated exon pairing, RNA secondary structure formed during transcription |
| Long (τ = 100–500) | Distant genomic positions that become functionally related after splicing; chromatin loops |

The extended affinity architecture:

$$\hat{a} = H_\theta\bigl(\mathcal{E}(P),\; \mathcal{E}(T),\; \mathcal{E}^L(L),\; C^{PL}(P,L),\; q\bigr)$$

This adds E(T) — the transcript-level delay representation — to the P16 architecture. The model now has access to both the processed protein signal and the upstream construction process that produced it.

### Exercise 2.1

(a) A typical human intron is 1,000–10,000 nucleotides long. A typical exon is ~150 nucleotides. What delay τ on the transcript T would be needed to bridge across a typical intron and couple two adjacent exons? Is this feasible in the delay family used in P15/P16 ([1, 2, 4, 8, 16, 32, 64, 128])? What extension is needed?

(b) The splicing map S: T → P is described as potentially learnable or differentiable. What would it mean to learn S jointly with the affinity prediction objective? What supervision signal would be needed? What data would be required?

(c) P17 adds a fourth loss term: λ_splice · L_splice. What target would L_splice predict? Where would the supervision for splicing come from (hint: see the data sources mentioned — RefSeq, Ensembl, GTEx)?

---

## Key Idea 3 — Four Extensions and Five Questions

### The Four Extensions

**Extension 1 — Transcript-aware input layer:** Use paired genomic/transcript data as input rather than (or in addition to) the mature protein sequence. Data is available (RefSeq, Ensembl, GTEx). This extension is primarily a data engineering and input representation problem.

**Extension 2 — Intron-aware delay ablation:** The P16 research programme's Step 2 (delay-scale ablation) gains biological interpretability when applied to the transcript. Removing delays at the intron-bridging scale and measuring the impact gives a direct test of whether introns carry predictive information for structure and affinity.

**Extension 3 — Co-translational regulariser:** Add a causal (autoregressive) component that simulates the sequential growth of the polypeptide. The model is encouraged to predict structure position-by-position, with early positions constraining later ones — matching the actual physical process of folding in the ribosome exit tunnel.

**Extension 4 — Joint objective:** Extend training to include a splicing reconstruction term. If the model can predict both the correct splicing pattern and the correct affinity, it has learned a representation that spans from genomic sequence to functional phenotype.

### The Five Open Questions

P17's five questions are ordered from most immediately testable to most ambitious:

1. **Predictive power of transcript vs. mature sequence** — directly testable with existing data and modest compute
2. **Recovery of biological periodicities without supervision** — a strong interpretability test; if the learned delays match known intron/exon length distributions, the model is genuinely learning biology
3. **Generalisation improvement from intron-containing embeddings** — the key scientific test; does transcript-level information help with novel folds?
4. **End-to-end DNA-to-phenotype prediction** — the most ambitious question; requires genomic + structural + functional data at scale
5. **New diagnostics from delay-attention maps** — an interpretability question; what does the model look at, and does that match what biologists know?

### Exercise 3.1

(a) Question 2 asks whether learned delays can recover exon length distributions without explicit supervision. Design an experiment to test this. What would you measure after training? What result would confirm that the model learned biological structure?

(b) Question 4 asks about end-to-end prediction from DNA to phenotype (binding affinity). This is an extremely ambitious goal. Using P16's data construct table, trace all the compression steps from "genomic DNA sequence" to "binding affinity value." How many model-conditioned inferences are involved? At each step, what is the admissibility boundary?

(c) P17 closes by noting the programme will now move to **alternative domains** beyond proteins — applying the TBT as a general nonlinear dynamical prediction instrument. Name two domains where a sequential construction signal (analogous to the amino acid sequence) could be identified, and describe what the "observable," "hidden state," and "attractor" would be in each case.

---

## Synthesis — Closing the Protein Cluster

P15, P16, and P17 form a coherent cluster within the broader TBT programme:

- **P15** (MARINA): Does attractor reconstruction from delay coordinates actually work for a protein? Yes — proof of concept at 1.01 Å RMSD on consumer hardware.
- **P16** (Multiscale Correspondence, Part 1): What is the right theoretical framework for extending this to affinity prediction? Affinity is a multiscale correspondence over compressed construction signals; here is the complete framework and six-step research programme.
- **P17** (Part 2 addendum): Is the "construction signal" framing merely a useful metaphor? No — it is literally correct. Protein biosynthesis is a dynamical time series. Using the full transcript deepens the observable and enriches the delay structure.

Together these papers establish a principled research programme for protein structure and affinity prediction grounded in dynamical systems theory rather than statistical pattern matching. The programme is explicitly limited by current compute and data constraints — but the framework is in place for future investigators.

**What makes this cluster scientifically interesting beyond proteins:** the same structure — a sequential symbolic construction signal whose hidden geometry can be reconstructed from delay coordinates — will appear wherever a complex physical or conceptual object is built sequentially from a finite symbolic alphabet. The protein case happens to be unusually clear because the biological construction process is so well understood. But the framework is domain-agnostic.

---

## Consolidation Questions

1. P17 says introns are "an integral part of the temporal signal that evolution has tuned." Using the FSM framework (ATT_38's generonic pipeline), at which stage of the pipeline do introns operate? Are they part of the interaction, the capture, or the projection?

2. Co-translational folding means that early residues in the sequence begin adopting structure before late residues have been synthesised. Does this make the protein biosynthesis process more or less like a Takens-embeddable time series? (Consider: does Takens' theorem require the full time series to be available before reconstruction begins?)

3. The co-translational regulariser (Extension 3) adds a causal autoregressive component. Standard Transformer attention is bidirectional — each position can attend to all other positions. The causal version restricts attention to only past positions. What does this sacrifice, and what does it gain? Is the full-attention or causal-attention model more physically faithful to the biosynthesis process?

4. P17's Question 4 asks whether a model could be trained end-to-end from genomic DNA sequence to binding affinity. This closes the loop "from DNA to phenotype." From the admissibility perspective (P14), what would be required for the output â to be an admissible measurement claim rather than a model-conditioned inference? Is this possible in principle?

5. The final line of P17 notes that "recognising transcription, splicing, and translation as literal time series simply equips the framework with a richer observable and a deeper set of delay scales with which to reconstruct hidden geometric and functional constraints." In FSM terms, what is the relationship between the richness of the observable and the admissibility of the reconstruction?

---

## Further Reading

- **P16** (Multiscale Correspondence, Part 1) — the parent paper; essential prerequisite
- **P15** (MARINA) — the proof-of-concept implementation
- **ATT_38** (The Generonic Boundary) — the generonic pipeline; transcription-splicing-translation is its most literal biological instance
- **ATT_81** (Functional Symbolic Trajectory) — the codon/transcript as the most literal available example of a symbol carrying its full construction history
- **M02** (FSET) — the formal theorem enabling delay embedding of finite symbolic sequences; intron-containing transcripts are a natural extension of the FSET setting
- **P01** (TBT for language) — the parent architecture for all TBT domain applications

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
