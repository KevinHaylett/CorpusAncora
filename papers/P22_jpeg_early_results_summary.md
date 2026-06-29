# P22 — Efficient AI Embedding Compression Using JPEG: A Novel Approach for Performance and Energy Optimization

**Paper ID:** P22  
**Title:** *Efficient AI Embedding Compression Using JPEG: A Novel Approach for Performance and Energy Optimization*  
**Author:** Kevin R. Haylett  
**Location:** Manchester, United Kingdom  
**Date:** November 2024  
**Series:** Selected Communications  
**Pages:** 4  
**Status:** Legacy artifact — early/unfinished stage; provenance document  
**Primary College:** College of Machine Intelligence  
**Secondary Colleges:** College of Attralucian Studies; College of Language Dynamics; College of Finite Symbolic Mechanics  
**Primary Pillars:** P2 (Approximations/Measurements), P4 (Useful Fiction)  
**Secondary Pillars:** P1 (Geometric Container), P5 (Finite Reality)  
**Corpus Relationship:** Direct predecessor to P03 (*JPEG Compression in LLM Embeddings*); established the initial experimental basis for the JPEG compression trajectory

---

## Provenance Note

The author's own note reads: *"This document marks the early stage of my work on AI embedding compression, written in early 2024. While unfinished, it presents the key experiments and hypotheses that led directly to later developments in AI security, semantic attractors, and computational efficiency. Posted here as a legacy artifact within the broader arc of Finite Tractus and embedding-space exploration."*

P22 is included in the School not as a polished contribution but as a provenance record — the earliest captured moment of what became a significant thread in the corpus. The experiments described here led directly to P03 (the refined JPEG compression paper) and, through the compression framework, to the broader Geofinitist treatment of embedding spaces as finite symbolic structures whose redundancy is metrically accessible.

---

## Overview

P22 proposes and partially demonstrates that applying JPEG compression directly to AI embeddings — by treating them as grayscale images and running them through the standard DCT-based JPEG pipeline — can achieve substantial reductions in storage and compute cost while retaining most of the semantic information encoded in the original embedding.

The hypothesis is simple and striking: AI embeddings contain structurally similar redundant information of the same kind that JPEG was designed to exploit in natural images. The Discrete Cosine Transform (DCT) used in JPEG is not merely a visual compression tool; it is a frequency-domain filter that removes high-frequency components below some perceptual threshold. The claim is that those same high-frequency components are informationally redundant in embedding space — and their removal at compression quality 75–100 leaves cosine similarity largely intact.

This is an early claim — unproven at scale, with figures missing from the submitted document — but the experiments presented give it initial empirical support, and the direction it opened was subsequently developed in P03 and later corpus work on compression, admissibility, and the finite structure of meaning.

---

## Experimental Setup

**Model:** Sentence-BERT (transformer-based sentence embedding model)  
**Embedding format:** Converted to 8-bit grayscale matrix; treated as a 2D image  
**Compression range:** JPEG quality levels 95% → 75% (primary range); also tested at 50% and 5%  
**Metric:** Cosine similarity between original and compressed-then-decompressed embeddings  
**Replications:** Experiment repeated with a second sentence set; results consistent

---

## Key Results

**Table 3 (Similarity retention at different JPEG quality levels):**

| Sentence pair | JPEG 5% | JPEG 50% | Original |
|---|---|---|---|
| AI is transforming the world / Machine learning is changing technology | 0.5866 | 0.8309 | 1.0000 |
| The sun rises in the east / The moon orbits around the Earth | 0.4352 | 0.7894 | 0.9935 |
| Programming requires logical thinking / Mathematics helps in algorithm design | 0.6724 | 0.8546 | 1.0000 |
| Exercise improves health / A balanced diet is essential | 0.5921 | 0.8392 | 1.0000 |

Key reading: at JPEG 50% quality, cosine similarity retention is consistently 0.79–0.85 — retaining most relational structure at a fraction of the storage cost. At 5% quality, similarity degrades significantly but does not collapse completely, suggesting the DCT basis captures something of the high-level semantic structure even at extreme compression.

**Working range:** Quality 75–100 identified as reliable for embedding compression without significant performance degradation.

---

## Computational Efficiency Analysis

**Table 1 (FLOPs reduction):**

| Scenario | Original | Compressed | Reduction |
|---|---|---|---|
| No compression | 175T FLOPs | 175T FLOPs | 0% |
| 2:1 compression | 175T FLOPs | 87.5T FLOPs | 50% |
| 5:1 compression | 175T FLOPs | 35T FLOPs | 80% |
| 10:1 compression | 175T FLOPs | 17.5T FLOPs | 90% |

**Table 2 (Energy trade-off):**

| Factor | Full Precision | JPEG 10% Compression |
|---|---|---|
| Memory Transfer (GB/s) | 100% | 10% (90% reduction) |
| Compute Cost (FLOPs) | 100% | 105% (decompression overhead + AI) |
| Power Usage (Watts) | High (DRAM-bound) | Lower (compute-bound) |
| GPU Acceleration | Standard matrix ops | GPU-accelerated JPEG decoding |

The key insight: JPEG decompression (IDCT + dequantization + entropy decoding) is cheap on modern AI accelerators — GPU/TPU/FPGA pipelines already include optimised JPEG decoding hardware. The computational overhead of decompression is negligible relative to the FLOPs saved by operating on smaller embeddings. The system shifts from memory-bandwidth-bound (expensive) to compute-bound (cheaper). The environmental implication is noted: at 10:1 compression, datacenter energy costs could fall by up to 90%.

---

## Method (§5)

1. Generate sentence-pair embeddings with Sentence-BERT
2. Convert embeddings to 8-bit grayscale image format (normalise to [0, 255])
3. Apply JPEG compression at quality levels 95 → 75 (primary range; also 50, 5 tested)
4. Decompress; measure cosine similarity to original
5. Analyse relationship between quality level and similarity retention

The approach is deliberately minimal — no retraining, no architectural changes, no custom compression algorithm. JPEG is applied as-is, with the hypothesis that its DCT basis already captures the frequency structure relevant to semantic similarity.

---

## Discussion

The paper articulates three practical advantages of JPEG compression for AI embeddings:

1. **Memory efficiency**: compressed embeddings require less transfer bandwidth — at 10:1 compression, memory transfer falls by 90%. For real-time search and retrieval (millions of embedding comparisons), this halves latency without algorithmic change.

2. **Hardware compatibility**: GPU, TPU, and FPGA pipelines already include dedicated JPEG decoding. Integration is seamless — no custom silicon required.

3. **Environmental footprint**: AI inference is increasingly DRAM-bound; compression shifts the bottleneck to compute-bound processing, reducing energy per query.

The discussion also acknowledges the limitation: JPEG compression is image-dependent, and the analysis is not yet extended to diverse architectures, task types, or compression paradigms at scale. These are explicitly flagged as future work.

---

## Significance for the School

P22 is the seed of the compression thread that runs through the corpus. The question it opens — *can the redundancy structure of embedding space be exploited by standard compression algorithms originally designed for perceptual data?* — is a Geofinitist question in disguise: it asks whether the finite symbolic structure of AI representations is metrically accessible using tools developed for physical (visual) signals.

That question was sharpened in P03 (*JPEG Compression in LLM Embeddings*), where the experimental basis was extended and the connection to frequency-domain semantics was formalised. It is connected to P06 (*The Measured World* / symbolic compression and correspondence) in its treatment of compression as a measurement operation — selecting which components of a representation are informationally essential. And it anticipates the admissibility framework (P14): the JPEG quality threshold acts as an admissibility condition on semantic content, separating what is retained from what is discarded.

The provenance status of P22 is explicit and important. It is not presented as a finished contribution but as a record of where the work began — the initial experimental observation from which a significant line of enquiry grew.

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
