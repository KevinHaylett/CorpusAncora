# Lesson — P22: Efficient AI Embedding Compression Using JPEG

**Paper ID:** P22  
**Lesson Type:** Provenance Reading — Historical Context and Trajectory Tracing  
**Status note:** P22 is an explicitly unfinished legacy document. This lesson is designed to be read *alongside* P03, using P22 as a provenance lens on how the compression thread began.  
**Recommended pairing:** Read P22 first, then P03 immediately after. The contrast between the two makes both richer.  
**Approximate study time:** 45 minutes (including P03 pairing)

---

## Purpose of This Lesson

P22 is not in the School because it is polished. It is in the School because it is *first*. The author's own note acknowledges this directly: unfinished, early, legacy. Its value is provenance — it marks the moment when the JPEG compression hypothesis was first tested, before the Geofinitist framework was fully articulated, and before P03 gave the experiments a cleaner form.

Reading P22 teaches something distinct from reading any finished paper: it shows what a productive line of enquiry looks like at its origin. The experiments are rougher, the figures are missing, the framing is more conventional. But the core question — *can standard image compression exploit the redundancy structure of AI embeddings?* — is already sharp. Following that question through the corpus is a study in how Geofinitist ideas develop.

---

## Part I — Reading P22

### Before You Read

Hold these questions as you read P22:

1. The paper treats AI embeddings as images. What is the implicit claim being made by that choice? What has to be true of embedding space for JPEG to work on it?
2. The hypothesis is that embeddings contain the same kind of redundancy that JPEG was designed to exploit in natural images. Is this obvious? Why might it be true? Why might it not?
3. The paper is explicitly described as unfinished — figures are missing, the method is lean, the theoretical framing is pre-Geofinitist. What can you learn from an unfinished paper that a finished one doesn't show you?

### What to Focus On

**The experimental logic (§2, §5):** The method is clean despite the paper's early state. Sentence-BERT embeddings → 8-bit grayscale matrix → JPEG compress → decompress → cosine similarity. The simplicity is the point: no custom algorithm, no retraining, no architectural change. Just JPEG, applied directly.

Notice what the results say at the two extremes. At JPEG 50% quality: cosine similarity retained at 0.79–0.85. At JPEG 5% quality: similarity falls to 0.43–0.67. The degradation is not linear — there is a region of stability (75–100 quality) and then a sharper drop. This structure matters: it implies the embedding is not uniformly distributed across DCT frequencies; the bulk of the semantic content lives in the low-frequency components.

**The energy analysis (§2.1, Tables 1–2):** The computation here is straightforward. If inference requires X FLOPs for a full-precision embedding and compression achieves 10:1, then the compressed inference requires 0.1X FLOPs for the embedding operations, plus a small decompression overhead. The shift from DRAM-bound to compute-bound is the key: AI hardware is already faster at compute than at memory access; compression favours the faster dimension.

**The provenance note:** Read it again carefully. "Key experiments and hypotheses that led directly to later developments in AI security, semantic attractors, and computational efficiency." Three downstream threads are identified. The lesson traces each.

---

## Part II — Connecting to the Corpus

### Thread 1: To P03

P03 (*JPEG Compression in LLM Embeddings*) is the direct development. Where P22 ran experiments at quality 75–95, P03 extended and refined the empirical basis. Where P22's figures are missing, P03 has them. Read P22 → P03 as a diptych.

**The key question for this pairing:** What conceptual shift happened between P22 and P03? Is P03 simply more experimental data, or did the framing change? Look for how P03 contextualises the JPEG result within the broader Geofinitist vocabulary of compression, trajectory, and finite symbolic structure.

### Thread 2: To P06 (Compression and Correspondence)

P06 (*The Measured World* / symbolic compression and correspondence) provides the theoretical scaffolding for what P22 empirically demonstrated. In P06, compression is not merely a practical technique but a measurement operation: applying a compression algorithm to an embedding is equivalent to probing which components of the representation are informationally essential to task performance.

**Question:** P22 found that cosine similarity was well-preserved at JPEG 50%. From the perspective of P06, what does this mean for the *informational content* of the high-frequency components that JPEG removes? Are they noise? Or are they part of the representation that is simply not load-bearing for semantic similarity tasks?

### Thread 3: To P14 (Admissibility)

P14 (*Admissibility and Finite Symbols*) introduces the admissibility condition: the set of symbolic representations that are deemed acceptable within a given finite symbolic system. Read through P22's JPEG quality threshold as an admissibility condition. At quality 75–100, the representation passes — it retains enough structure to remain semantically valid. Below 75, it begins to fail the admissibility test: cosine similarity drops, and the compressed embedding may no longer reliably function as a semantic proxy for the original.

**Question:** Is the JPEG quality level a measurement resolution parameter (in the sense of P2 — Approximations/Measurements), an admissibility threshold (P14), or both? Can you articulate the difference?

### Thread 4: To ATT_23 and ATT_65 (Finite Tractus)

The author's note places P22 explicitly within "the broader arc of Finite Tractus and embedding-space exploration." ATT_65 and the Finite Tractus thread treat meaning as a finite symbolic trajectory through representation space. P22, read through that lens, is asking a Finite Tractus question: *how much of the trajectory can be removed and the semantic endpoint still reached?* JPEG compression removes the high-frequency perturbations; the question is whether those perturbations are part of the trajectory that matters.

---

## Part III — Exercises

**Exercise 1 — The Structural Analogy:**
JPEG was designed for natural images, where high-frequency components often correspond to fine visual detail that the human eye is relatively insensitive to. What is the analogous claim for AI embeddings? Write a one-paragraph argument for *why* the high-frequency DCT components of an AI embedding might be semantically unimportant. Then write a one-paragraph argument for *why* they might matter.

**Exercise 2 — Compression as Measurement:**
Formulate the JPEG compression experiment as a measurement problem. What is being measured? What is the resolution parameter? What is the uncertainty? How does this connect to the measurement framework in P06 and the admissibility condition in P14?

**Exercise 3 — Provenance Tracing:**
The author identifies three downstream developments from P22: AI security, semantic attractors, and computational efficiency. Using the paper IDs available in the School, identify which papers in the corpus most plausibly correspond to each of these three threads. Give a brief (2–3 sentence) rationale for each identification.

**Exercise 4 — What Changed?**
Read P22 and P03 side by side. Write a brief comparison (one page): What remained the same from P22 to P03? What changed? What does the comparison reveal about how the compression thread developed within the Geofinitist framework?

**Exercise 5 — The Missing Figures:**
P22's §4 (Figures) contains only a placeholder: "Insert your original figure files here." The plots show cosine similarity retention vs. compression level. Without seeing the plots, sketch what you expect the curve to look like based on Table 3's data at 5%, 50%, and near-original. What shape would you predict? Why might the actual curve have been informative even in a way Table 3 is not?

---

## Part IV — Key Points to Retain

1. **The core empirical claim:** JPEG compression of AI embeddings at quality 75–100 retains cosine similarity well (>0.95 retention in most cases). This is non-obvious and productive.

2. **The mechanism:** Embedding redundancy lives in the high-frequency DCT components. JPEG removes them; semantic structure — captured by low-frequency components — survives.

3. **The efficiency case:** 10:1 compression → 90% FLOP reduction; memory-bandwidth savings; GPU-native JPEG decoding. The efficiency case does not require architectural change.

4. **Provenance role:** P22 is the origin point of the compression thread. Reading it alongside P03 and P06 situates each within a developing line of thought.

5. **The Geofinitist reading (retrospective):** P22's question — *can standard physical-signal compression exploit the redundancy structure of symbolic representation?* — is a measurement-first question. It treats the embedding as a finite symbolic object with internal structure, asks which components are informationally load-bearing, and probes the threshold between essential and redundant. That is the Geofinitist frame, even if P22 does not yet name it as such.

---

## Cross-Reference Map

| Document | Connection to P22 |
|---|---|
| P03 | Direct development — refined experiments, extended framing; read as diptych with P22 |
| P06 | Theoretical scaffolding — compression as measurement operation; symbolic correspondence |
| P14 | Admissibility — JPEG quality threshold as admissibility condition on semantic representation |
| P04 | FSET — embedding space as finite symbolic structure with recoverable geometry |
| P09 | Static vector insufficiency — critique of static embeddings; P22 asks how much can be removed while preserving what is static |
| ATT_52 | FPU — compression as the inverse of unfolding; JPEG compression removes the "unessential" trajectory components |
| ATT_65 | Finite Tractus — embedding-space exploration as the broader frame explicitly named in P22's provenance note |
| M04 | Finite Tractus foundations — JPEG compression as an early empirical touch-point for the manifold hijack question |

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
