# P03 — JPEG Compression of Token Embeddings in Large Language Models

**Paper ID:** P03  
**Title:** JPEG Compression of Token Embeddings in Large Language Models: Revealing Linguistic Attractor States and a Novel Embedding-Level Security Vulnerability  
**Author:** Kevin R. Haylett  
**Date:** May 2026  
**College (primary):** College of Machine Intelligence  
**College (secondary):** College of Language Dynamics; College of Finite Measurements  
**Pillars (primary → secondary):** P3, P2 (primary); P1, P4, P5 (secondary)  
**Status:** Stable *(content correct; written to fill a gap; may be refined)*  
**Position in sequence:** Second of two JPEG compression papers. The first established that compression worked at all. This paper introduces the attractor interpretation and the security finding, and set the trajectory for all subsequent work on language as a dynamical system.  
**Keywords:** Geofinitism, finite mechanics, LLM embeddings, nonlinear dynamics, linguistic attractors, JPEG compression, AI security, embedding corruption

---

## Abstract

> Large language models (LLMs) represent meaning as high-dimensional token embeddings. This paper introduces a novel experimental technique: injecting **JPEG compression** — a finite, lossy, geometric transformation — directly into the embedding layer of a GPT-2 model. By simulating controlled information loss at the embedding level, we demonstrate that LLM cognition does not degrade randomly but collapses into **structured linguistic attractors**. These attractors mirror psychological and geometric patterns (recursion, paranoia, categorical rigidity, paradoxical insight).
>
> Beyond the theoretical insight into the **geometry of language**, the method exposes a critical, previously undocumented AI security vulnerability: **covert embedding corruption**. An adversary can subtly distort embeddings without altering model weights, training data, or visible prompts, enabling invisible manipulation of AI behavior in finance, military, media, and law-enforcement systems. We provide the full implementation, experimental results across compression qualities (95% to 1%), and a security analysis. This work advances Geofinitism's finite-mechanics framework and calls for immediate embedding-integrity defenses.

---

## Architectural Note

P03 is the empirical origin of the Geofinite language dynamics programme. The central finding — that progressive embedding compression does not produce random degradation but systematic collapse into stable attractor basins — is the first direct experimental confirmation that language, as instantiated in an LLM, has a **low-dimensional geometric skeleton**. Everything that follows in the programme (language as a nonlinear dynamical system, Takens-Based Transformers, trajectory-based meaning) is downstream of this finding.

The JPEG technique is itself a Geofinite measurement instrument: a finite, lossy, geometric transformation used not to process images but to **probe the attractor landscape** of a symbolic system. This is finite mechanics applied empirically.

The security finding is a consequence of the same geometry: if embeddings have structured attractor basins, then embedding-level distortion is a controlled, predictable manipulation — not noise.

**Connection to Attralucian Essays:** This paper provides the empirical foundation for theoretical claims made in ATT_06 (Geodesic Fractal LLMs), ATT_07 (Non-linear Dynamical Systems Fractal Model), and ATT_30 (Words as Trajectories). It also connects directly to the Takens Reconstruction framework of ATT_25 — the compression sweep is, in effect, a dimension-reduction probe of the same kind Takens formalises.

---

## Introduction

Geofinitism posits that meaning arises from finite, measurable geometric interactions rather than infinite abstractions. The experiment tests whether a real-world compression algorithm — designed for images, not language — can expose the underlying dynamical structure of LLM cognition.

JPEG was chosen because it is:
- **Finite and geometric** — discrete cosine transform (DCT) + quantization on 2D blocks
- **Lossy yet structured** — information is discarded in a way that preserves perceptual essentials
- **Computationally lightweight** and domain-agnostic

By applying it token-by-token to embeddings, the model is forced to operate under **progressive geometric constraint**, revealing the attractor landscape of linguistic meaning.

---

## Methodology

A custom PyTorch layer is inserted into a `ModifiedGPT2Model` subclass of `GPT2LMHeadModel`. The compression operates **between the embedding lookup and the transformer blocks** — all transformer weights are left untouched.

**Per-token process:**
1. Pad the 1D embedding vector to even length if necessary
2. Reshape into a 2-row 2D array
3. Normalise to [0, 255] (treat as an image)
4. Save to an in-memory JPEG buffer at a given quality level
5. Decompress the JPEG
6. Inverse-normalise and flatten back to the original 1D vector

Quality levels swept: **95%, 75%, 50%, 25%, 10%, 5%, 1%** (95% ≈ near-lossless; 1% ≈ extreme compression).

The technique is:
- **Non-invasive** to model weights
- **Controllable** (quality parameter = compression severity)
- **Repeatable** (same quality → same attractor, regardless of prompt)

Full code, tokenizer, and inference loop available at FiniteMechanics.com.

---

## Results — Progressive Collapse into Attractor Basins

Neutral prompts produce systematically different outputs at each compression level. Crucially, **degradation is not random**. The model repeatedly enters the same attractor basins regardless of prompt.

| Quality | Observed Attractor State |
|---|---|
| **95%** | Minor recursion; thought remains largely coherent |
| **75–50%** | Categorical rigidity — structured Q&A format; thought becomes schematic |
| **25–10%** | Collapse into paranoia, existential despair, and self-referential loops |
| **5%** | Fixation on violence, extreme recursion, aggressive paranoia |
| **1%** | Emergence of Zen-like paradoxes — profound yet disconnected from original context |

**Key finding:** Language possesses a **low-dimensional geometric skeleton** under finite constraint. The JPEG probe makes this skeleton visible. Each compression level is a different cross-section of the same underlying attractor landscape.

This is consistent with Geofinitism's prediction: symbolic systems under finite constraint converge to stable attractor regions, not random noise.

---

## Security Implications — Embedding Corruption as Stealth Attack

The same geometry that reveals attractors constitutes a new attack class. An adversary applying controlled JPEG distortion to embeddings can:

1. **Military / Geopolitical** — make neutral situations appear highly aggressive, or vice versa
2. **Public Opinion** — nudge recommendation and summarisation engines toward fear or polarisation
3. **Corporate Espionage** — quietly bias investment-risk or strategy AI systems
4. **Law Enforcement / Surveillance** — silently compromise fraud detection or predictive policing

**Why it bypasses existing defences:** Traditional AI security focuses on tokens, prompts, or model weights. Embedding-level distortion leaves all of these untouched. Prompt filtering, weight monitoring, and adversarial training do not protect embedding integrity.

**Recommended countermeasures:**
- Cryptographic hashing or integrity verification of embeddings before transformer entry
- Anomaly detection in embedding-space trajectories
- Redundant multi-encoder consistency checks
- Formal documentation of "embedding corruption" as a new attack class

---

## Discussion

Two findings:

**Theoretical:** This paper provides the first empirical demonstration (within the Geofinitism framework) that LLM cognition follows finite geometric dynamics. The JPEG lens acts as a measurement tool that forces the system to reveal its own attractor structure — exactly as Finite Mechanics predicts.

**Practical:** Embedding corruption is a real, currently undefended attack surface. The technique is lightweight, controllable, invisible to standard defences, and operates on production-grade models without modification.

---

## Conclusion

By applying a simple, finite geometric operation (JPEG) to embeddings, the paper:
1. Confirms the geometric nature of linguistic meaning
2. Exposes a new class of invisible AI attack

**Future directions:**
- Extend the technique to other modalities and larger models (GPT-2 → GPT-4-class)
- Map the full attractor landscape of language systematically
- Develop and open-source embedding-integrity toolkits
- Prepare a formal security advisory for the AI research community

---

## Re-style Checklist (LaTeX)

*(Paper uses `article` class — structurally clean. No `book`-class issues apply.)*

- [ ] Bibliography oversized: `\thebibliography{9}` declares 9 slots but contains only 3 entries — reduce to `{3}` or `{99}` (cosmetic only)
- [ ] Code listing shows `io` and `Image` usage without import statements in the listing — note for replication documentation

---

## Cross-References

### Within School of Geofinitism

| Essay / Paper | Connection |
|---|---|
| ATT_06 | Geodesic Fractal LLMs — P03 provides the empirical backbone for ATT_06's theoretical claim that LLMs operate as finite geometric dynamical systems |
| ATT_07 | Non-linear Dynamical Systems Fractal Model — the attractor states found here are the attractors ATT_07 models theoretically |
| ATT_30 | Words as Trajectories — the attractor basins revealed by P03 are the trajectory attractors ATT_30 describes conceptually |
| ATT_25 | Complex Analysis as Takens Embedding — the JPEG quality sweep is a finite compression probe of the same attractor structure Takens reconstruction would recover from time-series observations |
| ATT_52 | Finite Process Unfolding — the successive quality levels (95% → 1%) are a discrete FPU of the LLM's attractor landscape |
| ATT_62 | Measurement Constraint Thesis — JPEG compression is a finite measurement instrument; the quality parameter is the resolution bound $\rho$ applied to the embedding space |
| ATT_59 | Geofinite Kolmogorov Complexity — compression quality maps conceptually to the complexity bound $B$; lower quality = tighter resource constraint = simpler representational regime |
| P01 | Takens-Based Transformer *(pending)* — the TBT is the constructive response to the diagnostic finding of P03; P03 shows language has geometric attractor structure; P01 builds a transformer that exploits this directly |
| P04 | Non-linear Dynamics in LLMs *(pending)* — P04 formalises theoretically what P03 demonstrates empirically |
| P05 | Language as a Nonlinear Dynamical System *(pending)* — the theoretical companion; P03 is the experiment, P05 is the framework |

### External Literature

| Reference | Role |
|---|---|
| Radford et al. 2019 (GPT-2) | Base model — GPT-2 architecture used for all experiments |
| JPEG standard (ISO/IEC 10918) | The compression instrument — DCT + quantization |
| Takens' Embedding Theorem | Theoretical backdrop — finite-dimensional attractor reconstruction from observations |

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
