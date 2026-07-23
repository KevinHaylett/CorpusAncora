# PE08 — Measurement-First World Models, Takens-Based Transformers, and Trajectory Computation

**Pensée ID:** PE08  
**Title:** *Measurement-First World Models, Takens-Based Transformers, and Trajectory Computation*  
**Series:** Pensées — Speculative Arguments and Research Programmes  
**Author:** Kevin R. Haylett  
**Date:** June 2026  
**Pages:** 7  
**Primary Colleges:** College of Machine Intelligence; College of Finite Measurements  
**Secondary Colleges:** College of Attralucian Studies; College of Finite Symbolic Mechanics; College of Philosophy; College of Language Dynamics  
**Primary Pillars:** P2, P3, P4  
**Secondary Pillars:** P1, P5  
**Status:** Stable

---

## Core Claim

Topology should not be treated as the primitive object in world modelling. It emerges from ordered measurement, memory, and trajectory reconstruction. A Takens-Based Transformer, suitably modified, provides a route toward video, CCD, signal, and multimodal world modelling in which each frame or measurement field is treated as a state within a dynamical trajectory rather than as a static image patch.

The central inversion:

> Measurement comes first. Time series comes second. Memory binds the sequence. **Topology is reconstructed** from the measured trajectory.

Versus the prevailing implicit order: data → representation → latent manifold → prediction.

---

## Overview

PE08 is a research programme document that argues the dominant paradigm for world modelling — including LeCun's JEPA/V-JEPA programme, vision transformers, and diffusion models — remains inside a stochastic machine-learning frame that assumes topology rather than reconstructing it from measurement. The Pensée proposes a measurement-first architecture based on extending the Takens-Based Transformer (TBT) beyond language to visual, sensory, and experimental time-series data. It opens seven research paths and closes with seven concrete open questions for implementation.

---

## Section 1 — Starting Point: The Limitation of Static-Vector Thinking

The common criticism that LLMs cannot possess a world model because they are "merely word machines" is characterised as partly correct but too shallow. The deeper point is that **any serious model of intelligence must be understood as a dynamical system**. An LLM's internal representations evolve across a sequence; each token is recontextualised by the prior trajectory; attention constructs relations across an unfolding symbolic path — not fixed meanings retrieved from static storage.

The correct framing: an LLM is a **finite symbolic trajectory engine**. Its native domain is language, but the deeper computational principle is **trajectory continuation under constraint**. This opens the question of whether the same principle can be extended beyond language.

LeCun's V-JEPA and V-JEPA 2 are engaged as the most ambitious current alternatives: self-supervised foundation world models trained on video, aimed at visual understanding, prediction, and robot-control. The concern is that these still operate within a stochastic ML frame that seeks better latent representations without fully articulating the prior dynamical question: how does measured temporal experience construct the topology on which any world model depends?

---

## Section 2 — Measurement Before Topology

The measurement-first pipeline is stated as a structural alternative to the prevailing data-flow order:

**Prevailing order:** data → representation → latent manifold → prediction

**Measurement-first order:** measurement → ordered time series → memory → phase-space reconstruction → emergent topology → continuation / prediction / action

This is not a philosophical preference — it changes the architecture. If topology is assumed first, the model searches for a useful latent space. If measurement is primary, the model must preserve the structure of observation over time. The topology is an **achieved reconstruction**, not a pre-existing container.

Takens-style delay-coordinate embedding is positioned as the key technical mechanism: conditions under which the dynamics of a system can be reconstructed from time-delayed observations. The foundational emphasis: the time series is not incidental. **It is the carrier of the system.**

---

## Section 3 — Why "Patches" Matters

The word "patches" in vision transformers (ViT) reveals an architectural inheritance: the input is treated as a spatial field divided into local regions. Each patch is a local region cut from a larger surface — like a patch of grass.

This assumption becomes problematic when applied to time-series reconstruction. Converting a delay embedding or Hankel construction into a two-dimensional image-like object and then dividing it into patches risks treating reconstructed dynamics as a spatial image rather than a temporal state reconstruction.

The distinction:

**Patch approach:** time series → delay embedding → image patches → standard self-attention

**TBT approach:** time series → delay-coordinate state reconstruction → attention over dynamical states

The sharper formulation:

> "They have not made attention Takens-based. They have made Takens image-based, then applied patch attention."

The Takens embedding already carries time. Patching or rasterising it may lose, blur, or spatialise the temporal information the embedding was meant to preserve. Sora (OpenAI) is noted as an instance of this: extracting spacetime patches from compressed video, with patches acting as transformer tokens.

---

## Section 4 — Video Frames as Measured Field Tokens

A CCD or video frame is not merely a picture. It is a **finite measured field produced by an instrument**. A video sequence is therefore a measured trajectory:

frame₁ → frame₂ → frame₃ → frame₄ → ...

Each frame is a full measured state situated within a temporal sequence — not an image patch or a word token. The proposed model input:

video frame + support channels → trajectory embedding → attention over reconstructed dynamical states

**Support channels** may include: time index, exposure, sensor geometry, camera calibration, optical flow, depth estimate, uncertainty, provenance, motion estimate, object hypothesis, task instruction, language prompt, and memory state.

These channels carry different functional roles and must not be flattened into anonymous feature space. Some carry measurement. Some carry instrument conditions. Some carry uncertainty. Some carry symbolic instruction. Some carry memory.

Text remains essential but is repositioned: it acts as **entry channel, task channel, naming channel, indexing channel, and communicative bridge**. The video or CCD sequence supplies the measured trajectory.

---

## Section 5 — Diffusion Models as Potential-Landscape Traversal

Diffusion models (DDPMs, latent diffusion, Sora) are interpreted as traversing a learned landscape of potential: beginning from noise and iteratively unfolding toward coherent image or video.

This is dynamical in a broad sense: noise → partial structure → stronger structure → coherent generated field.

But it is not measurement-first reconstruction. The distinction:

> **Diffusion** builds a landscape of **potential**.  
> **TBT** reconstructs a landscape of **measured continuation**.

A combined architecture may be particularly powerful: diffusion for generative possibility, TBT for measurement-constrained trajectory reconstruction, language for symbolic task control.

---

## Section 6 — Core Conceptual Distinctions

The emerging distinction is not LLMs versus world models, or diffusion versus transformers. It is a deeper distinction between **kinds of trajectory**:

- LLMs → continue symbolic trajectories
- Diffusion models → traverse generative potential landscapes
- Video models → learn spatiotemporal continuations
- Takens/TBT models → reconstruct dynamical state from measurement history

A genuine world model may need all four, but ordered correctly. The **proposed 9-step order**:

1. Measurement
2. Time series
3. Memory
4. Delay/state reconstruction
5. Emergent topology
6. Trajectory continuation
7. Generative possibility
8. Symbolic control and exposition
9. Action or intervention

This places measurement and memory prior to topology — and topology prior to generative possibility or symbolic exposition.

The deeper hierarchy from Section 7.7: finite measurement → symbolic registration → ordered trajectory → memory → reconstruction → topology → continuation → exposition.

---

## Section 7 — Seven Research Paths

**7.1 Video-TBT Prototype** — build a modified TBT for simple video sequences. Not high-quality generation but dynamical reconstruction. Test cases: dot moving left to right, bouncing ball, occluded object, approaching car, rotating object, two-body interaction. Evaluation: temporal structure preservation, continuation prediction, occlusion handling, plausible vs. merely image-like continuation.

**7.2 Support-Channel Formalism** — formalise support channels as a defining architectural feature. Distinguish: primary measured field, timing channel, provenance channel, uncertainty channel, instrument channel, task channel, language channel, memory channel, action channel. Support channels constrain admissible trajectory evolution — they do not merely add information.

**7.3 Patch-Based vs. Trajectory-Based Transformer** — comparative study. Compare ViT patch-token, spacetime-patch video diffusion, ordinary time-series transformer, and TBT. Evaluation beyond classification accuracy: trajectory continuation, occlusion recovery, phase consistency, time-reversal sensitivity, sensor perturbation, physical plausibility.

**7.4 Hybrid Diffusion-TBT Architecture** — TBT reconstructs the measured dynamical trajectory; diffusion generates possible futures; support-channel system constrains which futures are admissible. Allows the model to distinguish: what is possible, what is plausible, what is measured, what is admissible, what is merely generatively attractive.

**7.5 CCD / Instrument-Based World Modelling** — begin with controlled CCD sequences rather than internet video. CCD is a parallel measurement instrument; provenance, geometry, exposure, and uncertainty can be explicitly recorded as support channels. Test cases: moving light source, two-source interference patterns, near-field LED/CCD interactions, rotating object under fixed illumination, mechanical oscillation.

**7.6 Language as Entry Channel Rather Than Whole Model** — text functions as instruction and indexing channel, not the whole model. Examples: "track the moving object," "predict the next frame," "describe the measured continuation." Language interrogates and conditions the trajectory model rather than replacing it.

**7.7 Mathematical Development: From Embedding to Functional Symbolic Trajectory** — Takens embedding is one mechanism within a wider theory of Functional Symbolic Trajectories. The deeper object is the finite trajectory through measured, symbolic, and remembered states. Provides a bridge between TBT, Geofinitism, measured numbers, support channels, uncertainty, provenance, and admissibility.

---

## Section 8 — Provisional Research Claim

A world model should not begin from a latent topology. It should begin from finite measurement. The topology of the world, for the model, is **reconstructed from ordered measurement, memory, and trajectory traversal**.

The research opportunity: build architectures in which symbolic trajectory, measured trajectory, support channels, memory, uncertainty, and generative possibility are brought into one nonlinear dynamical framework.

---

## Section 9 — Seven Open Questions

1. **Implementation of delay embedding in Video-TBT** — how to realise delay-coordinate embedding for high-dimensional video frames or CCD fields? Direct frame-token sequences, learned per-frame embeddings, or hierarchical coarse+local refinement?

2. **Integration of heterogeneous support channels** — how to incorporate support channels without disrupting dynamical structure? Delay-embedded themselves, or conditioning signals constraining the evolution operator?

3. **Evaluation metrics for dynamical fidelity** — beyond accuracy: trajectory continuation error, occlusion recovery, phase-space consistency (Lyapunov exponents, recurrence plots), time-reversal sensitivity, robustness to sensor perturbation, physical plausibility under known dynamics.

4. **Scaling to real video and CCD data** — delay embedding is elegant for low-dimensional attractors; what strategies (learned embeddings, multi-scale delay structures, hybrid latent + explicit delay) are needed for realistic high-resolution data while retaining measurement-first character?

5. **Hybrid Diffusion-TBT architectures** — concrete design: TBT maintains reconstructed measured trajectory and admissible region; diffusion proposes futures conditioned on that region. How to structure training objectives and inference so the model distinguishes measured continuation from generatively plausible futures?

6. **Relationship to existing dynamical-systems approaches** — how does TBT relate to recent work interpreting transformers through delay embeddings, Koopman operators, or Neural ODEs? Opportunities for hybridisation with reservoir computing or physics-informed architectures?

7. **Controlled experimental testbeds** — beyond synthetic cases, what minimal real-world CCD or sensor experiments provide strongest early validation while keeping provenance and calibration fully known?

---

## Significance for the School

PE08 is the first document in the corpus to explicitly extend the TBT framework beyond language toward video, CCD, and multimodal world modelling. It applies the Geofinite measurement-first principle to a domain where the prevailing field is moving fast — world models, video generation, robot control — and argues that the entire paradigm is inverted relative to what the measurement-first account requires.

The "patches" critique (Section 3) is the sharpest statement in the corpus of what distinguishes the TBT from vision-transformer approaches that use Takens-like structures as preprocessing devices. The nine-step ordering (Section 6) is the most explicit account in the corpus of how measurement, memory, topology, generation, symbolic control, and action should be sequenced in any genuine world model.

**Key connections:** P01 (*TBT* — the core architecture extended here); P02 (*Pairwise Phase Space Embedding* — the phase-space interpretation of attention being extended to video); PE02 (*Geometric Transformers for Physiological Time Series* — companion extension of TBT to non-language domains); P03 (*JPEG Compression* — the measurement-distortion experiments that ground the NDS interpretation); P05 (*Language as NDS* — language as trajectory-based dynamical system; PE08 extends the principle beyond language); ATT_85 (*The Instrument Came First* — CCD as instrument; video frame as finite measured field; instrument-first argument applied to world modelling); ATT_86 (*On Geofinitism* — five generative commitments, especially Generonic Instantiation and Uncertainty, directly operative in the support-channel formalism and the measurement-first pipeline)

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
