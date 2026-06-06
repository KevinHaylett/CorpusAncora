# P10 — From Formula to Process: Bridging Machine Learning Mathematics and Nonlinear Dynamics

**Full title:** From Formula to Process: Bridging Machine Learning Mathematics and Nonlinear Dynamics  
**Paper ID:** P10  
**Author:** Kevin R. Haylett, PhD — Manchester, UK  
**Date:** 2026 (references dated 2025a, 2025b)  
**Journal:** Journal of Geofinitism  
**Primary College:** College of Machine Intelligence  
**Secondary Colleges:** College of Finite Symbolic Mechanics; College of Language Dynamics  
**Primary Pillars:** P3 (Dynamic Flow), P5 (Finite Reality)  
**Secondary Pillars:** P2 (Approximations/Measurements), P4 (Useful Fiction)  
**Status:** Stable  
**Source:** `P10-bridging-ml-to-takens.pdf` (PDF only — no .tex source)  
**Published at:** `finitemechanics.com/papers/` (implied by series)

---

## Abstract (verbatim)

> Modern machine learning is commonly introduced through a small set of powerful mathematical forms: affine transformations, activation functions, loss functions, gradient descent, probability distributions, and scaled dot-product attention. These forms are useful and successful descriptions. They allow engineers and researchers to reason about models at a high level of abstraction.
>
> These formulas are not wrong. They are highly effective symbolic compressions.
>
> However, the compression can hide something essential: no physical machine performs a formula as a whole. A processor does not instantiate an affine transformation or an attention head as a single mathematical object. It performs a finite sequence of operations—fetching, multiplying, accumulating, rounding, storing, and updating—coordinated across hardware. Even parallel execution on GPUs or TPUs is itself a carefully orchestrated set of finite state changes.
>
> The purpose of this essay is to build a bridge between the conventional symbolic description of machine learning and a process-based, finite, measurable description grounded in Finite Mechanics. The bridge does not discard the mathematics. It unfolds it.
>
> The central claim is simple: Modern AI is commonly described through symbolic mathematical compressions, but it is physically instantiated as finite sequential measurement and update processes. This distinction matters because it opens a direct path from machine learning into nonlinear dynamics, time-series reasoning, and a deeper account of how meaning, learning, and attention emerge from ordered processes rather than static formulas.

---

## Architectural Note

P10 occupies a unique position in the programme: it is simultaneously an accessible entry-point for ML researchers and a formal synthesis of the Geofinite reframing of AI. It explicitly bridges the conventional ML account (formulas, gradients, attention) and the Geofinite account (finite processes, sequential measurement, dynamical trajectories).

The paper is already branded **Journal of Geofinitism** in its running header — the first paper in the programme to carry this designation formally. It cites P02 (Haylett 2025a) and P01 (Haylett 2025b) as companion works and positions itself as the conceptual bridge that makes those technical papers accessible.

Where P08 and P09 critique what ML cannot do, P10 explains what ML *actually is* from a process perspective — and shows that this reframing opens the door to dynamical systems analysis naturally and constructively, without requiring ML researchers to abandon their existing account.

---

## Core Thesis

ML formulas are **symbolic compressions of finite sequential measurement-and-update processes**. Restoring the suppressed time index transforms timeless algebraic equations into dynamical trajectories. Once this is recognised, machine learning is naturally understood as a nonlinear dynamical system — and the Takens-Based Transformer is the direct architectural consequence of taking this recognition seriously.

The central bridge statement:
> *Machine learning formulas describe what is compressed; process dynamics describe how it is physically unfolded.*

---

## Key Concepts

### Two Descriptions of Every ML Function

Every machine learning function has two legitimate descriptions that a mature account requires both of:

| Description | Form | Character |
|---|---|---|
| **Symbolic compression** | $f_\theta(x)$ | The map — useful for mathematical reasoning, elegant, compact |
| **Operational unfolding** | $x \to \text{encode} \to \text{multiply} \to \text{accumulate} \to \text{activate} \to \text{store} \to \text{propagate}$ | The measured terrain — what the machine actually does |

The formula `z = Wx + b` compresses what is physically enacted as:
$$z_i = (((W_{i1}x_1 + W_{i2}x_2) + W_{i3}x_3) + \ldots) + b_i$$
which in turn hides: representation, rounding, finite precision, memory movement, clock cycles, hardware scheduling, and energy expenditure.

> "The formula is therefore not the physical event. It is a compact symbolic trace of the event."

### The Loss Landscape Is Not Simply Given

The conventional view treats training as search through a known landscape: $\theta^* = \arg\min_\theta \mathcal{L}(f_\theta(x), y)$.

The process view reveals this is incomplete. The machine never sees the whole landscape — it samples it through sequential measurement. Training is an iterative construction of local terrain measurements:

$$\theta_t \to f_{\theta_t}(x_t) \to \hat{y}_t \to \mathcal{L}(\hat{y}_t, y_t) \to \nabla_\theta \mathcal{L}_t \to \Delta\theta_t \to \theta_{t+1}$$

> "The landscape is revealed through sequential measurement."

### Learning as Sequential Measurement

At each training step: **Measure → Compare → Update.**

$$S_{t+1} = F(S_t, x_t, y_t, E_t)$$

where $S_t$ includes parameters, optimizer state, activation patterns, and memory structures. The model is not merely "fitting data" — it is **reshaping its own future measurement responses** by repeatedly encountering finite symbolic differences. This is the bridge to nonlinear dynamics: a learning system is a state-evolving system.

### The Missing Time Index

Many ML formulas suppress time for compactness — but this hides the dynamical nature of the system. Restoring the $t$ subscript transforms static equations into trajectories:

| Compressed (timeless) | Process (time-restored) |
|---|---|
| $z = Wx + b$ | $z_t = W_t x_t + b_t$ |
| $\mathcal{L}(f_\theta(x), y)$ | $\mathcal{L}_t(f_{\theta_t}(x_t), y_t)$ |
| $\theta \leftarrow \theta - \eta\nabla_\theta \mathcal{L}$ | $\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}_t$ |

> "Without $t$, the equation looks like timeless optimization. With $t$, it becomes a dynamical process."

### The Machine as Finite Measuring Apparatus

A model measures: input relations, prediction error, local gradients, token compatibility (attention), distributional mismatch (cross-entropy/KL), uncertainty (entropy), alignment (dot products/cosine similarity). But each "measurement" is a **finite symbolic operation inside a constructed system** — not direct contact with reality.

This yields a precise, non-mystical account:

> "A model constructs finite symbolic measurements over learned relational structures, and its outputs are trajectories through those structures."

An embedding is not meaning — it is a finite symbolic measurement-like projection. Cosine similarity is not the geometry of meaning — it is one observable taken from that geometry.

### Attention as Relational Measurement

The conventional formula $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ is a compression. Unfolded as process:

$$s_{ij} = q_i \cdot k_j, \quad \tilde{s}_{ij} = \frac{s_{ij}}{\sqrt{d_k}}, \quad a_{ij} = \frac{\exp(\tilde{s}_{ij})}{\sum_l \exp(\tilde{s}_{il})}, \quad o_i = \sum_j a_{ij} v_j$$

Each dot product is a **measured geometric alignment** between two symbolic states at different sequence positions. The attention block is therefore a pairwise comparison across a finite context window — structurally identical to delay-coordinate embedding (Takens 1981). The learned projections $W_Q$ and $W_K$ play the role of delay coordinates; the context window supplies the finite history.

> "The conventional account says: attention finds relevance. The dynamical account says: attention reconstructs relational geometry from a finite symbolic time series."

### Three Worked Examples — The Unfolding Programme

P10 provides three worked examples that concretely demonstrate the formula→process→dynamical reinterpretation:

**Affine layer:** $z = Wx + b$ → $z_i = \sum_j W_{ij}x_j + b_i$ → sequential fetch/multiply/accumulate/store. *Geofinite interpretation:* symbolic compression of a finite sequential measurement-and-accumulation process.

**Attention:** compressed formula → expanded $s_{ij}, \tilde{s}_{ij}, a_{ij}, o_i$ sequence → *Dynamical interpretation:* attention reconstructs local symbolic geometry from finite history available in the context.

**Training:** $\theta^* = \arg\min_\theta \mathcal{L}$ → $\theta_0 \to \theta_1 \to \ldots \to \theta_T$ → each step: $\hat{y}_t = f_{\theta_t}(x_t), E_t = \mathcal{L}(\hat{y}_t, y_t), g_t = \nabla_\theta E_t, \theta_{t+1} = \theta_t - \eta g_t$. *Dynamical interpretation:* training is a trajectory through parameter-state space, driven by measured error signals.

### Nonlinear Dynamics as Natural Language of Process

Once ML is viewed as sequential measurement, a prompt is an initial perturbation, a model's response is a trajectory, the context window is a finite container, hallucination and collapse are trajectory phenomena. Temperature changes the sharpness of transition probabilities. Repetition and sudden shifts in tone are attractor/basin phenomena — expected, not surprising.

> "Language generation is not merely classification over vocabulary. It is finite symbolic trajectory formation."

### The TBT as Concrete Dynamical Realisation

P10 closes by showing that the TBT (P01/MARINA) operationalises every insight of the bridge:

| Insight from the bridge | TBT realisation |
|---|---|
| Formulas should be unfolded into finite sequential operations | Explicit delay embedding replaces the compressed attention formula |
| Learning is sequential measurement and update | Each forward pass reconstructs the current manifold state from measured history |
| The loss landscape is sampled, not known | Manifold projection learns local curvature from observed trajectories |
| Time indices matter | Every step advances the trajectory by one discrete update |
| Meaning emerges from ordered processes | Semantics are geometric flows on the learned attractor |

Empirical results: 15M parameters, CPU-only; Brown Corpus training loss 1.88 / validation 4.21; memory fibres (narrow tubular attractors) for factual recall; Channel Theory for structural separation of user/system/internal channels.

The TBT is not merely a language model — it is a **generic finite symbolic dynamical system simulator**: replace token vocabulary with quantized sensor readings and the same delay-reconstruction + manifold-projection loop recovers any observable nonlinear system's attractor.

---

## Five Pillars in P10

| Pillar | Role in P10 |
|--------|-------------|
| **P3 — Dynamic Flow** (primary) | Learning as sequential dynamical process; training as trajectory through parameter-state space; inference as trajectory through symbolic state space; attention as reconstruction of relational geometry; ML systems as nonlinear dynamical systems |
| **P5 — Finite Reality** (primary) | No physical machine performs a formula as a whole — only finite sequential operations; finite measuring apparatus; finite context window; O(1) memory, O(log N) per-token in TBT; all ML "measurements" are finite symbolic operations |
| **P2 — Approximations/Measurements** (secondary) | Loss as measured discrepancy; gradient as measured local direction; attention as measured geometric alignment; cosine similarity as one observable from a geometry; the machine as a finite measuring apparatus throughout |
| **P4 — Useful Fiction** (secondary) | ML formulas as useful symbolic compressions — not wrong, but incomplete; the timeless form as a useful fiction when the time index is suppressed; the symbolic form as the map, the process as the terrain |

---

## Connections to Other Work

- **P01** (TBT/MARINA): explicitly cited as Haylett 2025b; P10 is the accessible conceptual bridge that P01 implements concretely; the TBT section (§17) reproduces P01's core architecture in plain prose
- **P02** (Pairwise Phase Space Embedding): explicitly cited as Haylett 2025a; §7 (Attention as Relational Measurement) describes the connection P02 proves formally — attention as structurally identical to delay-coordinate embedding
- **P06** (The Measured World): P10's "formula as symbolic compression" thesis directly extends P06's compression framework into ML mathematics; $z = Wx + b$ is an instance of P06's compression mechanism
- **P08** (Autoregression Is Not Takens): P10 frames the positive case (what ML is) that complements P08's critique (what autoregression cannot guarantee); together they are the diagnostic and constructive sides of the same argument
- **P09** (Static Vector Insufficiency): P10's process view explains *why* static embeddings are inadequate — they are degenerate compressions that permanently suppress the $t$ index, eliminating the trajectory structure that meaning requires
- **P04** (Finite-Symbol Embedding Theorem): P10 provides the accessible motivation for why Takens applies to ML; P04 provides the formal mathematical licence
- **P05** (Language as Nonlinear Dynamical System): P10 establishes that ML systems are dynamical systems, grounding and extending P05's dynamical linguistics thesis in the architecture of physical AI
- **ATT_52** (Finite Process Unfolding): P10's "unfolding" programme is FPU applied to ML formulas — recovering the temporal structure hidden inside the compressed algebraic form
- **ATT_60** (Geofinite Learning Thesis): P10 is the accessible companion; ATT_60 provides the formal Geofinite account of learning under measurement constraints
- **ATT_06/ATT_07** (Geodesic Fractal / NDS Fractal): early formulations of the Geofinite AI reframing that P10 formalises and extends with the explicit bridge methodology

---

## Re-style Notes

| Issue | Location | Fix |
|---|---|---|
| PDF source only | — | No .tex available; Document Editor must reconstruct from PDF if restyling needed |
| Double period in abstract | "nonlinear dynamical systems modeling.." | Remove extra period |
| Inconsistent closing punctuation in §17 | `systems.".'` and `as time series.".'` | Remove extra period/quotes after closing quotation marks |
| "Journal of Geofinitism" already in running header | Throughout | ✓ Correct — first paper to carry the Journal designation formally |

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
