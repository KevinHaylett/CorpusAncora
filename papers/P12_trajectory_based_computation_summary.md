# P12 — Trajectory-Based Computation

**Full title:** *Trajectory-Based Computation: Binary Logic, Ket Evolution, and Finite Symbolic Map Engines*  
**Paper ID:** P12  
**Author:** Kevin R. Haylett — Manchester, UK  
**Date:** May 2026  
**Journal:** Selected Communications  
**Primary College:** College of Machine Intelligence  
**Secondary Colleges:** College of Finite Symbolic Mechanics; College of Finite Measurements and Physics  
**Primary Pillars:** P5 (Finite Reality), P1 (Geometric Container)  
**Secondary Pillars:** P2 (Approximations/Measurements), P3 (Dynamic Flow), P4 (Useful Fiction)  
**Status:** Stable  
**Pages:** 25  
**Source:** `P12_trajectory_based_computation.pdf`

---

## Abstract (verbatim)

> This chapter develops a finite symbolic and nonlinear-dynamical language for comparing three computational trajectories: conventional binary computation, quantum ket computation, and Finite Symbolic Mechanics / Alphonic map computation. Binary computation is treated as explicit symbolic transformation through discrete states, registers, memory, and logic operations. Quantum computation is reframed as trajectory pre-loading followed by synchronous coupled evolution under Hilbert-space transformation rules and final symbolic flattening through measurement. FSM / Alphonic computation is proposed as a map-based alternative in which relational structure is preloaded into finite symbolic maps activated by a physical coupling layer. An experimental demonstration using near-field LED map formation is included as a concrete proof-of-concept.

---

## Overview

P12 applies the Functional Symbolic Trajectory framework (developed in M06 and M08) directly to computation, comparing three computational grammars under a unified FSM lens. It is both a theoretical comparative analysis and an experimental paper — Chapter 2 reports a concrete bench-top near-field LED experiment that provides the first physical proof-of-concept for Alphonic map-based computation.

The paper's central claim: **a computer is not merely a machine for calculating — it is a device for preparing, constraining, transforming, measuring, and stabilising symbols**. All three computational grammars (binary, quantum, Alphonic map) are varieties of this single activity, differing in how they carry, preload, and activate symbolic trajectories.

---

## Chapter 1 — Three Computational Grammars

### The Unifying Framework

All computation proceeds as **functional symbolic trajectory**: a finite sequence of symbolic states S₀ → S₁ → S₂ → ... → S_N, where each transition is governed by an admissible transformation rule. The same arithmetic result may be obtained by binary addition, lookup table, analogue circuit, optical transform, neural network, or quantum subroutine — the function is not uniquely bound to one symbolic pathway.

**The Generonic Boundary as coupling layer**: every computational system has an exogenous measurement entry point — a boundary at which physical events are converted into endogenous symbolic states. Measurements enter as unstable finite symbols that may be absorbed, rejected, averaged, renamed, or used to open new trajectories.

### Grammar 1 — Binary Computation

Binary computation is **explicit symbolic transformation**: a sequence of register states B₀, B₁, ..., B_k connected by logic and arithmetic operations L₁, ..., L_k.

$$B_0 \xrightarrow{L_1} B_1 \xrightarrow{L_2} B_2 \xrightarrow{L_3} \cdots \xrightarrow{L_k} B_k$$

**Strengths**: general, robust, programmable, scalable, deeply engineered, supports memory, branching, iteration, modularity, abstraction, error correction, symbolic layering.

**Cost**: many relations must be computed explicitly each time they are required — even when a relation is stable, repeatable, and known in advance, the trajectory must be re-derived step by step. There is no mechanism for preloading stable relational structure and activating it in parallel.

### Grammar 2 — Quantum Ket Computation

Quantum computation is reframed in P12 as **preloaded coupled trajectory evolution**:

$$|\psi_0\rangle \rightarrow U_1|\psi_0\rangle \rightarrow U_2 U_1|\psi_0\rangle \rightarrow \cdots \rightarrow |\psi_f\rangle \rightarrow \text{measurement}$$

Under the FSM reframing:
- **Superposition** = compressed symbolic trajectory directory — the ket |ψ⟩ is a preloaded summary of multiple symbolic paths in coupled form
- **Entanglement** = non-separable symbolic coupling between trajectory components
- **Unitary evolution** = coupled trajectory evolution: transformations applied to the whole coupled state simultaneously
- **Measurement** = **symbolic flattening** — the collapse from coupled trajectory to a single observed symbol

**The key clarification**: a quantum computer does not "try all answers at once." It prepares a coupled state that evolves as a whole under transformation rules; the final measured distribution is shaped by the structure of that evolution. The ket is a compressed symbolic trajectory directory, not a magical parallel computer.

**QM ∈ T_FSM** (Equation 1.27): quantum mechanics, once expressed using finite symbols, equations, kets, operators, diagrams, and interpretive language, is already inside the endogenous symbolic flow. QM does not stand outside FSM; it is one highly successful trajectory within the wider nonlinear-dynamical symbolic space. This is not a demotion — it is a clarification of its symbolic status.

### Grammar 3 — FSM / Alphonic Map Computation

P12 proposes a third grammar: **map-based computation** in which relational structure is preloaded into finite symbolic maps and activated by a physical coupling layer.

The core operation:
$$(A, B, \mathcal{F}) \xrightarrow{T} C$$

where A and B are input symbols, ℱ is the preloaded family of relations available to the transformation, and C is the output. The computation is not performed by deriving every relation step by step at runtime — the relations are already embedded in the map structure, activated in parallel by the coupling event.

**Map computation vs. binary computation:** Binary computes explicitly; map computation activates preloaded structure. Where binary must re-derive even stable, known relations every time, map computation embeds them once and activates on demand.

**Map computation vs. quantum computation:** Quantum ket computation uses Hilbert-space formalism and exponential state space; FSM map computation uses finite symbolic maps without the Hilbert-space overhead. In FSM language: quantum computation = "preloaded coupled ket trajectory"; map computation = "preloaded finite symbolic relation field."

**Relationship to optical computing:** Existing optical computing says "light performs computation." The FSM map proposal says light **creates or activates maps** — the maps hold the computational compression, and the output is a parallel symbolic relation field. The optics are the map-producing coupling layer, not the computation itself.

**Conditions for advantage:** Map computation may be powerful where preloaded relational structure can be activated and detected faster than the same relation can be derived through binary operations. It is not universally superior — it requires stable maps, calibrated sources, and high-quality decoding, and may have limited generality without rich reconfigurability.

### Error Correction and the Simulator Programme

P12 openly acknowledges that FSM map computation requires serious error correction (map instability, source drift, detector noise, decoding complexity). It proposes a software simulator as the critical next step — testing whether maps can encode useful relation families, how noise affects decoding, how compression scales, and what classes of tasks suit map computation. The concept does not require immediate physical implementation; the symbolic question can be explored computationally first.

---

## Chapter 2 — Near-Field Map Formation: Experimental Demonstration

The theoretical claims of Chapter 1 grew directly from a bench-top experiment in the near-field regime (source–detector separation shorter than the wavelength of emitted light).

**Setup:** A standard LED with its plastic lens removed, exposing a compact flat emitting surface, placed a few hundred micrometres from a CCD detector face. Single-photon-counting mode or standard imaging mode.

**Key observations:**

1. **Stable map formation:** A single stable source configuration S produces a stable, repeatable output map P: S → P
2. **Non-additive superposition:** Two source configurations S₁ and S₂ placed side-by-side produce a combined map P₁₂ that is **not** the sum P₁ + P₂. The difference map ΔP = P₁₂ − (P₁ + P₂) is itself **stable and repeatable**.
3. **Early stabilisation:** Once a few early events have arrived, the system has already selected one specific stable mode; remaining events fill in the already-determined relational field.

This non-additive structure is the physical signature of map-based computation: the combined output encodes relational information about the joint source configuration that is not recoverable from the individual maps alone.

**What the experiment establishes:** The formal structure of equation (2.5) in concrete physical form:
$$\text{trajectory pre-loading} \rightarrow \text{stable relational transformation} \rightarrow \text{parallel symbolic readout}$$

The experimenter defines stable sources once, activates them, reads the partial map, and decodes multiple pre-loaded functions in parallel. The computational compression lives in the stable map relation between source configurations — not in the optical path alone.

**What it is not:** Not a wave-interference narrative. Not a single-photon narrative. A stability narrative — the field is determined before it is fully measured.

---

## The Three Grammars Compared

| Feature | Binary | Quantum Ket | FSM Map |
|---|---|---|---|
| State carrier | Bit register | Ket (superposition in ℂ²ⁿ) | Finite symbolic map |
| Transformation | Explicit gate/logic sequence | Unitary operator sequence | Map activation by coupling layer |
| Parallelism | Sequential (or classical parallel) | Coupled trajectory evolution | Simultaneous relational readout |
| Measurement | Read register | Symbolic flattening of ket | Decode map output |
| Preloading | None (compute each time) | State preparation (costly) | Map definition (one-time cost) |
| Error handling | Mature, well-engineered | Active research frontier | Requires development |
| FSM description | Explicit trajectory derivation | Preloaded coupled ket trajectory | Preloaded finite symbolic relation field |

---

## Connections to Other Work

- **M06** (FSM Information Theory): the Functional Symbolic Trajectory framework developed in M06 is directly applied here to computation; the three computational grammars are three types of trajectory through the symbolic phase space
- **M07** (Principia Geometrica): the FSA (Finite Symbolic Admission) framework and the endogenous/exogenous measurement distinction (the Generonic Boundary as coupling layer) underlie P12's entire analysis; QM ∈ T_FSM is a direct consequence of M07's foundational position
- **M08** (Principia Geometrica II): P12 applies M08's Compressed/Unfolded Operations framework to binary computation — every binary program is an explicit unfolded trajectory; map computation preloads the compression
- **P11** (Takens and Symbols): P12 extends the trajectory-based language from language modelling to computation generally; the FSM/Alphonic map proposal is a third point in the triangle with language (P11) and computation (P12)
- **P01** (TBT/MARINA): the Takens-Based Transformer is an instance of trajectory-based computation applied to language; P12 places TBT within the wider landscape of all three computational grammars
- **ATT_38** (The Generonic Boundary): the Generonic Boundary as the coupling layer between exogenous interaction and endogenous symbol is directly invoked in P12's account of measurement in all three grammars
- **ATT_08** (Geofinitism): the measurement-first axiom applied to computation — a computation is admissible only if its symbols have finite generonic provenance
- **ATT_09** (The Ket Limit): P12 provides the broader computational context for ATT_09's FSM treatment of quantum kets; the two papers are companion pieces on quantum mechanics within FSM

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
