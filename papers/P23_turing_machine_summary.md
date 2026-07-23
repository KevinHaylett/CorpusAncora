# P23 — The Turing Machine as Finite Symbolic Trajectory: A Geofinite Reinterpretation

**Paper ID:** P23  
**Title:** *The Turing Machine as Finite Symbolic Trajectory: A Geofinite Reinterpretation*  
**Series:** Selected Communications  
**Author:** Kevin R. Haylett  
**Location:** Manchester, UK  
**Date:** July 2026  
**Pages:** 9 (including Appendix A)  
**Primary Colleges:** College of Finite Symbolic Mechanics; College of Machine Intelligence; College of Philosophy  
**Secondary Colleges:** College of Attralucian Studies; College of Finite Measurements; College of Language Dynamics  
**Primary Pillars:** P1, P2, P3, P5  
**Secondary Pillars:** P4  
**Status:** Stable

---

## Abstract

Standard accounts treat the Turing machine as an abstract model of computation defined by a finite set of states, a tape alphabet, and a transition function acting on an idealised infinite tape. Dynamical-systems interpretations map machine configurations to points or regions in phase space and computation to orbits under iteration. This paper develops a third perspective — Geofinitism — that returns to the machine's originary character as a disciplined engine of finite symbolic trajectories. Each component (tape cell, symbol, head, state, transition, and halt) is re-understood as a measured, admissible inscription rather than an ideal mathematical object. The Turing machine appears not as a device that escapes finitude but as one that renders finite symbolic labour formally tractable.

---

## Core Claim

The Turing machine is foundational not because it transcends finitude but because it **disciplines it**. The Geofinite reading recovers Turing's original emphasis on "finite means" — his 1936–37 paper specified a finite number of internal states, a finite alphabet, and a stepwise procedure — which subsequent formalisations obscured by abstracting the tape into an infinite substrate and the machine into a function from inputs to outputs.

> "The result is not a rejection of existing interpretations but a recovery of the measured symbolic discipline that made the Turing machine intelligible in the first place."

---

## Section 1 — Introduction

Three perspectives on the Turing machine:
1. **Classical model:** abstract computation, infinite tape, function from strings to strings
2. **Dynamical-systems interpretation:** machine configurations as points in phase space; computation as orbits; connects to topological entropy, chaos, and undecidability (Moore 1991; Delvenne, Kůrka & Blondel 2006)
3. **Geofinite perspective (this paper):** asks not "What can be computed?" but "Under what measured, finite symbolic conditions can any computation be said to occur at all?"

The Geofinite route treats the Turing machine as a **finite symbolic trajectory engine** — a device whose every step is a local, admissible act of inscription, reading, replacement, and displacement.

---

## Section 2 — The Classical Turing Machine

The standard tuple:

$$M = (Q, \Sigma, \Gamma, \delta, q_0, q_\text{halt})$$

where Q is a finite state set; Σ the input alphabet; Γ the tape alphabet (Σ ⊆ Γ); δ : Q × Γ → Q × Γ × {L, R} the partial transition function; q₀ the initial state; q_halt the halting state.

The tape is an infinite array of cells. The head reads, writes, and moves left or right at each step. Computation is the iterated application of δ until a halting state is reached.

Two tacit commitments underlie this description, which Geofinitism makes explicit:
- **(a)** the tape may be extended indefinitely without cost or measurement
- **(b)** each configuration is a mathematically exact, instantaneously accessible object

---

## Section 3 — The Geofinite Framework

Key concepts of Geofinitism as applied to computation:

- **Finite symbolic trajectory:** a finite or potentially open sequence of measured configurations linked by admissible local transformations
- **Measured inscription:** any mark or absence whose presence, location, and identity are established under finite conditions of distinguishability
- **Alphonic limit (α):** the finest grain of symbolic distinction that a given container and observer can stably maintain
- **Symbolic container (C):** the bounded admissible region within which inscriptions are placed and distinguished
- **Admissibility (A):** the consensus or rule-governed conditions under which an inscription, transition, or judgement counts as valid
- **Provenance/history (H):** the traceable record of prior inscriptions and transformations that conditions present admissibility
- **Measurement uncertainty (μ):** the irreducible margin within which distinctions are made

The **Geofinite Turing machine** is the enriched structure:

$$M^g = (Q, \Sigma, \Gamma, \delta, q_0, q_\text{halt} \mid C, \alpha, H, \mu, A)$$

The classical components now operate *inside*, and are *constituted by*, the measured layer on the right. Computation becomes the concrete generation of a finite symbolic trajectory:

$$T_0 \to T_1 \to T_2 \to \cdots \to T_n$$

where each T_i records the actually inscribed tape region, head position, current state, recognised symbol, applied rule, and resulting configuration — all under the constraints of C, α, H, μ, and A.

---

## Section 4 — Re-measuring the Components

### 4.1 The Tape as Symbolic Container

The classical infinite tape is, Geofinitely, an **admissibility commitment** rather than a measured fact. Every actual tape segment is a finite container C whose cells are bounded admissible regions. The division into cells is itself a measured commitment: boundaries must be inscribed or presupposed with sufficient stability relative to α. The blank symbol is not "nothing" — it is a recognised, admissible null-inscription that preserves container structure. The infinite tape is the formal promise that further containers may be adjoined according to rule.

### 4.2 The Head as Active Measurement Site

The head is not a pointer but the local site at which the symbolic world is **read, converted into state-symbol relations, rewritten, and displaced**. It performs a measurement-conversion act: extracts a distinction from the present inscription, consults the transition rule (itself an inscribed text), produces a new inscription, and shifts position. Its movement is not merely spatial displacement but a controlled change in the **locus of measurement** — the interface between container and rule, between present inscription and admissible continuation.

### 4.3 States, Transitions, and Admissibility

The transition table δ is not a mathematical function first but an **admissible grammar of permitted symbolic transformations** — an inscribed law that must itself be held, consulted, and followed under finite conditions. Each rule application is a local act of replacement whose validity depends on the current container, alphonic limit, provenance, and consensus.

Critical distinction: **determinism of the rule does not entail finite decidability of the future trajectory.** The two are conceptually distinct — a finding with direct implications for computability theory.

### 4.4 Halting as Symbolic Closure

Halting is not merely the absence of continuation; it is the **positive recognition that a trajectory has reached an admissible closure condition**. A completed finite run T₀ → ... → Tₙ can be inspected as a measured, auditable object.

For an open trajectory, however, one cannot in general distinguish "has not yet halted" from "will never halt" because the unbounded future is not available as a completed measured inscription. **This asymmetry is constitutive rather than merely practical** — it is the core of the halting problem, reread metrologically.

---

## Section 5 — The Universal Turing Machine as Textual Containment

The universal machine U simulates any machine M when supplied with a description of M and its input. Classically this demonstrates the universality of computation. Geofinitism emphasises the **textual character** of the achievement:

A machine description is itself a finite symbolic inscription that can be placed inside the container of another machine and treated as data. This is an early and explicit instance of **meta-container structure** — text acting upon text, formal system encoded inside formal system.

Self-reference and recursion are rendered tractable precisely because they remain within finite symbolic containment rather than escaping into pure abstraction. The universal machine exemplifies the Geofinite thesis: **computation is disciplined symbolic labour operating across layered containers**.

The layers are functional, not ontological — the simulator reading the description of the simulated is a chosen layering, not a metaphysical ascent. This connects directly to the "text-within-text" argument of P21.

---

## Section 6 — The Halting Problem Revisited

Turing's undecidability result is reread not solely as a theorem about functions or predicates but as a statement about the **boundary between completed and open symbolic trajectories**.

A completed halting trajectory is a finite, inspectable inscription. An open trajectory presents an ongoing demand for continuation whose termination (or non-termination) cannot be read off from any finite prefix under the machine's own admissibility conditions.

The halting problem is therefore simultaneously **logical and metrological**: it concerns what can be measured and judged as closed within finite symbolic resources. The classical result is preserved; its force is located in the asymmetry between finished and unfinished measured paths rather than in purely abstract undecidability.

> "The halting problem is the boundary of the measurable: you cannot measure the future because the future is not a physical extent available at the current epoch."

---

## Section 7 — Bridges to Gödel, Lorenz, and Wolfram

The Turing machine as Geofinite hinge connecting three traditions:

**Gödel:** Incompleteness theorems expose the wound of self-reference within formal systems. The universal machine renders that self-reference *mechanically operable* inside finite symbolic containers — the same move, differently expressed.

**Lorenz:** Deterministic unpredictability arising from finite measurement precision finds a symbolic counterpart: even with exact rules, the future of a trajectory may not be finitely decidable from its description. Sensitivity to initial conditions is a trajectory phenomenon, not just a continuous-system curiosity.

**Wolfram:** Computational irreducibility and the Ruliad (the totality of possible computations) are here grounded in the prior discipline of **finite symbolic rewriting**. Irreducibility appears as the practical necessity of following the measured trajectory rather than shortcutting it — the Geofinite name for what Wolfram identifies empirically.

In each case Geofinitism supplies the missing measured layer: the conditions under which distinctions, rules, trajectories, and closures can be admitted at all.

---

## Section 8 — Conclusion

Five re-descriptions that recover the originary power of Turing's construction:

1. **The tape** = a symbolic container whose extension is an admissibility promise
2. **The head** = an active site of measurement and replacement
3. **Halting** = symbolic closure
4. **The universal machine** = text interpreting text inside layered containers
5. **The halting problem** = the boundary between completed and open trajectories

Future work indicated: extending the framework to cellular automata, lambda calculus, and quantum computation; developing a formal metrology of symbolic containers; exploring implications for verification, proof, and the limits of formal systems.

---

## Appendix A — For Those New to Geofinitism

The paper includes an 8-section appendix providing a standalone introduction to Geofinitism for readers without prior familiarity. This makes P23 one of the most accessible entry points into the framework alongside ATT_00. Key subsections:

**A.1 The Ground Is the Current Limit of First-Order Measurement** — "We have a ruler." The Alphonic Limit is historical, technological, and cognitive — not a philosophical absolute. No ground beneath this ground; no appeal to transcendental or Platonic foundation.

**A.2 A Symbol Is Finite, Real, and Measurable** — A symbol is a physical mark with finite spatial, temporal, or energetic extent. Every symbol is: finite (bounded region), real (measurable physical extent), measurable (distinguishable at the current alphonic limit). No infinite symbols; no ideal symbols.

**A.3 Models Are Second-Order Constructions** — Below the measurement limit, we model rather than measure. Electrons, quarks, continua, infinitesimals are second-order constructions — endogenous creations generated within the measured container. Not false; enormously useful; but not the ground.

**A.4 The Layering Is Functional and Chosen** — Container hierarchy is functional, not fixed. Physics can be a container for mathematics, or mathematics for physics. The layers are tools for organising trajectories, not an ontological order.

**A.5 Truth Is Admissible Closure** — Truth is not correspondence to a transcendent reality. A theorem is true if it can be generated as a finite symbolic trajectory from admitted axioms under admitted rules within a given container at a given alphonic limit. Not relativism: *measured realism*.

**A.6 Computation Is a Finite Symbolic Trajectory** — A computation is a finite sequence of measured acts. The Turing machine is not a model of computation; it is *computation itself made visible*.

**A.7 Geofinitism Is Contingent, Not Absolute** — Geofinitism claims only that any foundation must begin with actual, current conditions of measurement. If instruments improve, the ground shifts. This is a strength: alignment with the historical, technological, and scientific character of human knowledge.

**A.8 Reading This Paper** — Summary glossary: tape = admissibility claim; head = measurement site; universal machine = text on text; halting problem = open/closed boundary; Gödel/Lorenz/Wolfram = grounded, not applied.

**Connection to Technical Work:** FSET provides the formal licence for the FST reconstruction of discrete symbolic dynamics. The TBT/MARINA architecture treats language generation as trajectory reconstruction. In this view, a Turing machine and a language model are both engines for generating functional symbolic trajectories.

---

## Significance for the School

P23 is the first paper in the School to address the Turing machine directly. It completes an important circuit: Geofinitism began with criticism of ideal symbolic models (P01–P05 on language, P08 on autoregression, P12 on trajectory-based computation), moved through formal apparatus (P04: FSET, P11: Takens for symbols), and now reaches back to the foundational model of computation itself.

The Appendix A makes P23 a companion document to ATT_00 as an entry point to the framework. Together they provide two routes in: ATT_00 as orientation for the corpus, P23 as orientation for Geofinitism via the familiar territory of the Turing machine.

The paper's three-way contrast (classical / dynamical-systems / Geofinite) is methodologically significant: it positions Geofinitism not as an alternative within the existing landscape but as a prior grounding layer for *both* classical and dynamical readings.

**Key connections:** P04 (FSET — the formal licence P23 invokes in the appendix); P08 (Autoregression Is Not Takens — shared argument that rule-following ≠ trajectory prediction); P11 (Takens Applies to Symbol Sequences — FST foundation); P12 (Trajectory-Based Computation — directly parallel); P18 (From Formal Logic to Functional Symbolic Trajectories — Hilbert/Brouwer/Russell and the same classical-to-Geofinite movement); P19 (P vs NP and the Missing Axiom of Finite Measurement — computability meets measurement); P21 (Text Within Text — the meta-container and text-on-text argument P23's Section 5 extends); ATT_00 (Appendix A companion); ATT_08 (Measurement-First Philosophy — foundational alignment); ATT_28 (Commitment and Admissibility — admissibility doctrine); ATT_38 (The Generonic Boundary — inscription as emergence); ATT_80 (Semantic Boundary Markers — boundary between open and closed trajectories); M02 (FSET Monograph — formal underpinning)

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
