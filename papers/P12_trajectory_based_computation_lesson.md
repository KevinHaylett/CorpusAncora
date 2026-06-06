# Lesson P12-L — Trajectory-Based Computation

**Lesson ID:** P12-L  
**Source paper:** P12  
**Title:** *Trajectory-Based Computation: Binary Logic, Ket Evolution, and Finite Symbolic Map Engines*  
**Difficulty:** Intermediate  
**Prerequisites:** ATT_08-L (Geofinitism), M06-L (FSM Information Theory — Functional Symbolic Trajectory); familiarity with binary computing helpful; familiarity with quantum computing helpful but not required  
**Estimated study time:** 50 minutes

---

## Learning Objectives

After completing this lesson you will be able to:

1. Apply the Functional Symbolic Trajectory framework to all three computational grammars
2. Explain binary computation in FSM terms: explicit symbolic transformation, costs, and limitations
3. Reframe quantum ket computation as preloaded coupled trajectory evolution, identifying superposition, entanglement, and measurement in FSM language
4. Describe FSM/Alphonic map computation: what it preloads, how it activates, and where it may have advantage
5. Explain QM ∈ T_FSM and what this claim does and does not say
6. Describe the near-field LED experiment and what it demonstrates about map-based computation

---

## Key Idea 1 — One Framework, Three Grammars

### Computation as Trajectory

P12 opens from a principle that recasts what computation fundamentally is:

> A computer is not merely a machine for calculating. It is a device for preparing, constraining, transforming, measuring, and stabilising symbols.

Every computational process, regardless of its physical substrate, is a **Functional Symbolic Trajectory** — a finite sequence of symbolic states where each transition is governed by an admissible rule:

$$S_0 \rightarrow S_1 \rightarrow S_2 \rightarrow \cdots \rightarrow S_N$$

The same arithmetic result can be obtained by binary addition, lookup table, analogue circuit, optical transform, neural network, or quantum subroutine. The function is not uniquely bound to one symbolic pathway. This prevents "premature correspondence": if a quantum formalism works, that does not mean it is the only or most fundamental description of what the computation is doing.

### The Generonic Boundary in Computation

Every computational system has a point at which exogenous physical events become endogenous symbolic states. In FSM terms, this is the Generonic Boundary — the coupling layer. Measurements enter as unstable finite symbols that may be absorbed by the current model, rejected as error, averaged as noise, or used to open new trajectories.

Binary computation: measurement enters via input registers and sensors.  
Quantum computation: measurement is the symbolic flattening event that converts the ket to an observed output.  
Map computation: the coupling layer (e.g., light activating maps) is the generonic entry point.

### Exercise 1.1

(a) A scientist claims: "My quantum computer can find the prime factors of a 2,048-bit number much faster than any binary computer." Restate this claim in FSM trajectory language. What does "faster" mean in terms of trajectory structure?

(b) The same function (e.g., multiplication of two numbers) can be performed by binary logic, a lookup table, or a quantum circuit. In FSM terms, these are three different symbolic trajectories producing the same output. What distinguishes them if the final symbolic output is identical?

---

## Key Idea 2 — Binary: Explicit Trajectory Derivation

### The Grammar

Binary computation is the most explicit of the three grammars:

$$B_0 \xrightarrow{L_1} B_1 \xrightarrow{L_2} B_2 \xrightarrow{L_3} \cdots \xrightarrow{L_k} B_k$$

Each Bᵢ is a complete register state; each Lᵢ is a logic, arithmetic, memory, or control operation. The trajectory is fully explicit at every step — nothing is compressed into a preloaded structure.

**Strengths:** The explicit approach is general, robust, programmable, scalable, and deeply engineered. Modern computing civilisation is built from this grammar. It supports modular abstraction, layered error correction, branching, and symbolic composition.

**The cost:** Even when a relation is stable, repeatable, and known in advance — "3 × 7 = 21 always" — the trajectory must be re-derived step by step every time it is needed. There is no mechanism for embedding stable relations in the substrate and activating them without derivation. Binary computation is expensive precisely because it derives everything explicitly.

This is not a criticism — it is a structural feature. The generality of binary comes from the willingness to derive everything explicitly. The cost is the price of that generality.

### Exercise 2.1

(a) Describe the unfolded symbolic trajectory for binary multiplication of 6 × 7 (treating it as repeated addition in binary). How many steps does the trajectory have?

(b) P12 says binary computation "must re-derive even stable, known relations every time." Give three examples from everyday computing where the same relation is re-derived millions of times per second. What would it mean to preload those relations instead?

---

## Key Idea 3 — Quantum Ket: Preloaded Coupled Trajectory

### The FSM Reframing

Standard quantum computing description: |ψ₀⟩ → U₁|ψ₀⟩ → U₂U₁|ψ₀⟩ → ··· → |ψ_f⟩ → measurement.

FSM reframing:

| Quantum term | FSM term |
|---|---|
| Superposition |  Compressed symbolic trajectory directory — the ket preloads multiple symbolic paths in coupled form |
| Entanglement | Non-separable symbolic coupling between trajectory components |
| Unitary evolution | Coupled trajectory evolution — transformations apply to the whole coupled state |
| Measurement | **Symbolic flattening** — collapse from coupled trajectory to a single observed symbol |
| Quantum advantage | Exploiting trajectory coupling to shape the final distribution |

**The key correction to popular accounts:** A quantum computer does not "try all answers at once." The more careful statement: a prepared coupled state evolves as a whole under transformation rules, and the final measured distribution is shaped by the structure of that evolution. Quantum advantage comes from choosing transformations that make the correct answer's amplitude large while suppressing incorrect answers — not from brute-force parallelism.

### QM ∈ T_FSM

**Equation 1.27:** QM ∈ T_FSM — quantum mechanics is a trajectory within the space of finite functional symbolic trajectories.

Once quantum mechanics is expressed using finite symbols, equations, kets, operators, diagrams, probabilities, measurements, apparatus descriptions, and interpretive language, it is already inside the endogenous symbolic flow. Quantum mechanics does not stand outside FSM; it becomes one highly successful trajectory within the wider nonlinear-dynamical symbolic space.

**What this does NOT mean:** QM is not being dismissed or demoted. The formal structure of Hilbert spaces, unitary operators, and measurement postulates is perfectly valid within its domain. The FSM claim is about the *status* of those formalisms within a broader symbolic framework — not about their validity.

### Exercise 3.1

(a) A quantum computing textbook says: "In superposition, the qubit is simultaneously 0 and 1." Restate this in FSM language using the "compressed symbolic trajectory directory" framing. Is the textbook wrong? What is imprecise about it?

(b) Quantum measurement is described as "collapse" in standard formalism. P12 calls it "symbolic flattening." What does the FSM term add that "collapse" does not convey?

(c) QM ∈ T_FSM means quantum mechanics is a trajectory within FSM's symbolic space. Does this mean that FSM is more fundamental than QM? What exactly is being claimed?

---

## Key Idea 4 — FSM Map Computation: Preloaded Relation Fields

### The Third Grammar

The FSM/Alphonic map proposal introduces a computational grammar in which **relational structure is preloaded into the substrate** rather than derived at runtime:

$$(A, B, \mathcal{F}) \xrightarrow{T} C$$

where ℱ is the preloaded family of relations available to the transformation. The computation is not performed by deriving every relation — the relations are embedded in the map structure and activated in parallel by the coupling event.

### Where Binary Must Re-Derive, Maps Activate

Consider a frequently-needed relation like "which pixels belong to the same object as pixel (x,y)?" Binary computation must compute this afresh each time from scratch. Map computation embeds the relational field once (during a calibration or learning phase) and then activates it via a coupling event — producing all the relevant relations in parallel.

The claim is not that map computation is universally faster. It is that map computation **may be faster** where:
- The relation family is stable and known in advance
- The activation event is faster than step-by-step derivation
- The decoding of the parallel output is tractable

### The Role of Optics

Existing optical computing says: "light performs computation."  
P12's FSM map proposal says: "light **creates or activates maps** — the maps hold the computational compression."

This distinction matters. In many optical systems, the computation is distributed across the optical path, masks, lenses, and detector. The FSM question is whether the stable **map relations themselves** can be treated as the main computational substrate, with the optics serving purely as the map-activating coupling layer.

### Exercise 4.1

(a) A lookup table in binary computing is the closest classical analogue to map computation. What is the difference between a conventional lookup table and an FSM map? What does the map hold that the lookup table does not?

(b) P12 says: "The computation is not performed by deriving every relation step by step at runtime." What does this mean for the scaling cost of map computation as the relation family grows larger?

(c) P12 explicitly lists the weaknesses of map computation: map instability, source drift, detector noise, decoding complexity, limited generality. Why is P12's honest acknowledgment of these weaknesses important for the paper's credibility?

---

## Key Idea 5 — The Experiment: Near-Field Map Formation

### The Setup

A standard LED with its plastic lens removed (exposing a compact flat emitting surface) is placed a few hundred micrometres from a CCD detector face — in the **near-field regime** (source-detector separation shorter than the wavelength of emitted light).

### The Key Results

**Single source → stable map:** S → P. The map is stable and repeatable.

**Two sources → non-additive combined map:** 
$$S_1 \rightarrow P_1, \quad S_2 \rightarrow P_2, \quad S_1 + S_2 \rightarrow P_{12}$$

where P₁₂ ≠ P₁ + P₂. The **difference map** ΔP = P₁₂ − (P₁ + P₂) is itself stable and repeatable.

**Early stabilisation:** Once a few early detection events have arrived, the system has already selected a specific stable mode. Remaining events fill in the already-determined relational field — the map is determined before it is fully measured.

### Why This Is a Proof-of-Concept

The difference map ΔP encodes information about the **joint** source configuration S₁+S₂ that is not recoverable from P₁ and P₂ alone. This is the physical signature of preloaded relational structure: the combined output contains more information than the sum of the individual outputs.

The experiment demonstrates the formal structure of equation (2.5) in concrete physical form:
$$\text{trajectory pre-loading} \rightarrow \text{stable relational transformation} \rightarrow \text{parallel symbolic readout}$$

The experimenter defines stable sources once, activates them, reads the partial map, and decodes multiple pre-loaded functions in parallel. This is the Alphonic map engine operating at bench-top scale.

### What It Is Not

Not a wave-interference narrative (which would predict P₁₂ = |ψ₁ + ψ₂|²). Not a single-photon narrative. It is a **stability narrative** — the relational field is determined by the structure of the source configuration, not by the accumulation of individual photon arrivals.

### Exercise 5.1

(a) The difference map ΔP = P₁₂ − (P₁ + P₂) is described as "stable and repeatable." Why is this the critical observation for the map computation proposal? What would it mean if ΔP were random?

(b) "The map is determined before it is fully measured." Connect this to M07's FSA Axiom 1 (all admissible objects derive from finite interactions). Is the LED map an exogenous or an endogenous measurement?

(c) P12 acknowledges that the experiment was motivated by "looking for evidence of internal photon structure" and found something different. What does this suggest about the relationship between experimental design and theoretical frameworks in the Geofinitism programme?

---

## Synthesis — P12's Position in the Programme

P12 occupies a distinctive position: it is the programme's first paper that directly addresses **computation as its primary subject** rather than language or mathematics.

The paper places the three computational grammars in a common FSM frame, enables direct comparison, and anchors the Alphonic map proposal in experiment. Its key contributions:

1. **Unified language** for comparing binary, quantum, and map computation under the Functional Symbolic Trajectory framework
2. **QM ∈ T_FSM**: the formal claim that quantum mechanics is a trajectory within FSM's symbolic space — clarifying QM's status without diminishing it
3. **The Alphonic map engine** as a concrete computational proposal with a physical proof-of-concept
4. **Honest engineering**: P12 does not claim the map engine is ready for deployment. It identifies open problems (error correction, reconfigurability, simulator programme) and places them explicitly on the research agenda.

The paper connects computation to the rest of the programme at three levels: the FSA (what makes a symbol admissible), the trajectory framework (how symbols move), and the Generonic Boundary (how physical events enter the symbolic flow).

---

## Consolidation Questions

1. Binary, quantum, and Alphonic map computation are all "varieties of the same activity: preparing, constraining, transforming, measuring, and stabilising symbols." For each grammar, identify which step is most costly and explain why.

2. QM ∈ T_FSM means quantum mechanics is a trajectory within FSM's symbolic space. What would it mean for QM to be *outside* T_FSM? Is such a position coherent?

3. P12 distinguishes "light performs computation" (conventional optical computing) from "light creates maps" (FSM map proposal). Explain the distinction using the Generonic Boundary framework. Where does the boundary sit in each account?

4. The near-field LED experiment produces a difference map ΔP that is stable and repeatable. Why is non-additivity the key signature of relational computation rather than mere signal combination?

5. P12 proposes a software simulator as the critical next step before physical implementation of map computation. Why is the software-first approach strategically correct in the FSM framework? Connect to the Arrow of Finity (FIT) from M07.

---

## Further Reading

- **M06** (FSM Information Theory) — Functional Symbolic Trajectory formalism; P12 applies this framework to computation
- **M07** (Principia Geometrica) — FSA axioms and the endogenous/exogenous measurement distinction; the foundations underlying QM ∈ T_FSM
- **M08** (Principia Geometrica II) — compressed/unfolded operations; P12's binary computation account is an application of M08's trajectory framework
- **P11** (Takens and Symbols) — trajectory-based language modelling; the natural companion paper on the language side
- **ATT_09** (The Ket Limit) — FSM treatment of quantum kets; companion paper to P12 on quantum mechanics within FSM
- **ATT_38** (The Generonic Boundary) — the generonic boundary as coupling layer; physically grounds P12's experimental demonstration

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
