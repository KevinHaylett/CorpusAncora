# Summary — PE02: Geometric Transformers for Physiological Time Series Prediction

**Pensée ID:** PE02  
**Title:** *Geometric Transformers for Physiological Time Series Prediction: A Phase-Space Embedding Approach*  
**Series:** Pensée  
**Author:** Kevin R. Haylett, Manchester, UK  
**Date:** June 2026  
**Length:** 4 pages  
**Primary Colleges:** College of Attralucian Studies; College of Machine Intelligence  
**Secondary Colleges:** College of Finite Symbolic Mechanics; College of Finite Measurements  
**Primary Pillars:** P1 (Geometric Container), P3 (Dynamic Flow), P5 (Finite Reality)  
**Secondary Pillars:** P2 (Approximations / Measurements), P4 (Useful Fiction)

---

## Core Claim

Physiological signals — ECG, heart rate, SpO₂, blood pressure — are not merely data streams. They are dynamical trajectories whose future states, including critical transitions such as arrhythmia, ischaemia, or cardiac arrest, can be anticipated by modelling their geometry in reconstructed phase space. Standard transformer architectures, with dot-product attention agnostic to trajectory curvature, are the wrong instrument for this. A geometric transformer — replacing dot-product attention with distance-based weights computed over delay-embedded phase-space states — offers a principled, interpretable, and scalable alternative.

---

## Framework

### Background

**Takens' embedding** provides the theoretical foundation. For a univariate signal x(t), the delay-embedded vector at time tᵢ is:

$$v_i = [x(t_i),\, x(t_i - \tau),\, x(t_i - 2\tau),\, \ldots,\, x(t_i - (m-1)\tau)] \in \mathbb{R}^m$$

This reconstruction recovers the system's attractor geometry from a single observable, provided the embedding dimension m and delay τ are chosen appropriately. Physiological signals (ECG, HRV) exhibit nonlinear behaviour poorly captured by linear models; delay embeddings reveal hidden structure in the signal dynamics.

**Standard transformer attention** is the problem: it computes similarity via learned dot products without any representation of trajectory or curvature. It is dynamically blind. The proposal replaces this with geometric attention that reflects actual distances between embedded states in phase space.

### Model Architecture

**Input preprocessing**: raw signals (e.g., ECG in mV) are normalised per patient/session. Multivariate inputs x(t) ∈ ℝᵈ (HR + SpO₂ + BP) are supported. An identity or linear projection maps to ℝᵏ; no tokenisation is required — physiological signals are treated as continuous trajectories, not discrete tokens.

**Delay embedding layer**: the phase-space trajectory is constructed as:

$$v_i = [x_i,\, x_{i-\tau},\, \ldots,\, x_{i-(m-1)\tau}] \in \mathbb{R}^{m \cdot k}$$

**Geometric attention mechanism**: pairwise distances between embedded states replace dot products:

$$w_{ij} = \exp\!\left(-\frac{\|v_i - v_j\|^2}{\sigma^2}\right)$$

$$z_i = \sum_j w_{ij} v_j$$

The output z_i ∈ ℝ^{m·k} is a smooth, data-adaptive manifold trajectory, projected onto a hypersphere for curvature stability.

**Transformer layers**: each layer consists of GeometricAttention → feedforward network (nonlinear projection) → layer normalisation → optional smoothness loss:

$$L_{\text{smooth}} = \sum_i \|z_i - z_{i-1}\|^2$$

The smoothness loss penalises abrupt trajectory changes, enforcing the physical expectation that physiological state transitions are locally continuous.

**Decoder outputs** (depending on task): next signal state (regression); risk score (probability of cardiac event); categorical label (normal vs arrhythmic).

### Training

Proposed datasets: MIT-BIH Arrhythmia Database (ECG); MIMIC-III/IV (ICU multivariate time series). Total loss:

$$L = L_{\text{task}} + \lambda_1 L_{\text{smooth}}$$

where L_task is MSE for regression or cross-entropy for classification. Optimisation: AdamW with warm-up learning rate scheduler; batch size 32–64.

---

## Comparison with Existing Approaches

| Feature | RNN/LSTM | CNN | Geometric Transformer |
|---|---|---|---|
| Captures nonlinearity | ✓ | ✓ | ✓✓ |
| Interpretability | ✗ | ✗ | ✓ |
| Early warning capability | Limited | Static | ✓ |
| Visualisable attractors | No | No | Yes |

RNNs and CNNs can model nonlinearity, but neither provides interpretable trajectory geometry or visualisable attractors. The geometric transformer does both by operating directly in the reconstructed phase space.

---

## Use Case: Predicting Cardiac Events

A concrete instantiation for ventricular fibrillation (VF) prediction:

- **Input**: 10-second sliding ECG window sampled at 250 Hz
- **Delay embedding**: m = 3, τ = 5 (captures ~120 ms cardiac cycles)
- **Output**: risk probability for VF within the next 30 seconds
- **Deployment**: real-time monitoring with 1-second refresh; integration into ICU dashboards or wearables

The embedding parameters are chosen to capture the characteristic timescale of the cardiac cycle. Early signs of system bifurcation (loss of attractor regularity, trajectory divergence) are detectable before the clinical event.

---

## Future Work

The Pensée identifies four directions for development:

1. **Learnable delays and embedding orders**: τ and m are currently fixed hyperparameters; making them learnable would allow the model to discover the optimal phase-space geometry for each signal type and patient
2. **Multimodal extension**: EEG, respiration, blood pressure — the framework extends naturally to any combination of physiological time series once they are delay-embedded
3. **Dynamical systems metrics**: Lyapunov exponents, fractal dimension, recurrence plots — adding these as auxiliary training signals or interpretability tools would ground the model more firmly in nonlinear dynamics theory
4. **Clinical benchmarking**: validation against clinical-grade AI models (e.g., FDA-cleared arrhythmia detectors) is necessary before deployment

---

## Significance for the School

PE02 is the first application of the TBT/geometric-attention programme to medical signals. The P-series papers (P01–P04, P11) established the theoretical foundations of Takens-based transformers for language. PE02 extends the same mechanism to real-valued, multivariate physiological data — demonstrating that the framework is domain-general. Where the protein structure papers (P15–P17) applied geometric dynamics to molecular sequences, PE02 applies it to continuous clinical signals, pointing toward a future where the geometric transformer functions as a general instrument for any physical system that generates a measurable time series.

The Pensée is a programme document: the architecture is specified, the use case is concrete, the datasets are named, and the future directions are listed. It awaits compute resources and clinical collaboration for full realisation.

---

*Kevin R. Haylett — School of Geofinitism*  
*Simul Pariter.*
