# PAPER_IMPLEMENTATION_SPEC.md

**Paper:** "EEG-Based Early Detection of Autism Spectrum Disorder Using Two-Stage Feature
Extraction and a Spatio-Temporal Graph Transformer" — J. Beno Ranjana, R. Muthukkumar
(National Engineering College, Kovilpatti, India).

**Source of extraction:** Full text of the 33-page PDF provided in the project
(`ASD_Detection.pdf`), read page-by-page including embedded figures (Figures 1–7).

**Legend for every item below:**
- **A** = Explicitly specified by the paper (number/name/formula given)
- **B** = Can be directly derived from the paper (follows unambiguously from what's stated)
- **C** = Not specified by the paper — requires an implementation assumption
- **D** = Ambiguous or internally inconsistent in the paper

---

## 1. Dataset (Section 3.1, Table 2)

| Item | Value | Category |
|---|---|---|
| Dataset name | National Database for Autism Research (NDAR) | A |
| Dataset URL given in paper | `https://ndar.nih.gov/edit_collection.html?id=2021` | A |
| Total subjects | 189 | A |
| ASD subjects | 97 | A |
| Non-ASD (neurotypical) subjects | 92 | A |
| Sex balance | "approximately equally divided between males and females" (no exact counts) | A (qualitative only) |
| EEG device | 128-channel Electrical Geodesics HydroCel Geodesic Sensor Net (HCGSN) + NetStation software | A |
| Number of channels | 128 | A |
| File format | MAT | A |
| Anti-aliasing filter | 4 kHz | A |
| Sampling rate | 500 Hz | A |
| Recording state | Resting state: 60 s looking calmly at a screen, then 30 s eyes closed | A |
| Impedance monitoring | Performed throughout, no numeric threshold given | A (qualitative only) |
| Trial/window length used for classification | **Not stated** | C |
| Train/validation/test split ratios or protocol (k-fold, subject-wise, etc.) | **Not stated anywhere in the paper** | C |
| Exact per-class train/test counts used to produce the Table 8 confusion matrix (95+2 ASD, 12+80 non-ASD = 189 total) | Confusion matrix totals equal the *full* dataset (97 ASD, 92 non-ASD) | D — see §9 "Ambiguities", item D1 |

**Data availability note (paper, §"Data availability"):** the dataset is stated as
confidential / available only on reasonable request to the corresponding author. **We will
not download, simulate as real, or fabricate this dataset.** Phase 0 uses only synthetic
placeholder tensors for structural validation (see §8).

---

## 2. Preprocessing (Section 3.2)

### 2.1 Butterworth band-pass filter (Section 3.2.1, Eqs. 1–3)

| Parameter | Value | Category |
|---|---|---|
| Filter family | Butterworth | A |
| Sampling frequency `fs` | 500 Hz | A |
| Cutoff frequencies `fc` | 0.1 Hz – 50 Hz (band-pass) | A |
| Filter order `n` | 4 | A |
| Normalization | `f_norm = fc / f_Nyquist`, `f_Nyquist = fs/2` | A |
| Filter application domain | digital (paper states "can be implemented in both analog and digital forms"; the equations given, Eq. 3, are a digital IIR difference equation) | B |
| Filtering direction (causal vs. zero-phase / forward-backward `filtfilt`) | **Not stated** | C |
| Filter applied per-channel or across all channels jointly | Implied per-channel (each channel is an independent time series) | B |

The paper's Eq. 3 is a generic IIR difference equation form (`y[n] = (1/i[0]) * (Σ j[k]x[n-k]
− Σ i[k]y[n-k])`), i.e. the standard Butterworth digital filter transfer function. This maps
directly onto `scipy.signal.butter` + `scipy.signal.lfilter`/`filtfilt`.

### 2.2 Savitzky–Golay (SG) smoothing (Section 3.2.2, Eqs. 4–5)

| Parameter | Value | Category |
|---|---|---|
| Filter type | Savitzky-Golay, linear least-squares polynomial fit | A |
| Applied as a **cascade of two SG filters of the same order** (stage 2 uses stage-1 output as input) | A | A |
| Polynomial degree | **Not numerically specified** | C |
| Window length `v` (number of points in the convolution window) | **Not numerically specified** (only appears symbolically in Eq. 4–5) | C |
| Whether cascade stages use identical or different `(degree, window)` settings | "same order" implies identical settings for both stages | B |

---

## 3. Data Augmentation (Section 3.3, Algorithm block)

| Parameter | Value | Category |
|---|---|---|
| Technique | Additive/multiplicative Gaussian-noise-based augmentation | A |
| Applied per channel | Yes — `σ = std(E_i^(ch))` computed per channel of each subject's signal | A |
| Noise parameters | `σ1 = σ + 0.1`, `σ2 = σ − 0.1` | A |
| PDF used | Gaussian: `P_G(R,σ) = (1/(σ√(2π))) · exp(−R²/(2σ²))` | A |
| Random variable `R` | `R = {R1, …, Rm}`, drawn per augmentation iteration | A |
| Output signals per input signal | `AS_1..m(ch) = E_i(ch) * P_G(R_m, σ1)`; `AS_{m+1}..{m*2}(ch) = E_i(ch) * P_G(R_m, σ2)` → `m*2` augmented signals total per original recording | A (formula) |
| Number of augmentation iterations `m` | **Not numerically specified** | C |
| Distribution from which `R_m` is drawn (paper never states this — it appears as a generic "random variable") | **Not specified** (only the PDF used to *weight* the signal is specified, not the sampling distribution of `R`) | C / D — see §9 item D2 |

**D2 — Mathematical ambiguity flagged:** the algorithm defines the augmented signal as the
*original signal multiplied element-wise by the Gaussian PDF value* (`E_i(ch) * P_G(R_m,
σ)`), not the more conventional "signal + Gaussian noise". This is unusual: multiplying by a
PDF (whose values are ≤ `1/(σ√2π)`, always positive) rescales amplitude rather than
perturbing it in a symmetric, mean-preserving way, and does not match how Gaussian-noise
EEG augmentation is normally described in the cited references [44, 45]. We treat the
formula exactly as written for a literal-spec implementation, but flag this as a likely
transcription issue in the source paper, and will implement it as a configurable, labeled
"paper-literal" augmentation mode, separate from a "conventional additive-Gaussian-noise"
mode used as an alternative in ablations (both clearly labeled as to which is which).

---

## 4. Two-Stage Feature Extraction (Section 3.4)

### 4.1 Stage 1 — Node-based (per-electrode) features (Table 3, Table 5)

**Table 3 defines 10 time-domain features with exact formulas:**

| # | Feature | Formula (as given) | Category |
|---|---|---|---|
| 1 | Mean | `μx = (1/N) Σ x(t)` | A |
| 2 | Variance | `σx² = (1/N) Σ (x(t) − μ)²` (paper's Eq. is written without the square explicitly, but this is the standard variance and is treated as B) | A/B |
| 3 | RMS | `RMS = sqrt((1/N) Σ x(t)²)` | A |
| 4 | Hjorth Activity | `H_activity = σx²` | A |
| 5 | Hjorth Mobility | `H_mobility = sqrt(σv² / σx²)`, `v(n) = dx(n)/dn` | A |
| 6 | Hjorth Complexity | `H_complexity = sqrt(σa²/σv²) / sqrt(σv²/σx²)`, `a(n) = d²x(n)/dn²` | A |
| 7 | Skewness | `S = (1/N) Σ (x(n) − μ)³ / σx³` | A |
| 8 | Kurtosis | `K = (1/N) Σ (x(n) − μ)⁴ / σx⁴` | A |
| 9 | Peak-to-Peak Amplitude | `A_ptp = max(x) − min(x)` | A |
| 10 | Signal Entropy | `SE = −Σ p_n log(p_n)` | A |

**Explicit statement:** "10 distinct time-domain features are extracted from each electrode,
resulting in a total of 1280 features per time sample (128 channels × 10 features)." This
confirms the paper intends 128 channels/nodes at the feature-extraction stage. — **A**

**Frequency-domain features (prose, Eqs. 6–9, Table 4):**

| Feature | Formula | Category |
|---|---|---|
| Instantaneous power | `P(n) = A(n)²` | A |
| Relative Spectral Power (RSP) | `Ψr = Ωb / Ωr` | A |
| Theta-to-Alpha Ratio (TAR) | `Λθα = Ωθ / Ωα` | A |
| Theta-to-Beta Ratio (TBR) | `Λθβ = Ωθ / Ωβ` | A |
| Frequency bands | Delta 0.5–4 Hz, Theta 4–8 Hz, Alpha 8–13 Hz, Beta 13–30 Hz, Gamma >30 Hz (Table 4) | A |
| PSD estimation method (Welch, multitaper, FFT periodogram, etc.) | **Not specified** | C |
| Gamma band upper bound (paper says ">30", no upper cutoff — but the Butterworth pre-filter already caps the signal at 50 Hz) | Effective gamma band ≈ 30–50 Hz once combined with preprocessing | B |

### 4.2 Table 5 — "Description of Three Feature Sets" (node feature sets used experimentally)

| Set | Type | Feature names listed in Table 5 |
|---|---|---|
| A1 | Time-domain | Mean, Variance, RMS, **Hjorth Parameters** (grouped as one item) |
| A2 | Frequency-domain | Instantaneous Power, RSP, TAR, TBR |
| A3 | Time-frequency domain | Mean, Variance, **Power Value**, **Absolute Mean**, **Ratio of Absolute Mean** |

**D3 — Major inconsistency flagged:** Table 3 rigorously defines **10** time-domain features
with formulas (including Skewness, Kurtosis, Peak-to-Peak, Entropy), but Table 5's "A1"
column only *names* 4 items (Mean, Variance, RMS, "Hjorth Parameters" — itself an umbrella
for 3 sub-features: Activity/Mobility/Complexity). It's unclear whether A1 =
{Mean, Variance, RMS, Hjorth×3} = 6 features, or whether A1 is meant to include all 10
Table-3 features and Table 5 is just an abbreviated list. Additionally, A3's listed features
— "Power Value", "Absolute Mean", "Ratio of Absolute Mean" — are **never mathematically
defined anywhere in the paper** (no equation, no reference to Eq. numbers). This is a
genuine specification gap, not just a naming issue. **Category D (inconsistent) + C
(undefined quantities requiring an implementation assumption).**

*Proposed resolution (to be confirmed/adjusted in Phase 1, not decided now):* implement
A1 as the full Table-3 10-feature set (most defensible reading — Table 3 is the canonical
definition table), implement A2 exactly as the 4 named+defined frequency features, and
implement A3 as a documented **concatenation of A1 ∪ A2** (a common way papers describe
"time-frequency" feature sets) while explicitly labeling "Power Value" ≈ instantaneous
power, "Absolute Mean" ≈ mean of `|x|`, and "Ratio of Absolute Mean" ≈ a band-ratio analogous
to TAR/TBR but on absolute-mean rather than power — each such mapping will be labeled
in code and docs as an **assumption**, not a paper fact.

### 4.3 Stage 2 — Edge-based / connectivity features (Section 3.4, Eqs. 10–15)

| Estimator | Formula | Category |
|---|---|---|
| Partial Correlation (PC) | `c_{i,j\|k} = (c_ij − c_jk·c_ik) / sqrt((1−c_ik²)(1−c_jk²))`; `PC(i,j) = min_k(c_{i,j\|k})`, ∀k≠i,j | A |
| Phase Lag Index (PLI) | `PLI = |sign(Δφ_t)|`, `Δφ_t = φ1(t) − φ2(t)` | A |
| Phase extraction method | "wavelet transform or Hilbert transform" — **either explicitly allowed, method choice not fixed** | A (both permitted) / C (must pick one) |
| PLI temporal aggregation (single scalar per channel pair from an instantaneous-in-time quantity) | **Not specified** — standard practice is `PLI_ij = |mean_t(sign(Δφ_t))|`; the paper's Eq. 12 gives only the instantaneous form | C — see §9 item D4 |
| Granger Causality (GC) | AR: `y_t = Σ a_i y_{t-i} + ε_t`; BAR: `y_t = Σ a_i y_{t-i} + Σ b_i x_{t-i} + ε'_t`; `GC_{y→x} = ln(var(ε_t)/var(ε'_t))` | A |
| AR/BAR model order `p` | **Not numerically specified** | C |
| Directionality of GC used for the (symmetric?) adjacency matrix | GC as defined is directional (`y→x` ≠ `x→y`); the paper's adjacency matrix `A ∈ R^{v×v}` does not state whether GC-based adjacency is kept directional (asymmetric matrix, standard GAT can support this) or symmetrized | C/D |

---

## 5. Graph Construction & Thresholding (Sections 3.4.1 and 4.2, Eqs. 23–24)

| Item | Value | Category |
|---|---|---|
| Graph definition | `G = {V, E, A}`; `V` = electrodes; `A ∈ R^{v×v}` | A |
| Number of vertices `v` | Stated as 128 in Section 3.1/3.4 body text, **but Figure 3's legend explicitly states "V – Vertices (32 electrodes)"** | **D5 — direct numeric contradiction, see §9** |
| Thresholding rule (binary adjacency) | `w_ij = w_ij if w_ij ≥ Threshold else 0` (Eq. 23 — note: this equation number **duplicates** the FFN equation also numbered "(23)" in Section 3.5; a paper-numbering defect, not our error) | A (rule) / D (numbering clash, cosmetic) |
| Threshold definition, version 1 (Section 3.4.1 prose) | "mean of all connection strengths" — i.e., per single-estimator adjacency matrix, threshold = mean of *that matrix's* entries | A |
| Threshold definition, version 2 (Section 4.2, Eq. 24) | `T_θ = (1/M) Σ_{k=1..M} ζ_k`, where `ζ_k` = "kth connectivity estimator value" and `M` = "total number of connectivity estimators" (i.e., M=3: PC, PLI, GC) — this describes averaging **across estimators per edge**, not averaging entries within one matrix | A |
| Reconciling versions 1 and 2 | **Not reconciled by the paper** — these are two different operations (within-matrix mean vs. across-estimator mean per edge) | **D6 — see §9** |
| Weighted vs. binary adjacency used as GAT input | Both described: a weighted matrix is built, then a "corresponding binary adjacency matrix is derived" — unclear which one (weighted, thresholded-weighted, or pure binary 0/1) is what actually feeds the GAT | C/D |

---

## 6. STGT Architecture (Section 3.5, Figure 4, Eqs. 16–23)

### 6.1 Graph Attention Network (GAT) — spatial stage

| Item | Value | Category |
|---|---|---|
| Attention coefficient (raw) | `E_ij = a(Wx_i, Wx_j)` (Eq. 16), realized via `E_ij = LeakyReLU(a^T[Wx_i \|\| Wx_j])` (Eq. 17) | A |
| Normalized attention | `α_ij = softmax_j(E_ij) = exp(E_ij) / Σ_{l∈V} exp(E_il)` (Eq. 18) | A |
| Node update | `h_i' = σ(Σ_{j∈V} α_ij · W x_j)` (Eq. 19) | A |
| Number of stacked GAT layers | **2**, per Figure 4's left-hand flow diagram (two boxes labeled "Graph Attention Network" in series before the Transformer) | A (from figure) |
| Number of GAT attention heads | **Not specified** (paper only says "multiple self-attention mechanisms are employed"; the "4 attention heads" figure in Section 4.3 is stated for the **Transformer**, not explicitly for GAT) | C — see §9 item D7 |
| GAT hidden dimension | **Not specified** | C |
| Multi-head aggregation in GAT (concat vs. average across heads) | **Not specified** | C |
| Activation `σ` in Eq. 19 | Not explicitly named (ELU is standard in GAT literature; paper doesn't name it) | C |

### 6.2 Linear projection between GAT and Transformer

| Item | Value | Category |
|---|---|---|
| "Linear projection enhancement technique" between GAT output and Transformer input | Mentioned in prose, no dimensions/formula given | C |

### 6.3 Transformer — temporal stage (Eqs. 20–23, Figure 4)

| Item | Value | Category |
|---|---|---|
| Architecture style | Encoder-style multi-head self-attention + FFN, with residual connections and layer normalization (paper text calls it "encoder-decoder" once, but Figure 4 depicts an encoder-only stack: LayerNorm → Multi-head Attention → Add → LayerNorm → FeedForward → Add) | A (figure) / D (text says "encoder–decoder", figure shows encoder-only — flag D8) |
| Q, K, V shapes | `Q,K,V ∈ R^{d_r × d}` (Eq. 20–22) | A (symbolic only, no numeric `d_r`/`d`) |
| FFN | `FFN(x) = ReLU(0, xW1+b1)W2+b2` (Eq. 23) | A |
| Number of Transformer layers (ablation-tested) | 2, 3, 4, 5 (Table 7); **3 is the value used in the paper's final/best reported configuration** | A |
| Number of attention heads | 4 (default, used in most ablation rows); one row tests 8 heads | A |
| Hidden dimension | 64 (default); one row tests 128 (paired with 8 heads) | A |
| Final classification head | Reshape → Linear layer → Softmax (2-class output: Autism / Non-autism) | A |
| Loss function | Cross-entropy | A |
| Optimizer | Adam | A |
| Learning rate, weight decay, batch size, epochs, early stopping | **Not specified anywhere** | C |

---

## 7. Reported Results & Evaluation Protocol (Section 4)

### 7.1 Metrics (Eqs. 25–30)

Sensitivity, Specificity, Accuracy, Precision, Recall, F1-score — all explicitly defined
with standard formulas. — **A**

### 7.2 Confusion matrix of the final model (Table 8)

| | Predicted Autism | Predicted Non-Autism |
|---|---|---|
| **Actual Autism** | TP = 95 | FN = 2 |
| **Actual Non-Autism** | FP = 12 | TN = 80 |

Sensitivity = 97.94%, Specificity = 86.96%, Accuracy = 92.59%. — **A**

**D1 (referenced above):** TP+FN = 97 = total ASD subjects in the whole dataset; TN+FP = 92 =
total non-ASD subjects in the whole dataset. This strongly suggests Table 8's confusion
matrix was computed either (a) on the **entire dataset** with no held-out test set (i.e. reported
on training data / no train-test split), or (b) coincidentally a test set the same size as the
full class counts. **The paper never states its train/test split protocol**, so we cannot
determine which. This is an important reproducibility gap that materially affects how
Phase 2+ experiments should be designed (we will need to make and clearly document an
assumption, e.g. stratified k-fold or hold-out split, and will not claim to match Table 8
exactly).

### 7.3 Table 6 — Accuracy by connectivity estimator × feature set (STGT vs. two ablated architectures: "Spatial Graph Transformer" and "Temporal Graph Transformer")

Full values transcribed in the spec appendix (§10). The "Average(%)" column in Table 6 is
numerically consistent with a simple arithmetic mean of the three estimator-accuracy columns
in the same row (verified for A3/STGT: (91.62+94.54+91.61)/3 = 92.59 ✓). — **A/B**

### 7.4 Table 7 — Ablation study

Full values transcribed in §10. **D9 — flagged inconsistency:** the best single ablation
row is `A3 + PLI, 3 layers, 4 heads, 64-dim → 94.44% ± 0.65`, which is *higher* than the
paper's ultimately-reported "proposed model" result of 92.59% (`A3 + PLI+PC+GC combined, 3
layers, 4 heads, 64-dim`). The paper does not explain why the lower-scoring combined-estimator
configuration, rather than the higher-scoring single-estimator (PLI-only) configuration, was
selected as the final/headline model. Also unexplained: whether "PLI+PC+GC" in Table 7's
last row denotes a genuinely jointly-trained model on a fused/averaged adjacency matrix, or
is simply restating the Table-6 arithmetic average of the three separate single-estimator
runs (both produce 92.59). We flag this rather than silently assuming one interpretation.

### 7.5 Table 9 — Comparison with existing methods

Baselines and their accuracy/sensitivity/specificity/F1, as reported by the paper's authors
(not independently re-implemented by us): SNRA (93.56%), GCN (95%), Deep GCN (92.73%),
Visible GCN (93.7%), proposed STGT (92.59%). — **A** (as claims made in the paper; we treat
these as reported by the source paper, not something we will reproduce in Phase 0/1).

### 7.6 ROC / PR curves (Figure 7)

AUC = 0.97; average precision = 0.97. — **A**

---

## 8. End-to-End Pipeline Diagram

```
Raw EEG (.mat, 128 channels, 500 Hz, resting-state recording)
  │
  ▼
Preprocessing
  ├─ Butterworth band-pass filter (0.1–50 Hz, order 4)
  └─ Savitzky–Golay smoothing (cascade, 2 stages, order/window = ASSUMPTION)
  │
  ▼
Data Augmentation (Gaussian-noise-based, paper-literal formula; m = ASSUMPTION)
  │
  ▼
┌───────────────────────────── Two-Stage Feature Extraction ─────────────────────────────┐
│                                                                                          │
│  Stage 1: Node-based (per-electrode) features         Stage 2: Edge-based connectivity  │
│  ─ Time-domain (Table 3, 10 features)                  ─ Partial Correlation (PC)       │
│  ─ Frequency-domain (Table 4/Eqs. 6-9, 4 features)      ─ Phase Lag Index (PLI)          │
│  ─ Feature sets A1 / A2 / A3 (Table 5)                  ─ Granger Causality (GC)         │
│         │                                                        │                      │
│         ▼                                                        ▼                      │
│   Node feature matrix  F  (per subject/window)         Weighted adjacency per estimator │
│                                                                    │                      │
│                                                                    ▼                      │
│                                                          Thresholding (mean-based,        │
│                                                          exact rule = AMBIGUOUS, see D6)  │
│                                                                    │                      │
│                                                                    ▼                      │
│                                                          Binary/weighted adjacency A       │
└──────────────────────────────────────┬──────────────────────────────────────────────────┘
                                        ▼
                          Graph construction: G = {V, E, A}
                                        │
                                        ▼
                    Graph Attention Network (GAT) — 2 stacked layers (Fig. 4)
                    heads/hidden-dim = ASSUMPTION (see D7)
                                        │
                                        ▼
                        Linear projection (dims = ASSUMPTION)
                                        │
                                        ▼
        Transformer encoder (multi-head self-attn + FFN, LayerNorm, residuals)
        layers ∈ {2,3,4,5} (paper's final = 3), heads = 4 (default), hidden = 64 (default)
                                        │
                                        ▼
                    Reshape → Fully-Connected (Linear) layer
                                        │
                                        ▼
                              Softmax (2-class)
                                        │
                                        ▼
                        Classification: ASD vs. Non-ASD
                                        │
                                        ▼
     Evaluation: Confusion Matrix, Sensitivity, Specificity, Accuracy, Precision,
                 Recall, F1, ROC-AUC, PR-AUC
```

---

## 9. Consolidated List of Ambiguities / Inconsistencies (Category D)

| ID | Description | Impact |
|---|---|---|
| D1 | Table 8 confusion-matrix totals (97 ASD, 92 non-ASD) equal the *entire* dataset's class counts; no train/test split protocol is described. | Cannot reproduce the exact reported numbers without inventing a split methodology. |
| D2 | Augmentation formula multiplies signal by a Gaussian PDF value rather than adding noise — mathematically unusual vs. cited literature. | Two augmentation modes will be implemented and clearly labeled: "paper-literal" and "conventional additive". |
| D3 | Table 5's A1 (4 named items) doesn't match Table 3's 10 rigorously-defined features; A3's three feature names ("Power Value", "Absolute Mean", "Ratio of Absolute Mean") are never mathematically defined anywhere in the paper. | Feature-set composition for A1/A3 requires a documented implementation assumption. |
| D4 | PLI formula (Eq. 12) is instantaneous (per time-sample); no aggregation rule to a single scalar per channel-pair is given. | Standard PLI temporal-averaging formula will be used as an assumption, clearly labeled. |
| D5 | Figure 3's legend states "V – Vertices (32 electrodes)", directly contradicting the 128-channel acquisition and the "128 channels × 10 features = 1280" statement elsewhere in the same paper. | Core graph size (v=32 vs v=128) is unresolved; affects every downstream tensor shape. Must be an explicit, prominent configuration choice, not silently picked. |
| D6 | Two different threshold definitions: (a) §3.4.1 — mean of entries *within* one estimator's adjacency matrix; (b) §4.2 Eq. 24 — mean *across the three estimators* per edge. | Determines what the final adjacency matrix actually represents; requires a documented implementation choice. |
| D7 | Number of GAT attention heads and hidden dimension are never stated (only Transformer heads/dim are given in the ablation table). | Requires an implementation assumption. |
| D8 | Prose calls the temporal module an "encoder–decoder transformer", but Figure 4 depicts an encoder-only stack (no decoder, no cross-attention, no autoregressive generation). | We will implement encoder-only, consistent with the figure and with a 2-class classification output (a decoder would not make sense here), and note the textual discrepancy. |
| D9 | The ablation study's best single result (A3+PLI, 3 layers → 94.44%) exceeds the paper's headline "proposed model" result (A3+PLI+PC+GC combined → 92.59%), with no explanation of why the lower-scoring configuration was chosen as final. Also unclear whether "PLI+PC+GC" is a genuinely fused-graph model or an arithmetic average of three independent runs. | Does not block implementation, but must be documented so Phase 2+ experiments don't silently assume the combined estimator is "better". |
| D10 (minor) | Equation numbering: both the FFN formula (Section 3.5) and the thresholding rule (Section 4.2) are labeled "(23)" in the source paper. | Cosmetic; noted for traceability only, no implementation impact. |
| D11 | Electrode names used for anatomical/interpretability discussion (FP1, FP2, AFz, F3, F4, F7, F8, FC1, FC2, FC5, FC6, Fz, FCz, T7, T8, TP9, TP10, FT9, FT10, P3, P4, P7, P8, O1, O2) follow 10-20-style nomenclature, but the acquisition device is a 128-channel HydroCel Geodesic Sensor Net, whose native channel labels are numeric (E1–E128). No channel-name-to-index montage/mapping is provided. | A named-electrode ↔ GSN-channel-index mapping will be required before any of the anatomical-region analysis in §4.2/§4.5 of the paper can be reproduced; flagged as missing (Category C), not fabricated. |

---

## 10. All Numerical Parameters Explicitly Given In The Paper (Consolidated)

- Subjects: 189 total (97 ASD, 92 non-ASD)
- Channels: 128 (device), but see D5 for the Figure-3 32-electrode contradiction
- Sampling rate: 500 Hz
- Anti-aliasing filter: 4 kHz
- Resting-state protocol: 60 s eyes-open + 30 s eyes-closed
- Butterworth filter: band-pass 0.1–50 Hz, order 4
- Gaussian augmentation: σ1 = σ+0.1, σ2 = σ−0.1
- Time-domain node features: 10 (Table 3); "1280 = 128 × 10" stated explicitly
- Frequency bands: Delta 0.5–4 Hz, Theta 4–8 Hz, Alpha 8–13 Hz, Beta 13–30 Hz, Gamma >30 Hz
- Connectivity estimators: 3 (PC, PLI, GC)
- Transformer layers tested: {2, 3, 4, 5}; final model uses 3
- Transformer attention heads: 4 (default), 8 (one ablation row)
- Transformer hidden dimension: 64 (default), 128 (one ablation row, paired w/ 8 heads)
- GAT stacked layers: 2 (from Figure 4, not from prose)
- Final confusion matrix: TP=95, FN=2, FP=12, TN=80
- Reported metrics: Accuracy 92.59%, Sensitivity 97.94%, Specificity 86.96%, F1 93.16%,
  ROC-AUC 0.97, PR average precision 0.97
- Full Table 6 (Accuracy % by feature set × connectivity estimator × 3 model variants:
  STGT / Spatial Graph Transformer / Temporal Graph Transformer) and full Table 7 (ablation
  over transformer layers/heads/hidden-dim) are reproduced verbatim in
  `docs/paper_tables/table6_table7.md` for traceability.
- Baseline comparison accuracies (Table 9, as claimed by the paper, not reproduced by us):
  SNRA 93.56%, GCN 95%, Deep GCN 92.73%, Visible GCN 93.7%.

---

## 11. Missing Hyperparameters Requiring Future Decisions (Category C — NOT chosen yet)

1. EEG epoch/window length and stride used for classification (seconds, overlap).
2. Savitzky–Golay polynomial degree and window length `v` (both cascade stages).
3. Number of Gaussian-noise augmentation iterations `m`.
4. Distribution used to draw the random variable `R_m` in the augmentation algorithm.
5. PSD estimation method (Welch/multitaper/periodogram) and its window/segment parameters.
6. Phase-extraction method for PLI (Hilbert transform vs. wavelet transform) and PLI's
   temporal-aggregation formula.
7. AR/BAR model order `p` for Granger Causality.
8. Whether the final graph is 32-node or 128-node (D5), and whether GC-based edges are kept
   directional or symmetrized.
9. Precise reconciliation of the two threshold definitions (D6) and whether GAT input is a
   weighted or binary adjacency matrix.
10. Number of GAT attention heads, GAT hidden dimension, multi-head aggregation rule
    (concat/average), and the GAT activation function `σ`.
11. Linear-projection layer dimensionality between GAT output and Transformer input.
12. Transformer feed-forward inner dimension (`d_ff`), positional encoding scheme (EEG
    graph sequences have no inherent word-order — a scheme must be chosen or justified as
    unnecessary), and dropout rates throughout.
13. Optimizer hyperparameters: learning rate, weight decay, β1/β2, batch size, number of
    epochs, LR schedule, early-stopping criterion, gradient clipping.
14. Train/validation/test split protocol (ratio, subject-wise vs. window-wise, k-fold,
    stratification, random seed).
15. Electrode-name (10-20-style) ↔ 128-channel GSN montage mapping (needed only for the
    anatomical-region discussion in §4.2/4.5 of the paper, not for the core model).

---

## 12. Testability Design (traceability to repo structure in §13)

| Component | Independently testable via | Test data |
|---|---|---|
| Butterworth filter | `src/asd_stgt/preprocessing/butterworth.py` | Synthetic sine + noise signal, known frequency response |
| SG smoothing | `src/asd_stgt/preprocessing/savgol.py` | Synthetic polynomial + noise signal |
| Gaussian augmentation | `src/asd_stgt/augmentation/gaussian_noise.py` | Synthetic single-channel signal, check output count = m*2 |
| Each of the 10 time-domain node features | `src/asd_stgt/features/node/time_domain.py` | Hand-computed toy vectors with known mean/var/skew/etc. |
| Frequency-domain features | `src/asd_stgt/features/node/frequency_domain.py` | Synthetic multi-sine signal with known band power |
| PC connectivity | `src/asd_stgt/features/connectivity/partial_correlation.py` | Small synthetic multi-channel array with known correlation structure |
| PLI connectivity | `src/asd_stgt/features/connectivity/pli.py` | Synthetic signals with a fixed known phase lag |
| GC connectivity | `src/asd_stgt/features/connectivity/granger_causality.py` | Synthetic AR-coupled bivariate series with known causal direction |
| Graph construction/thresholding | `src/asd_stgt/graph/build_graph.py` | Small synthetic adjacency values, verify thresholding rule |
| GAT layer | `src/asd_stgt/models/gat.py` | Random small graph, verify output shape + attention rows sum to 1 |
| Transformer encoder | `src/asd_stgt/models/transformer.py` | Random sequence tensor, verify output shape |
| Full STGT (synthetic end-to-end) | `src/asd_stgt/models/stgt.py` | Fully synthetic dataset (random node features + random graphs + random binary labels) — validates the pipeline runs and shapes are consistent; **explicitly not a claim of reproducing the paper's results** |

---

## 13. Repository Structure (see also IMPLEMENTATION_STATUS.md)

```
asd-stgt/
├── README.md
├── IMPLEMENTATION_STATUS.md
├── PAPER_IMPLEMENTATION_SPEC.md
├── requirements.txt
├── .gitignore
├── configs/                     # YAML configs for pipeline stages (created as needed per phase)
├── data/
│   ├── raw/                     # never committed; real EEG data goes here (not included)
│   ├── interim/                 # intermediate preprocessing outputs
│   └── processed/               # final feature/graph tensors ready for modeling
├── docs/
│   └── paper_tables/            # verbatim transcriptions of paper tables for traceability
├── src/asd_stgt/
│   ├── preprocessing/           # Butterworth, Savitzky-Golay
│   ├── augmentation/            # Gaussian-noise augmentation
│   ├── features/
│   │   ├── node/                # time-domain + frequency-domain node features
│   │   └── connectivity/        # PC, PLI, GC
│   ├── graph/                   # graph construction + thresholding
│   ├── models/                  # GAT, Transformer, STGT (full model)
│   ├── training/                # training loop, loss, optimizer setup
│   ├── evaluation/               # metrics, confusion matrix, ROC/PR
│   └── utils/                   # shared helpers (I/O, config loading, seeding)
├── tests/
│   ├── unit/                    # one test module per component in §12
│   └── integration/             # synthetic end-to-end pipeline test
├── experiments/
│   ├── ablation/                # ablation run configs/scripts (Phase 2+)
│   └── results/                 # generated metrics/plots (Phase 2+, gitignored data, tracked code)
├── saved_models/                # trained checkpoints (gitignored)
├── notebooks/                   # exploratory analysis (not part of the reproducible pipeline)
└── scripts/                     # CLI entry points (e.g. run_preprocessing.py) — added as needed
```

This structure keeps every pipeline stage independently importable and testable, matches
the paper's own two-stage decomposition, and avoids premature complexity (no experiment
tracking framework, no web UI, no packaging/distribution scaffolding at this stage).

---

## 14. What Phase 0 Explicitly Does NOT Do

- No dataset download, simulation-as-real, or fabrication.
- No real model training.
- No claim that any part of the paper has been reproduced.
- No hyperparameters chosen (only *identified as missing*, per §11).
- No resolution of the ambiguities in §9 — these remain open decisions for Phase 1 onward,
  to be made explicitly and labeled as assumptions when implementation begins.
