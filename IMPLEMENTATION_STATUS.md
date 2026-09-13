# IMPLEMENTATION_STATUS.md

## Project Name
ASD-STGT — Reproduction/implementation of a Two-Stage Feature Extraction + Spatio-Temporal
Graph Transformer pipeline for EEG-based ASD detection.

## Research Paper
"EEG-Based Early Detection of Autism Spectrum Disorder Using Two-Stage Feature Extraction
and a Spatio-Temporal Graph Transformer" — J. Beno Ranjana, R. Muthukkumar, National
Engineering College, Kovilpatti, Tamil Nadu, India. (Source PDF provided in project files:
`ASD_Detection.pdf`.)

## Current Phase
**Phase 0 — Paper Implementation Specification.** No ML pipeline code has been implemented
yet, beyond an empty, importable package skeleton for structural validation.

## Completed Phases
- [x] Phase 0 — Paper study, exact implementation specification, repository skeleton,
      documentation. (This phase.)

## Current Implementation Status
- No preprocessing, augmentation, feature extraction, graph construction, GAT, Transformer,
  training, or evaluation code exists yet.
- No dataset has been downloaded, simulated as real, or fabricated.
- Repository skeleton exists and is importable (`src/asd_stgt/` package with empty
  `__init__.py` files per subpackage) so that later phases can add code incrementally
  without restructuring.
- `PAPER_IMPLEMENTATION_SPEC.md` contains the full specification extracted from the paper,
  including tensor-shape flags, numeric parameters, and 11 identified ambiguities
  (D1–D11) plus 15 missing hyperparameters (categorized C) — see that file for details.

## Paper-Specified Parameters (summary — full detail in PAPER_IMPLEMENTATION_SPEC.md §10)
- Dataset: NDAR, 189 subjects (97 ASD / 92 non-ASD), 128-channel HCGSN, 500 Hz, resting-state.
- Preprocessing: Butterworth band-pass 0.1–50 Hz order 4; cascade Savitzky-Golay smoothing.
- Augmentation: Gaussian-noise-based (σ±0.1 σ), paper-literal formula flagged as unusual (D2).
- Node features: 10 time-domain (Table 3) + 4 frequency-domain (Eqs. 6–9); 128×10=1280 stated.
- Connectivity: Partial Correlation, Phase Lag Index, Granger Causality (3 estimators).
- Graph: G={V,E,A}; thresholding = mean-based (two conflicting definitions, D6).
- Model: 2-layer GAT (from Figure 4) → linear projection → Transformer encoder
  (3 layers / 4 heads / 64-dim in the final reported config; ablated over 2–5 layers) →
  FC → Softmax, cross-entropy loss, Adam optimizer.
- Reported result: Accuracy 92.59%, Sensitivity 97.94%, Specificity 86.96%, F1 93.16%,
  ROC-AUC 0.97.

## Implementation Assumptions Made So Far
**None yet.** Phase 0 deliberately makes zero modeling/hyperparameter decisions. All 15
missing-hyperparameter items are listed, unresolved, in
`PAPER_IMPLEMENTATION_SPEC.md` §11, to be decided explicitly (and labeled as assumptions,
not paper facts) starting in Phase 1.

## Known Ambiguities / Inconsistencies (full detail in PAPER_IMPLEMENTATION_SPEC.md §9)
| ID | One-line summary |
|---|---|
| D1 | Confusion-matrix totals equal the full dataset; no train/test split protocol stated. |
| D2 | Augmentation formula multiplies signal × Gaussian PDF value, not conventional additive noise. |
| D3 | Table 5 feature-set naming inconsistent with Table 3; A3's 3 feature names never defined. |
| D4 | PLI formula given only as instantaneous; no temporal-aggregation rule stated. |
| D5 | **Figure 3 legend says "32 electrodes"; body text says 128 channels — direct contradiction.** |
| D6 | Two different, unreconciled threshold definitions (§3.4.1 vs. §4.2 Eq. 24). |
| D7 | GAT attention-head count and hidden dimension never specified. |
| D8 | Text says "encoder–decoder transformer"; Figure 4 shows encoder-only. |
| D9 | Best ablation row (94.44%) outperforms the paper's chosen "final" model (92.59%), unexplained. |
| D10 | Minor: Eq. numbering "(23)" used twice in the source paper (FFN and thresholding rule). |
| D11 | 10-20-style electrode names used for interpretation; no mapping to 128-ch GSN channel indices given. |

**D5 in particular blocks a concrete decision on core tensor shapes (32-node vs. 128-node
graph) and must be resolved as an explicit, documented configuration choice at the start of
Phase 1 — not silently defaulted.**

## Planned Phases (subject to refinement; not started)
- **Phase 1:** Preprocessing module (Butterworth + Savitzky-Golay) with unit tests on
  synthetic signals. Resolve/decide D5 (graph size) and document the decision. No real data.
- **Phase 2:** Augmentation module (both "paper-literal" and "conventional additive" modes)
  with unit tests.
- **Phase 3:** Stage-1 node feature extraction (time-domain, Table 3) with unit tests against
  hand-computed values.
- **Phase 4:** Frequency-domain node features (Eqs. 6–9) with unit tests.
- **Phase 5:** Stage-2 connectivity estimators (PC, PLI, GC) with unit tests against synthetic
  signals with known ground-truth correlation/phase-lag/causal structure.
- **Phase 6:** Graph construction + thresholding module, with the two thresholding
  definitions (D6) implemented as explicitly named, switchable strategies.
- **Phase 7:** GAT model module, tested on synthetic small graphs (attention rows sum to 1,
  output shapes correct).
- **Phase 8:** Transformer encoder module, tested on synthetic sequences.
- **Phase 9:** Full STGT assembly + synthetic end-to-end integration test (random data only;
  validates shapes/gradients flow, not paper reproduction).
- **Phase 10+:** Training loop, evaluation/metrics module, ablation experiment scripts,
  configuration system. Real-data experiments only after a real, legitimately obtained NDAR
  dataset is provided by the user — never fabricated or downloaded automatically.

Each phase will remain runnable and independently testable at every commit, per the
project's governing instructions.

## Repository Structure
See `PAPER_IMPLEMENTATION_SPEC.md` §13 for the full annotated tree. Summary:
```
asd-stgt/
├── README.md, IMPLEMENTATION_STATUS.md, PAPER_IMPLEMENTATION_SPEC.md
├── configs/           (empty scaffold; populated per phase)
├── data/{raw,interim,processed}/   (empty; .gitkeep only, no real data committed)
├── docs/paper_tables/ (verbatim Table 6/7 transcription)
├── src/asd_stgt/{preprocessing,augmentation,features/{node,connectivity},graph,models,
│                 training,evaluation,utils}/   (importable empty package skeleton)
├── tests/{unit,integration}/        (empty scaffold; populated per phase)
├── experiments/{ablation,results}/  (empty scaffold)
├── saved_models/       (gitignored)
├── notebooks/
└── scripts/
```

## Exact Next Step
Begin **Phase 1**: implement the preprocessing module (`src/asd_stgt/preprocessing/`) —
Butterworth band-pass filter and cascade Savitzky-Golay smoothing — as independently
testable, pure functions operating on synthetic NumPy arrays, with accompanying unit tests
in `tests/unit/`. Before writing any preprocessing code, explicitly propose and document the
resolution to ambiguity D5 (32 vs. 128 electrode graph size) since it affects every later
tensor shape, and record that resolution at the top of Phase 1's section in this file.

## Verification Requirements (for every phase, including this one)
1. Repository must remain in a runnable/importable state after the phase's commit.
2. Every new component must have at least one independent unit test using synthetic data
   (no real EEG data used until legitimately provided by the user).
3. `IMPLEMENTATION_STATUS.md` must be updated at the end of every phase: current phase,
   completed phases, new assumptions made (clearly labeled), and the exact next step.
4. No phase may claim the paper has been "reproduced"; only "implemented per specification
   X, pending real-data validation" language is permitted.
5. Any new ambiguity discovered during implementation must be added to the ambiguities table
   in both this file and `PAPER_IMPLEMENTATION_SPEC.md`, not silently resolved in code
   comments only.
