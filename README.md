# ASD-STGT

Phased, test-driven implementation of the paper:

> "EEG-Based Early Detection of Autism Spectrum Disorder Using Two-Stage Feature Extraction
> and a Spatio-Temporal Graph Transformer" — J. Beno Ranjana, R. Muthukkumar.

**Status: Phase 0 (specification only). No ML pipeline is implemented yet.**

- Read `PAPER_IMPLEMENTATION_SPEC.md` first — the exact, categorized specification derived
  from the paper (what's explicit, what's derived, what's assumed, what's ambiguous).
- Read `IMPLEMENTATION_STATUS.md` for current progress, planned phases, and the exact next step.
- No dataset is included or downloaded by this repository. The source paper's dataset (NDAR)
  is described as confidential/available on request; this project will only use real data if
  and when the user legitimately provides it.
- No results in this repository should be interpreted as a reproduction of the paper's
  reported accuracy (92.59%) until real-data experiments are explicitly run and reported as such.

## Repository layout
See `PAPER_IMPLEMENTATION_SPEC.md` §13 for the full annotated structure.

## Setup
```bash
pip install -r requirements.txt
pytest tests/
```
(Test suite is currently empty — populated starting Phase 1.)
