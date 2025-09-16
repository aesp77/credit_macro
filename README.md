# credit_macro workspace

This workspace contains starter scaffolding for the CDS macro monitor and ML experiments.

Structure:
- data/         # raw and processed datasets
- output/       # generated outputs, dashboards, CSVs
- notebooks/    # exploratory notebooks and experiments
- src/
  - data/       # data ingest and preprocessing
  - models/     # model definitions (Keras-Core / torch backend)
  - utils/      # utility functions

Quick start
1. Create and activate a Python environment (recommended: Python 3.11+).
2. Install requirements: see `requirements.txt`.
3. Set Keras-Core to use the torch backend (optional):

   On PowerShell:

   $env:KERAS_BACKEND = "torch"

   or set in your environment before running.

Notes
- This scaffold uses Keras-Core (Keras 3) API with a torch backend. The repo does not push to GitHub for you; add a remote and push with your credentials.
- Next steps: implement pricing modules (CDS, swaption spreads), risk calculation, volatility estimators, and ML experiments using the starter model in `src/models`.
