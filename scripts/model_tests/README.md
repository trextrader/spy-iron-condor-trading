# Model Test Suite

This folder contains targeted, low-risk test scripts to validate the end-to-end
model pipeline (data → features → rules → normalization → model → checkpoint →
inference). Each script is standalone and can be run independently.

Recommended run order:
1) 01_checkpoint_inventory.py
2) 02_feature_schema_alignment.py
3) 03_norm_stats_sanity.py
4) 04_model_forward_contract.py
5) 06_ruleset_execution_smoke.py
6) 07_feature_pipeline_parity.py
7) 05_output_distribution.py

Notes:
- Most scripts accept --model and/or --data.
- Scripts are intentionally conservative and will exit with non-zero on failures.
