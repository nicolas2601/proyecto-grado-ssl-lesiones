"""Fine-tuning bank for the baseline backbones, cluster edition.

The Kaggle notebooks in ``notebooks/finetune*`` and the cluster runner in
``scripts/finetune_baseline.py`` share ONE protocol, declared in
:mod:`dermssl.finetune.protocol`. A test pins the notebook builder to it so
the two paths cannot drift silently.
"""
