"""Job lookup for any (model, dataset) pair, kept out of ``protocol.py``.

``protocol.py`` is hashed into every generated Kaggle notebook
(``build_finetune_tests.generator_sha``); editing it for a lookup helper would
mark 24 published notebooks stale without any protocol value changing.
"""

from __future__ import annotations

from dermssl.finetune import protocol
from dermssl.finetune.protocol import Job

OUTSIDE_MANIFEST = -1


class UnknownJobError(ValueError):
    """The (model, dataset) pair is not one the protocol defines."""


def job_for(model: str, dataset: str) -> Job:
    """The job for any pair in MODELS x DATASETS, whether or not ``manifest()`` lists it.

    ``manifest()`` is the SLURM array and stays at 16 jobs; a pair outside it
    (isic2020, docs/33 G1.3) gets ``index == OUTSIDE_MANIFEST`` but the same run
    name, result file and checkpoint conventions.
    """
    if model not in protocol.MODELS or dataset not in protocol.DATASETS:
        raise UnknownJobError(
            f"no job for model={model!r} dataset={dataset!r}; "
            f"models {protocol.MODELS}, datasets {tuple(protocol.DATASETS)}")
    for job in protocol.manifest():
        if (job.model, job.dataset) == (model, dataset):
            return job
    return Job(index=OUTSIDE_MANIFEST, model=model, dataset=dataset)
