"""protocol.job_for: any (model, dataset) pair, manifest() untouched."""

import pytest

from dermssl.finetune import protocol

MANIFEST_PAIRS = [(m, d) for d in ("ham10000", "isic2019") for m in protocol.MODELS]


def test_manifest_still_returns_the_same_16_jobs():
    jobs = protocol.manifest()
    assert [(j.model, j.dataset) for j in jobs] == MANIFEST_PAIRS
    assert [j.index for j in jobs] == list(range(16))
    assert protocol.FINETUNE_DATASETS == ("ham10000", "isic2019")


@pytest.mark.parametrize("model", protocol.MODELS)
@pytest.mark.parametrize("dataset", tuple(protocol.DATASETS))
def test_job_for_builds_every_model_dataset_pair(model, dataset):
    job = protocol.job_for(model, dataset)
    assert (job.model, job.dataset) == (model, dataset)
    assert job.run_name == f"ft-{model}-{dataset}-s42"


def test_job_for_isic2020_names_follow_the_committed_convention():
    job = protocol.job_for("dinov2_vitb14", "isic2020")
    assert job.results_file == "results_dinov2_vitb14_ft_isic2020.json"
    assert job.checkpoint_file == "ft_dinov2_vitb14_isic2020.pth"
    assert job.index == protocol.OUTSIDE_MANIFEST


def test_job_for_keeps_the_manifest_index_when_the_pair_is_in_it():
    assert protocol.job_for("resnet50", "isic2019").index == 11


@pytest.mark.parametrize("pair", [("resnet51", "ham10000"), ("resnet50", "pad_ufes")])
def test_job_for_rejects_unknown_pairs_with_a_typed_error(pair):
    with pytest.raises(protocol.UnknownJobError):
        protocol.job_for(*pair)
    assert issubclass(protocol.UnknownJobError, ValueError)
