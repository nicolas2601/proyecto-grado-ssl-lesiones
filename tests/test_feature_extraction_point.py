"""The bank compares backbones only if every one is read at the same depth.

timm's `num_classes=0` is not that depth for VGG: it returns the 4096-d
pre-logits (fc6/fc7 of the classifier head), not the GAP of the last conv
block like every other CNN in the bank. The 2026-09-03 HAM10000 re-run caught
it as a silent 512 -> 4096 jump, so the expected width is pinned here and
asserted at load time in both runners.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl.finetune import protocol  # noqa: E402

BASELINE_BUILDER = (REPO / "scripts" / "build_baseline_tests.py").read_text()
TORCH_RUN = (REPO / "src" / "dermssl" / "finetune" / "torch_run.py").read_text()


def test_every_model_declares_its_feature_width():
    assert set(protocol.FEATURE_DIMS) == set(protocol.MODELS)


def test_feature_widths_are_the_conv_gap_or_cls_widths():
    assert protocol.FEATURE_DIMS == {
        "alexnet": 256,
        "vgg19": 512,           # GAP of conv5, NOT timm's 4096 pre-logits
        "efficientnet_b0": 1280,
        "resnet50": 2048,
        "vit_b16": 768,
        "simsiam_r50": 2048,
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
    }


def test_the_list_of_backbones_read_at_the_conv_gap_is_pinned():
    """The declared list, not an `if` buried in each runner."""
    assert protocol.CONV_GAP_FROM_TIMM == ("vgg19",)


def test_vgg19_is_read_at_the_conv_block_in_the_notebook_builder():
    # The generated notebook runs on Kaggle without the package, so it carries
    # its own copy; what must hold is that the copy covers the declared list.
    for model in protocol.CONV_GAP_FROM_TIMM:
        assert f'name == "{model}"' in BASELINE_BUILDER, (
            f"{model} esta declarado en CONV_GAP_FROM_TIMM y el notebook no lo trata")
    assert "gap(m.features)" in BASELINE_BUILDER


def test_the_cluster_runner_reads_the_declared_list_instead_of_a_hardcoded_name():
    """A dead flag is worse than no flag: it looks honoured and is not.

    CONV_GAP_FROM_TIMM existed while both runners branched on a literal
    "vgg19", so adding a backbone to the list would have changed nothing and
    the new model would have been read at the wrong depth - the exact failure
    the list was introduced to prevent.
    """
    assert "protocol.CONV_GAP_FROM_TIMM" in TORCH_RUN
    assert "_gap(m.features)" in TORCH_RUN
    assert 'name == "vgg19"' not in TORCH_RUN, (
        "volvio el nombre hardcodeado; la lista deja de gobernar")


def test_both_runners_assert_the_width_instead_of_trusting_it():
    assert "FEATURE_DIMS" in BASELINE_BUILDER
    assert "FEATURE_DIMS" in TORCH_RUN


def test_generated_notebooks_carry_the_width_guard():
    for bank in ("baselines", "baselines_isic2020"):
        for model in protocol.MODELS:
            pattern = ("test_{m}.ipynb" if bank == "baselines"
                       else "test_{m}_isic2020.ipynb")
            path = REPO / "notebooks" / bank / model / pattern.format(m=model)
            code = "".join("".join(c["source"])
                           for c in json.loads(path.read_text())["cells"]
                           if c["cell_type"] == "code")
            # The dict being present proves nothing: deleting the assert that
            # consumes it left all seven tests green when falsified. What has
            # to exist is the comparison that aborts the run.
            assert "assert DIM == FEATURE_DIMS[MODEL]" in code, (
                f"{bank}/{model} declares FEATURE_DIMS but never asserts it; "
                "a moved extraction point would produce a plausible number")


def test_the_readme_no_longer_claims_vgg19_reproduces_the_old_run():
    readme = (REPO / "notebooks" / "baselines" / "README.md").read_text()
    assert "vgg19 siguen comparables con las corridas previas" not in readme
