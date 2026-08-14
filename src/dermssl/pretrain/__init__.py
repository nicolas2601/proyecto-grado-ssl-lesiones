"""dermssl.pretrain - label-free SSL pretraining core (v6).

Same contract as the parent package: every module imported here is numpy +
standard library only, so the logic that decides WHAT the cluster runs is unit
tested on any machine. The torch-dependent training code (models, loop) lives
in sibling modules that are imported lazily by ``scripts/pretrain.py`` and are
never required to run the test suite.

The one invariant this package exists to defend: no label ever reaches the
backbone. See ``config.assert_label_free`` and the loud gates around it.
"""

from . import budget, color_constancy, config, lesion_guided

__all__ = ["budget", "color_constancy", "config", "lesion_guided"]
