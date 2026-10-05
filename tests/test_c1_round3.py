"""Regression tests for the round-3 review findings.

The worst one here (input scale) is a bug that produces no error and no warning:
feeding a U-Net trained on [0,1] an array in [0,255] saturates every activation
and returns masks that look like masks. Nothing downstream would have flagged
it -- the 24 cells would simply have been conditioned on garbage.

The contract check lives in a torch-free module on purpose, so it can be tested
on a machine without the GPU stack.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dermssl import c1_mask_export as mx
from dermssl import c1_mask_provenance as mp
from dermssl import c1_provenance as pv

# P1.1 (input scale) moved to tests/test_image_contract.py when the contract
# became its own module; the import it left behind failed ruff in CI.


# ── P1.3 tolerance bypass ─────────────────────────────────────────────────────


class TestToleranceCannotBypassTheGate:
    def _features(self):
        a = np.zeros((4, 8), dtype=np.float32)
        b = a.copy()
        b[0, 0] = 1.0  # a drift no tolerance should ever accept
        return a, b

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_tolerance_is_refused(self, bad):
        """`1.0 > nan` is False, so NaN silently turned the gate into a no-op."""
        a, b = self._features()
        with pytest.raises(ValueError, match="finite"):
            pv.compare_to_bank(a, b, tolerance=bad)

    @pytest.mark.parametrize("bad", [0.0, -1e-5])
    def test_non_positive_tolerance_is_refused(self, bad):
        a, b = self._features()
        with pytest.raises(ValueError, match="positive|range"):
            pv.compare_to_bank(a, b, tolerance=bad)

    def test_absurdly_loose_tolerance_is_refused(self):
        """A tolerance of 1.0 would pass anything; the gate must stay a gate."""
        a, b = self._features()
        with pytest.raises(ValueError, match="1e-3|too loose"):
            pv.compare_to_bank(a, b, tolerance=1.0)

    def test_the_documented_default_is_accepted(self):
        a, b = self._features()
        assert not pv.compare_to_bank(a, b, tolerance=1e-4)["reproduces"]

    def test_nan_in_the_features_cannot_pass(self):
        a = np.zeros((4, 8), dtype=np.float32)
        b = a.copy()
        b[1, 1] = math.nan
        assert not pv.compare_to_bank(a, b, tolerance=1e-4)["reproduces"]


# ── P2.6 sha must be hex ──────────────────────────────────────────────────────


class TestShaValidation:
    def _manifest(self, **over):
        payload = {
            "producer": "student_r3",
            "producer_is_label_free": True,
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": "f" * 64,
            "image_ids": ["ISIC_0000001"],
            "mask_shape": [64, 64],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": {
                "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
                "tta": False,
                "resolution": 224,
                "threshold": 0.5,
                "upsample": "bilinear_prob_then_threshold",
            },
        }
        payload.update(over)
        return payload

    def test_sixty_four_non_hex_chars_are_refused(self, registered_checkpoint):
        with pytest.raises(ValueError, match="sha256"):
            mp.MaskSource.from_manifest(self._manifest(masks_sha256="z" * 64))

    def test_uppercase_hex_is_refused(self, registered_checkpoint):
        with pytest.raises(ValueError, match="sha256"):
            mp.MaskSource.from_manifest(self._manifest(masks_sha256="A" * 64))

    def test_lowercase_hex_is_accepted(self, registered_checkpoint):
        assert mp.MaskSource.from_manifest(self._manifest()).masks_sha256 == "a" * 64


# ── P2.7 partial artifacts ────────────────────────────────────────────────────


class TestPartialArtifacts:
    def _manifest(self, **over):
        payload = {
            "n_images": 1,
            "image_ids": ["ISIC_0000001"],
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": "f" * 64,
            "checkpoint_path": "student_r3/best.pt",
            "code_revision": "abc1234",
            "mask_shape": [64, 64],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": {
                "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
                "tta": False,
                "resolution": 224,
                "threshold": 0.5,
                "upsample": "bilinear_prob_then_threshold",
            },
        }
        payload.update(over)
        return mx.build_masks_manifest(**payload)

    def test_full_run_records_partial_false(self):
        assert self._manifest()["partial"] is False

    def test_partial_run_is_recorded(self):
        assert self._manifest(partial=True)["partial"] is True

    def test_partial_artifact_is_refused_by_default(self, registered_checkpoint):
        """A --limit smoke artifact must not silently become the real input."""
        manifest = self._manifest(partial=True)
        manifest["producer_is_label_free"] = True
        with pytest.raises(ValueError, match="partial"):
            mp.MaskSource.from_manifest(manifest)

    def test_partial_artifact_accepted_with_explicit_opt_in(self, registered_checkpoint):
        manifest = self._manifest(partial=True)
        manifest["producer_is_label_free"] = True
        assert mp.MaskSource.from_manifest(manifest, allow_partial=True).partial


# ── P1.5 fallback tally reaches the report ────────────────────────────────────


class TestFallbackReporting:
    def test_summary_flags_a_high_fallback_rate(self):
        tally = pv.FallbackTally()
        for i in range(100):
            tally.record(i < 5)  # 5 %
        summary = tally.summary()
        assert summary["cls_fallback_rate"] == pytest.approx(0.05)
        assert summary["warning"], "5% of images falling back must be surfaced"

    def test_low_rate_is_not_flagged(self):
        tally = pv.FallbackTally()
        for i in range(1000):
            tally.record(i < 5)  # 0.5 %
        assert not tally.summary()["warning"]

    def test_threshold_is_one_percent(self):
        assert pv.FallbackTally.WARN_ABOVE == pytest.approx(0.01)

    def test_zero_observations_does_not_warn(self):
        assert not pv.FallbackTally().summary()["warning"]


# ── run identity as a contract, not a comment ─────────────────────────────────


class TestSegmentationRunIdentity:
    """The sha proves the file did not change; it does not prove it is the right file.

    Two checkpoints on the pod are both named student_r3: the published `final`
    run (d81a492, Dice 0,718/0,785) and the rejected `v2` max_area_fraction
    ablation (91814a9, 0,716/0,783). Hashing the wrong one matches itself
    perfectly, so identity has to be declared and checked separately.
    """

    def _manifest(self, **over):
        payload = {
            "producer": "student_r3",
            "producer_is_label_free": True,
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": "f" * 64,
            "image_ids": ["ISIC_0000001"],
            "mask_shape": [64, 64],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": {
                "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
                "tta": False,
                "resolution": 224,
                "threshold": 0.5,
                "upsample": "bilinear_prob_then_threshold",
            },
        }
        payload.update(over)
        return payload

    def test_final_run_is_accepted(self, registered_checkpoint):
        source = mp.MaskSource.from_manifest(self._manifest())
        assert source.segmentation_run == "final"
        assert source.segmentation_commit == "d81a492"

    def test_the_rejected_v2_ablation_is_refused(self, registered_checkpoint):
        with pytest.raises(ValueError, match="segmentation_run|ablation"):
            mp.MaskSource.from_manifest(
                self._manifest(segmentation_run="v2", segmentation_commit="91814a9")
            )

    def test_right_run_with_wrong_commit_is_refused(self, registered_checkpoint):
        """A manifest claiming 'final' at the ablation's commit is inconsistent."""
        with pytest.raises(ValueError, match="commit"):
            mp.MaskSource.from_manifest(self._manifest(segmentation_commit="91814a9"))

    def test_missing_run_is_refused(self, registered_checkpoint):
        manifest = self._manifest()
        del manifest["segmentation_run"]
        with pytest.raises(ValueError, match="segmentation_run"):
            mp.MaskSource.from_manifest(manifest)

    def test_published_commit_is_pinned_in_code(self):
        assert mp.PUBLISHED_SEGMENTATION == ("final", "d81a492")


class TestManifestCarriesRunIdentity:
    def _built(self, **over):
        payload = {
            "n_images": 1,
            "image_ids": ["ISIC_0000001"],
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": "f" * 64,
            "checkpoint_path": "salidas/final/student_r3/best.pt",
            "code_revision": "abc1234",
            "mask_shape": [64, 64],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": {
                "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
                "tta": False,
                "resolution": 224,
                "threshold": 0.5,
                "upsample": "bilinear_prob_then_threshold",
            },
        }
        payload.update(over)
        return mx.build_masks_manifest(**payload)

    def test_carries_the_identity_it_was_given(self):
        manifest = self._built()
        assert manifest["segmentation_run"] == "final"
        assert manifest["segmentation_commit"] == "d81a492"

    def test_a_manifest_built_here_satisfies_the_consumer(self, registered_checkpoint):
        """Producer and consumer must agree, or step 0 builds an unusable artifact."""
        manifest = self._built()
        manifest["producer_is_label_free"] = True
        assert mp.MaskSource.from_manifest(manifest).segmentation_run == "final"


# ── round 4 ───────────────────────────────────────────────────────────────────


class TestSingleShaValidator:
    """One definition of 'is this a sha256', not three."""

    def test_the_validator_is_shared(self):
        from dermssl import c1_mask_export as export
        from dermssl import c1_mask_provenance as prov

        assert prov.is_sha256 is export.is_sha256

    def test_accepts_lowercase_hex(self):
        assert mx.is_sha256("a" * 64)

    @pytest.mark.parametrize("bad", ["A" * 64, "z" * 64, "a" * 63, "a" * 65, "", "deadbeef"])
    def test_rejects_everything_else(self, bad):
        assert not mx.is_sha256(bad)


# ── round 5: run identity derived from the file, not declared ─────────────────

@pytest.fixture
def registered_checkpoint(monkeypatch):
    """Pretend the pod measurement already happened.

    Production fails closed while FINAL_STUDENT_R3_SHA256 is None; these suites
    test what happens AFTER that gate, so they register a known sha rather than
    weakening the gate itself.
    """
    sha = "f" * 64
    monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", sha)
    return sha


_V2_SHA = "9" * 64      # stands in for the rejected ablation checkpoint
_FINAL_SHA = "f" * 64   # stands in for the published one


class TestRunIdentityIsTiedToCheckpointContent:
    """A manifest that names its own run proves nothing about the file.

    The producer used to write final/d81a492 by default, so a run built from
    the v2 ablation was labelled 'final' and every consumer believed it. The
    run now has to be *derived* from the checkpoint's sha, which is a property
    of the bytes rather than a claim about them.
    """

    def _manifest(self, **over):
        payload = {
            "producer": "student_r3",
            "producer_is_label_free": True,
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": _FINAL_SHA,
            "image_ids": ["ISIC_0000001"],
            "mask_shape": [64, 64],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": {
                "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
                "tta": False,
                "resolution": 224,
                "threshold": 0.5,
                "upsample": "bilinear_prob_then_threshold",
            },
        }
        payload.update(over)
        return payload

    def test_fails_closed_while_the_sha_is_unmeasured(self, monkeypatch):
        """Nothing may run on an unverified checkpoint -- not even the happy path.

        The constant is set now (measured 2026-09-26), so the gate is exercised
        by putting it back to None rather than by reading production state: a
        test that asserts the current value would have to be edited every time
        the value changes, and it is the REFUSAL that has to keep working.
        """
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", None)
        monkeypatch.setattr(mp, "S4_UNET_A2_SHA256", None)  # A2 is pinned too since 3f15487
        assert mp.checkpoint_registry() == {}
        with pytest.raises(ValueError, match="todav|not registered|unregistered"):
            mp.MaskSource.from_manifest(self._manifest())

    def test_the_registered_sha_is_the_one_measured_on_the_pod(self):
        """Pin the digest itself, so a typo or a revert is a failing test.

        Nothing else in the suite would notice: every other case monkeypatches
        the constant, so a corrupted production value passes all of them and
        only fails on the pod, hours into a run.
        """
        assert mp.FINAL_STUDENT_R3_SHA256 == (
            "ac87b133dd50e74a88d9137b2f1ba76b986ee176017b8b009b0caf52a00c5cc0"
        )
        assert mx.is_sha256(mp.FINAL_STUDENT_R3_SHA256)
        assert mp.resolve_run(mp.FINAL_STUDENT_R3_SHA256) == mp.PUBLISHED_SEGMENTATION

    def test_accepts_the_registered_checkpoint(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        source = mp.MaskSource.from_manifest(self._manifest())
        assert source.segmentation_run == "final"

    def test_the_v2_checkpoint_cannot_be_labelled_final(self, monkeypatch):
        """The exact bug opencode demonstrated."""
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(ValueError, match="checkpoint_sha256|unregistered|no corresponde"):
            mp.MaskSource.from_manifest(self._manifest(checkpoint_sha256=_V2_SHA))

    def test_declared_run_must_agree_with_the_one_derived_from_the_sha(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(ValueError, match="segmentation_run|contradice|contradict"):
            mp.MaskSource.from_manifest(self._manifest(segmentation_run="v2"))

    def test_missing_checkpoint_sha_is_refused(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        manifest = self._manifest()
        del manifest["checkpoint_sha256"]
        with pytest.raises(ValueError, match="checkpoint_sha256"):
            mp.MaskSource.from_manifest(manifest)


class TestProducerCannotDefaultTheRun:
    def test_build_manifest_requires_run_and_commit_explicitly(self):
        with pytest.raises(TypeError, match="segmentation_run|argument"):
            mx.build_masks_manifest(
                n_images=1,
                image_ids=["ISIC_0000001"],
                masks_sha256="a" * 64,
                checkpoint_sha256="b" * 64,
                checkpoint_path="x.pt",
                code_revision="abc1234",
                mask_shape=[64, 64],
            )


class TestCheckpointShaMustBeARealString:
    """`str(x)` on an arbitrary object is a coercion, not a check.

    An object whose __str__ returns a registered digest would be coerced into
    one and accepted. Same shape as every other defect in this chain: an
    implicit conversion that repairs a bad value instead of refusing it.
    """

    class _LooksLikeASha:
        def __init__(self, digest):
            self._digest = digest

        def __str__(self):
            return self._digest

    def _manifest(self, sha):
        return {
            "producer": "student_r3",
            "producer_is_label_free": True,
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": sha,
            "image_ids": ["ISIC_0000001"],
            "mask_shape": [64, 64],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": {
                "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
                "tta": False,
                "resolution": 224,
                "threshold": 0.5,
                "upsample": "bilinear_prob_then_threshold",
            },
        }

    def test_object_impersonating_a_digest_is_refused(self, registered_checkpoint):
        impostor = self._LooksLikeASha(registered_checkpoint)
        assert str(impostor) == registered_checkpoint, "the impostor does coerce"
        with pytest.raises(TypeError, match="str|cadena"):
            mp.MaskSource.from_manifest(self._manifest(impostor))

    def test_bytes_are_refused(self, registered_checkpoint):
        with pytest.raises(TypeError, match="str|cadena"):
            mp.MaskSource.from_manifest(self._manifest(registered_checkpoint.encode()))

    def test_a_plain_string_still_works(self, registered_checkpoint):
        assert mp.MaskSource.from_manifest(self._manifest(registered_checkpoint))


# ── ronda 7: la inferencia declarada tambien se verifica ──────────────────────


class TestConsumerRefusesAnUndeclaredInference:
    """Same hole as ``segmentation_run``, one level down.

    A checkpoint does not identify a mask: the published path normalises with
    ImageNet statistics, another lineage does not, TTA averages four views, and
    the threshold is a free parameter. A manifest that omits all of that
    describes a file, not a result -- and until now the consumer accepted it.

    Hardened before the first artifact exists on purpose. Waiting until after
    would exempt the only artifact the thesis actually uses.
    """

    def _manifest(self, **over):
        payload = {
            "producer": "student_r3",
            "producer_is_label_free": True,
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": _FINAL_SHA,
            "image_ids": ["ISIC_0000001"],
            "mask_shape": [450, 600],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": dict(mp.PUBLISHED_INFERENCE),
        }
        payload.update(over)
        return payload

    def _with_inference(self, **over):
        inference = dict(mp.PUBLISHED_INFERENCE)
        inference.update(over)
        return self._manifest(inference=inference)

    def test_the_published_inference_is_accepted(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        source = mp.MaskSource.from_manifest(self._manifest())
        assert source.inference == mp.PUBLISHED_INFERENCE

    def test_a_manifest_without_the_block_is_refused(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        manifest = self._manifest()
        del manifest["inference"]
        with pytest.raises(ValueError, match="inference"):
            mp.MaskSource.from_manifest(manifest)

    def test_tta_on_is_refused(self, monkeypatch):
        """The published Dice is the no-TTA number; TTA masks are better than it."""
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(ValueError, match="tta"):
            mp.MaskSource.from_manifest(self._with_inference(tta=True))

    def test_a_moved_threshold_is_refused(self, monkeypatch):
        """0,4 is a different segmenter; the protocol fixes it at 0,5."""
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(ValueError, match="threshold"):
            mp.MaskSource.from_manifest(self._with_inference(threshold=0.4))

    def test_another_inference_path_is_refused(self, monkeypatch):
        """The exact bug: dermssl.torch_unet.infer_mask skips ImageNet normalisation."""
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(ValueError, match="path|infer_mask"):
            mp.MaskSource.from_manifest(
                self._with_inference(path="dermssl.torch_unet.infer_mask")
            )

    def test_a_different_resolution_is_refused(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(ValueError, match="resolution"):
            mp.MaskSource.from_manifest(self._with_inference(resolution=256))

    def test_resizing_labels_instead_of_probabilities_is_refused(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(ValueError, match="upsample"):
            mp.MaskSource.from_manifest(
                self._with_inference(upsample="threshold_then_bilinear_labels")
            )

    def test_a_truthy_tta_is_not_a_false_tta(self, monkeypatch):
        """`bool("false")` is True; a string here must not be coerced either way."""
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises((ValueError, TypeError), match="tta"):
            mp.MaskSource.from_manifest(self._with_inference(tta="false"))


class TestThePublishedInferenceMatchesThePublishedModules:
    """The constants are copies; these tests are what keeps them honest.

    ``c1_mask_provenance`` stays torch-free and importable from ``src`` alone,
    so it cannot import ``evaluate_segmentation`` (which lives under
    plan-segmentacion and needs two more entries on sys.path). The values are
    therefore registered here and tied to their source by test, the same way
    the checkpoint sha is.
    """

    def _published_module(self):
        import sys
        from pathlib import Path

        plan = Path(__file__).resolve().parents[1] / "plan-segmentacion" / "comun"
        for extra in (plan / "evaluacion", plan / "nucleo"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        import evaluate_segmentation

        return evaluate_segmentation

    def test_the_threshold_is_the_published_one(self):
        assert mp.PUBLISHED_INFERENCE["threshold"] == self._published_module().DEFAULT_THRESHOLD

    def test_the_path_names_a_function_that_exists(self):
        self._published_module()  # puts the evaluation folder on sys.path
        import eval_student

        module, _, function = mp.PUBLISHED_INFERENCE["path"].rpartition(".")
        assert module.endswith("eval_student")
        assert callable(getattr(eval_student, function))

    def test_the_keys_are_the_ones_the_exporter_writes(self):
        assert set(mp.PUBLISHED_INFERENCE) == set(mx.INFERENCE_KEYS)


class TestMasksShaIsNotCoerced:
    """Round 7 (P2). The round-6 fix landed on ``checkpoint_sha256`` only.

    ``digest = str(manifest["masks_sha256"])`` turns any object whose
    ``__str__`` returns a registered-looking digest into that digest. The
    artifact hash is what ties a published number to a file on disk, so
    coercing it is the same defect one field over.
    """

    class _LooksLikeASha:
        def __str__(self) -> str:
            return "a" * 64

    def _manifest(self, **over):
        payload = {
            "producer": "student_r3",
            "producer_is_label_free": True,
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": _FINAL_SHA,
            "image_ids": ["ISIC_0000001"],
            "mask_shape": [450, 600],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
            "inference": dict(mp.PUBLISHED_INFERENCE),
        }
        payload.update(over)
        return payload

    def test_an_object_that_prints_like_a_digest_is_refused(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(TypeError, match="masks_sha256"):
            mp.MaskSource.from_manifest(self._manifest(masks_sha256=self._LooksLikeASha()))

    @pytest.mark.parametrize("bad", [None, 12345, b"a" * 64, ["a" * 64]])
    def test_non_string_hashes_are_refused(self, monkeypatch, bad):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        with pytest.raises(TypeError, match="masks_sha256"):
            mp.MaskSource.from_manifest(self._manifest(masks_sha256=bad))

    def test_a_real_digest_still_passes(self, monkeypatch):
        monkeypatch.setattr(mp, "FINAL_STUDENT_R3_SHA256", _FINAL_SHA)
        assert mp.MaskSource.from_manifest(self._manifest()).masks_sha256 == "a" * 64
