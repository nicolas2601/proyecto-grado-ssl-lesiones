"""Single-factor guard for the original/transformed paired contrast.

The project already has ``one_factor_gate.validate_one_factor`` on the cluster,
but it guards a different experiment: its ``ALLOWED`` set is
``{'freeze_blocks': 4, 'resolution': 336}`` and it refuses anything else. The
tone contrast does not move either of those; it moves ``adaptation_mode`` and
the transformed-manifest fields, so that gate rejects it.

Nothing else in the repository closes the gap. A review on 2026-11-03 checked
``run_contract.py`` and found that:

* ``RunVariant`` carries ``name, backbone, stage, split_hash, seed, probe,
  weights`` and no hyperparameter at all;
* ``validate_paired_variants`` compares the ``base`` and ``adapted`` stages of a
  backbone, which is a different contrast with a different object;
* ``record_execution_evidence`` validates one finished run's evidence and never
  compares two runs;
* ``check_initial_weights_evidence`` likewise.

So the property the whole contrast depends on - that the two arms differ in
exactly one factor - had no automatic check. This module is that check.

It is deliberately a refusal, not a report. A contrast whose arms differ in two
places produces a delta nobody can attribute, and the cheapest moment to learn
that is before the GPU runs, not after.
"""

from __future__ import annotations

# Fields that may differ between the control and the treated arm. These are the
# transform itself and the provenance that describes it - nothing that changes
# optimization, data order, or model capacity.
TREATMENT_FIELDS = frozenset(
    {
        "adaptation_mode",
        "transformed_image_root",
        "transformed_manifest_path",
        "transformed_data_revision",
    }
)

# Fields that are allowed to differ because they are about where a run writes,
# not about what it computes. Mirrors the OPERATIONAL set of the sibling gate.
OPERATIONAL_FIELDS = frozenset(
    {
        "output_dir",
        "run_name",
        "resume",
        "resume_checkpoint",
    }
)

CONTROL_MODE = "two_view_ssl"
TREATED_MODE = "original_transformed_consistency"


class PairedFactorError(ValueError):
    """Raised when the two arms are not a single-factor contrast."""


def _differing(control: dict, treated: dict) -> set[str]:
    keys = (set(control) | set(treated)) - OPERATIONAL_FIELDS
    return {key for key in keys if control.get(key) != treated.get(key)}


def validate_paired_factor(control: dict, treated: dict) -> set[str]:
    """Refuse unless the two effective configurations are a one-factor contrast.

    ``control`` and ``treated`` are the effective ``PretrainConfig`` dicts the
    two arms will actually run with - not the CLI strings, and not the protocol
    document. Reading them from the launch JSON instead would check the
    intention rather than the thing that runs.

    Returns the set of differing fields when it passes, so the caller can record
    in the parity receipt what actually differed rather than asserting that
    nothing did.
    """
    if control.get("adaptation_mode") != CONTROL_MODE:
        raise PairedFactorError(
            f"el brazo control debe correr en {CONTROL_MODE!r}, "
            f"no en {control.get('adaptation_mode')!r}"
        )
    if treated.get("adaptation_mode") != TREATED_MODE:
        raise PairedFactorError(
            f"el brazo tratado debe correr en {TREATED_MODE!r}, "
            f"no en {treated.get('adaptation_mode')!r}"
        )
    if not treated.get("transformed_image_root"):
        raise PairedFactorError("el brazo tratado no declara transformed_image_root")
    if control.get("transformed_image_root"):
        raise PairedFactorError(
            "el brazo control declara transformed_image_root; entonces no es un control"
        )

    changed = _differing(control, treated)
    outside = changed - TREATMENT_FIELDS
    if outside:
        raise PairedFactorError(
            "los dos brazos difieren fuera del tratamiento, asi que el delta no seria "
            f"atribuible: {sorted(outside)}"
        )

    # The seed is the one field whose equality is not merely desirable: a
    # different seed changes row order and batch boundaries, which is the exact
    # failure this gate exists to catch, and it would otherwise hide inside
    # "no differing fields" if both arms simply omitted it.
    if control.get("seed") is None or control.get("seed") != treated.get("seed"):
        raise PairedFactorError(
            f"los brazos deben compartir una semilla explicita; "
            f"control={control.get('seed')!r} tratado={treated.get('seed')!r}"
        )
    return changed
