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
#
# ``tone_consistency_weight`` is here, added 2026-11-05, and it is the one entry
# that needs an argument rather than being self-evident. It is a loss weight, so
# at first reading it belongs to optimization and should be excluded. It is not,
# for a structural reason: ``levers.Levers.validate`` refuses any value above
# zero unless ``adaptation_mode`` is already the treated mode. The weight cannot
# vary independently of a field that is in this set, and in the control arm it
# cannot be anything but zero. It is the dose of the treatment, not a second
# factor, and the contrast is impossible without it differing - the control
# cannot carry a nonzero weight and the treated arm cannot carry a zero one
# without the term being inert.
#
# The alternative considered and rejected was relaxing ``levers`` so the weight
# could sit nonzero and inactive in the control. That keeps the two configs
# textually identical at the cost of recording a number that did not describe
# what ran, which is the failure mode this module exists to prevent.
TREATMENT_FIELDS = frozenset(
    {
        "adaptation_mode",
        "transformed_image_root",
        "transformed_manifest_path",
        "transformed_data_revision",
        "tone_consistency_weight",
    }
)

# Fields that are allowed to differ because they are about where a run writes,
# not about what it computes. Mirrors the OPERATIONAL set of the sibling gate.
#
# ``checkpoint_every_steps`` and ``checkpoint_fractions`` were added 2026-11-05.
# They decide how often a run persists its state; a run that checkpoints every
# 500 steps and one that only writes at the end compute identical gradients and
# see identical batches. Keeping them in the comparison made a probe fail for a
# reason that has nothing to do with attributability.
#
# ``world_size`` and ``effective_lr`` are different in kind: they are not inputs
# at all. ``resolved-params.json`` reports them as DERIVED values, computed from
# the base learning rate, the batch size and the visible device count. They are
# excluded here because comparing them means comparing a report against a config
# that has not produced one yet - which is a false difference, not a real one.
# Their inputs (``batch_size_per_gpu``, ``max_steps``, ``seed``, the levers) are
# all still compared, so an actual divergence in the learning rate cannot hide.
OPERATIONAL_FIELDS = frozenset(
    {
        "output_dir",
        "run_name",
        "resume",
        "resume_checkpoint",
        "checkpoint_every_steps",
        "checkpoint_fractions",
    }
)

# Values that a finished run reports rather than receives. They are dropped from
# both sides before comparison; see the note above OPERATIONAL_FIELDS.
DERIVED_FIELDS = frozenset(
    {
        "world_size",
        "effective_lr",
    }
)

CONTROL_MODE = "two_view_ssl"
TREATED_MODE = "original_transformed_consistency"


class PairedFactorError(ValueError):
    """Raised when the two arms are not a single-factor contrast."""


def _differing(control: dict, treated: dict) -> set[str]:
    keys = (set(control) | set(treated)) - OPERATIONAL_FIELDS - DERIVED_FIELDS
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

    # Admitting tone_consistency_weight into TREATMENT_FIELDS opens a hole this
    # closes: with the weight merely allowed to differ, both arms could carry
    # zero and the treated arm would pay for the transformed views while the
    # term that is supposed to use them stays inert. The contrast would pass
    # every check and measure nothing.
    control_weight = float(control.get("tone_consistency_weight") or 0.0)
    treated_weight = float(treated.get("tone_consistency_weight") or 0.0)
    if control_weight != 0.0:
        raise PairedFactorError(
            f"el brazo control declara tone_consistency_weight={control_weight}; "
            "en el control el termino de tono no existe y el peso debe ser cero"
        )
    if treated_weight <= 0.0:
        raise PairedFactorError(
            "el brazo tratado declara tone_consistency_weight="
            f"{treated_weight}; con peso cero el termino de tono es inerte y el "
            "contraste no mediria el tratamiento"
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
