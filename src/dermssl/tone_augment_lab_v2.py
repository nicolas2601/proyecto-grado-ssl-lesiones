"""Mask-anchored CIELAB skin tone augmentation, Corbin eq. 7 with a measured ITA.

Sibling of :mod:`dermssl.tone_augment_lab`, the same way ``tone_augment_pairs``
was a sibling of ``tone_transforms``. ``tone_augment_lab`` is frozen by
contract and NOT modified here: its outputs back already-reported evidence,
and the paired comparison this module exists to run depends on that file
staying byte-identical. The geometry itself -- ``B_PER_ITA_DEGREE = 0.50``,
``L_PER_ITA_DEGREE = 0.12``, Corbin's eqs. 8-9 -- is reused unchanged and never
re-derived; what this module corrects is how far along that geometry to move.

Why a sibling instead of an edit: ``tone_augment_lab.sample_ita_delta`` draws
``ita_delta`` from a hash, decoupled from the image it is applied to. Corbin &
Marques (IEEE Access 11:78339, 2023) eq. 7 defines it as
``ita_delta = ITA_measured - ITA_objetivo`` -- anchored to a skin tone actually
present in the frame. What changed to make this possible:
``fitzpatrick.estimate_fst`` run with the teacher-026 segmentation mask
reaches MAE 0,91 degrees against an expert-mask ceiling on ISIC 2018
(n=1.813), bias -0,03 (``docs/progress/escalera-ita-mascara.md``, step 1-4).

REVISION 1 (superseded below, kept for the record). The first version sampled
``ITA_objetivo`` from a fixed global span, decoupled from the measured ITA.
On a realistic ISIC 2018 distribution (median ITA ~56) that span sits almost
entirely below the measured value, so the delta came out large and positive
on nearly every image and the frozen module's cap clipped 80%+ of them to the
same constant +25,0 -- measured: 79,9% of 5.000 synthetic draws landed within
0,025 degrees of the cap. The second revision replaced the fixed span with a
window centered on the measured ITA, which fixed the collapse (0,08% at the
boundary) but introduced REVISION 2's problem below.

REVISION 2 (superseded further below). ``ita_degrees = arctan((L*-50)/b*)`` is discontinuous
where skin b* crosses zero: the ratio flips sign and the angle jumps between
+90 and -90 (verified directly against ``fitzpatrick.ita_degrees``'s
docstring and implementation). A window-centered linear step treated this
curve as if it were linear everywhere, so solving for an exact target ITA by
linear extrapolation failed far from the origin: realized/requested ratio
collapsed to a median of 0,576, and a naive root search across the
discontinuity found the jump, not a root (47/200 converged, p95 error
75,66 degrees in a first attempt reported by a separate validation pass on
S-SYNTH, n=150 stratified by melanin).

The fix does not change the target: it changes how ``ita_delta`` is found for
a given target.

1. For each image, sweep a step ``t`` (the same quantity
   ``tone_augment_lab.augment_skin_tone`` calls ``ita_delta``) over a grid and
   measure the resulting skin ITA through the SAME round trip the frozen
   geometry performs (Lab shift, clip to sRGB gamut, back to Lab, median over
   skin pixels). Keep the maximal contiguous stretch around ``t = 0`` where
   that curve is finite and monotonically non-increasing in ``t`` -- the
   stretch where a root-find is well posed. This is the image's own reachable
   ITA range; it is NOT the same for every image, because gamut clipping and
   the location of the b*=0 discontinuity both depend on where the image's
   skin tone already sits.
2. ``ITA_objetivo`` is drawn inside that reachable range directly. There is
   no default policy cap layered on top anymore. REVISION 1 and 2 reused
   ``tone_augment_lab.MAX_ABS_ITA_DELTA`` (25,0) as that cap, reasoning that
   reusing an already-vetted number beat inventing a second one. That
   reasoning does not survive a direct check: ``MAX_ABS_ITA_DELTA`` is
   "Corbin's two-category limit in degrees", and this repo cites TWO
   different, inconsistent threshold tables for what a Corbin category is.
   ``fitzpatrick.FST_THRESHOLDS`` -- the table ``fst_bucket`` actually
   classifies with -- gives interior gaps 17/11/12/25, so two adjacent
   categories span 23 to 37 degrees (median 28). The docstring in
   ``tone_augment_lab.py`` instead cites Corbin's own printed ranges (T1 >55,
   T2 41-55, T3 28-41, T4 19-28, T5 10-19, T6 <=10), gaps 14/13/9/9, span 18
   to 27 (median 22). 25 sits inside the second range and outside neither by
   much, but it is anchored to a category table (Groh's) that this project
   has separately found does not hold on its own data: recalibrating those
   cuts converges to a constant predictor, 53,3% against a 55,8% trivial
   baseline (reported, not independently re-measured here). A cap derived
   from a retired categorization is not a number to keep propagating by
   default. This module's natural bound is already the PHYSICAL one from
   step 1 -- measured per image, not read off a disputed table -- so no
   additional cap is applied unless one is passed explicitly.
3. ``t`` is solved exactly for that target with ``scipy.optimize.brentq``
   inside the reachable bracket (optionally narrowed by an explicit
   ``policy_cap_degrees``, see below) -- a root-find, not a linear step,
   because the curve from (1) is monotonic there by construction, so the
   bracket always has a sign change.
4. If no usable monotonic stretch exists, this abstains:
   ``fallback_reason="no_reachable_ita_range"``, no delta invented. This is
   an explicit extension of the pre-existing abstention rule below, for the
   same reason: eq. 7 has no answer when the geometry it depends on cannot be
   inverted for this image.

A separate, EXPLICIT policy cap (visual plausibility, not physical
reachability -- the thing Corbin's "results beyond this maximum value did
not yield satisfactory visual outcomes" actually describes) can still be
passed via ``policy_cap_degrees``. It defaults to ``None``: no project
decision has set this number since the one this module used to inherit does
not hold up. Measured on S-SYNTH (n=150), the median per-image reachable
range is 75,4 degrees -- about three times the old 25 -- so whatever policy
value is eventually chosen, it is very unlikely to be the same number by
coincidence. This is left for the advisor to set; nothing here widens or
narrows it unilaterally.

INTEGRATION REQUIREMENT, not yet applicable because nothing calls this
module from a pipeline: ``None`` here is defensible only while unintegrated,
because it forces the decision to be explicit rather than silently
inherited. Applying the full 75-degree reachable range with no visual
plausibility cap would very likely produce images that do not look like
skin -- that risk has never been measured. Whoever wires this module into a
training pipeline MUST require ``policy_cap_degrees`` as an explicit,
non-defaulted argument AT THAT INTEGRATION POINT (not by changing the
default here, which would silently re-impose a choice this module
deliberately leaves open). The low-level functions in this module keep
``None`` as their own default; the integration call site is where the
requirement belongs.

A further restriction follows from how the reachable curve is measured:
``fitzpatrick.estimate_fst`` can fall back to its own Otsu-based skin mask
when the lesion mask it is given is degenerate or under-confident
(``measurement.fallback_used``). This module cannot see which pixels that
fallback mask selected, so it cannot build a curve that is guaranteed to
start at the measured ITA in that case. Rather than build a curve on a
DIFFERENT skin selection than the one that produced ``ITA_measured`` -- which
would silently break the "anchored to the image" property this whole module
exists for -- it abstains: ``fallback_reason="ita_measured_via_fallback_mask"``.

Performance: the curve in step 1 round-trips only the skin pixels, not the
full image, and subsamples to at most ``MAX_SKIN_PIXELS_FOR_CURVE`` (20.000)
when there are more -- the median is stable well below that count, and
round-tripping a full image per grid point does not finish in reasonable
time.

Two further rules this module enforces that eq. 7 alone does not:

* Abstention when ``ITA_measured`` itself is undefined --
  ``fitzpatrick.MIN_B_FOR_ITA`` reports roughly 30% of masked ISIC 2018 images
  as ``ita_indeterminate`` (``docs/progress/escalera-ita-mascara.md``). No
  delta is invented; the case is recorded with its reason. This default has
  not been formally ratified; if it changes, this module changes with it.
* A 50/50 mixing gate, independent of the eq. 7 draw, so half of any paired
  run's treatment-arm images keep their original tone even when ITA is
  measurable and reachable.

The objetivo draw and the mixing draw are hash-derived from
``(seed, step, image_sha256)``, under their own key namespace, for the same
reason ``tone_augment_lab.sample_ita_delta`` is: the paired control arm must
see an identical data order, so this module may not touch any RNG the sampler
also uses. The root-find itself is deterministic given that target and the
image's own pixels -- ``brentq`` draws no randomness.

Lesion preservation, the boundary feather, the gamut-clipping count and the
b*-sign-flip count are not reimplemented: once ``t`` is solved, it is handed
to ``tone_augment_lab.augment_skin_tone`` unchanged, which applies it to the
full image. That function can still record a sign flip on skin b* close to
``fitzpatrick.MIN_B_FOR_ITA``; this is measured and reported via its
``skin_median_b_after`` and ``b_sign_flipped_pixels`` fields, not hard-blocked
-- the same stance ``tone_augment_lab`` itself takes.

REVISION 4 (current). An independent end-to-end validation pass on real
S-SYNTH images found this module's solved ``t`` landing on the right target
inside its own curve, but the REAL output, re-measured with
``fitzpatrick.estimate_fst``, missing that target by a median of 2,33
degrees (p95 20,60), always in the lighter direction. Eight candidate causes
were ruled out one at a time (feather, stride subsampling, the initial ITA
measurement, both channels' shift fidelity, clipping, and the pre-transform
skin-set definitions) before finding two real, independent mismatches
between this module's curve and a real re-measurement of its output:

* The dominant one, by roughly an order of magnitude: this module's curve
  used to aggregate ITA as the MEDIAN OF PER-PIXEL ANGLES. ``estimate_fst``
  (and so any real re-measurement) aggregates it as the ANGLE OF THE
  MEDIAN L* AND MEDIAN b*. These differ whenever pixels have real L/b
  variance. Fixed in ``_skin_state_at_t``; see that function's docstring for
  the isolated reproduction.
* A smaller, separate effect: the curve's population is fixed at ``t = 0``,
  but a real re-measurement recomputes near-black border exclusion on the
  SHIFTED pixels, which can drop a texture-dependent fraction of that fixed
  population. See ``_population_shift_fraction``, which this module uses to
  abstain rather than silently report an unreachable target, instead of
  trying to make the curve itself track a population that changes size
  (an earlier attempt at that broke the bracket's monotonicity).
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import brentq

from dermssl.circle import lab_to_srgb
from dermssl.fitzpatrick import (
    BORDER_LUMINANCE_THRESHOLD,
    _resolve_border,
    estimate_fst,
    ita_degrees,
    srgb_to_lab,
)
from dermssl.tone_augment_lab import (
    B_PER_ITA_DEGREE,
    DEFAULT_FEATHER_SIGMA,
    L_PER_ITA_DEGREE,
    augment_skin_tone,
)
from dermssl.tone_normalization import _as_binary_mask, _check_image_range

# No default value: see REVISION 2 point 2 in the module docstring for why
# tone_augment_lab.MAX_ABS_ITA_DELTA (25,0) is NOT reused here. Pass a value
# explicitly to narrow the physically reachable range by a visual-plausibility
# policy; None applies no cap beyond physical reachability.
DEFAULT_POLICY_CAP_DEGREES: float | None = None

# Grid used to map each image's own reachable ITA range before solving for a
# target. 41 points over [-60, 60], matching the validation pass this fix is
# based on, so the two are directly comparable.
GRID_T_MIN = -60.0
GRID_T_MAX = 60.0
GRID_POINTS = 41

# The median of 20.000 skin pixels is stable; round-tripping every pixel of
# every grid point does not finish in reasonable time.
MAX_SKIN_PIXELS_FOR_CURVE = 20_000

BRENTQ_XTOL = 1e-3

# A declared engineering choice, NOT measured against real data: the largest
# fraction of the solving population allowed to fall below
# fitzpatrick.BORDER_LUMINANCE_THRESHOLD at the solved t before abstaining
# instead of reporting a target the real output will not land on. See
# _population_shift_fraction. 1% is conservative on purpose; revise with a
# real measured tolerance once one exists, rather than loosening this blind.
MAX_POPULATION_SHIFT_FRACTION = 0.01

# Independent of eq. 7: half of the images that pass abstention still keep
# their original tone. A draw above this threshold selects the transform.
MIX_THRESHOLD = 0.5

_MONOTONE_TOLERANCE = 1e-9


def _validate_key(seed: int, step: int, image_sha256: str) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {seed!r}")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("image_sha256 must be a 64-character hex digest")
    if any(character not in "0123456789abcdef" for character in image_sha256):
        raise ValueError("image_sha256 must be lowercase hex")


def _hash_unit_interval(namespace: str, seed: int, step: int, image_sha256: str) -> float:
    """Deterministic draw in ``[0, 1)``, independent of any mutable RNG."""
    key = f"{namespace}:{seed}:{step}:{image_sha256}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def sample_apply_draw(seed: int, step: int, image_sha256: str) -> float:
    """Deterministic draw in ``[0, 1)`` for the 50/50 mixing gate."""
    _validate_key(seed, step, image_sha256)
    return _hash_unit_interval("tone-augment-lab-v2-apply", seed, step, image_sha256)


def _unified_skin_fov(img: np.ndarray, border_mask: np.ndarray | None) -> np.ndarray:
    """A ``valid_fov`` that makes the TRANSFORMED set match the MEASURED set.

    The repo has two "skin" definitions that are supposed to be the same
    thing and are not: ``tone_normalization._skin_selector`` (what
    ``tone_augment_lab.augment_skin_tone`` actually shifts, ``~lesion_mask &
    valid_fov``, no near-black exclusion) and ``fitzpatrick._resolve_border``
    (what ``estimate_fst`` -- and this module's own curve -- measures,
    additionally excluding ``gray < BORDER_LUMINANCE_THRESHOLD``). On
    S-SYNTH they coincide exactly (0,0000% divergence, independently
    measured); on real dermoscopy with vignetting they need not.

    Unifying by EDITING ``_skin_selector`` would change
    ``tone_augment_lab.augment_skin_tone``'s effective behaviour on every
    caller, including already-reported ``tone_normalization`` results --
    the same frozen-contract risk as editing the file directly, just one
    import away. This sidesteps that: pass THIS as ``valid_fov`` to
    ``augment_skin_tone`` instead of the raw field-of-view, and
    ``_skin_selector(lesion_mask, this)`` reduces to exactly
    ``~lesion_mask & ~border`` -- the measurer's definition -- without
    touching ``_skin_selector``, ``tone_normalization.py`` or
    ``tone_augment_lab.py`` at all. The transformer adopts the measurer's
    criterion by construction, not by edit.
    """
    gray = img.mean(axis=-1)
    return ~_resolve_border(gray, border_mask)


def _skin_lab_for_curve(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    border_mask: np.ndarray | None,
    *,
    max_pixels: int = MAX_SKIN_PIXELS_FOR_CURVE,
) -> np.ndarray | None:
    """Skin Lab pixels, same selection ``fitzpatrick.estimate_fst`` used.

    Rebuilds ``~mask & ~border`` with the same ``_resolve_border`` call
    ``estimate_fst`` makes, so the curve's value at ``t = 0`` matches
    ``measurement.ita`` exactly. Returns ``None`` when there is no skin, which
    the caller treats the same as an unreachable range.

    Subsampling is by stride over the pixel array, not a random draw: the
    skin mask already fixes the pixel order, so this needs no RNG and no
    extra hash key.
    """
    skin = ~lesion_mask & _unified_skin_fov(img, border_mask)
    lab = srgb_to_lab(img)[skin]
    if lab.shape[0] == 0:
        return None
    if lab.shape[0] > max_pixels:
        stride = lab.shape[0] // max_pixels
        lab = lab[::stride][:max_pixels]
    return lab


def _skin_state_at_t(skin_lab: np.ndarray, t: float) -> tuple[float, float]:
    """Skin (ITA, b*) after shifting ``skin_lab`` by step ``t``, Corbin's round trip.

    ITA is the angle OF THE MEDIANS, ``ita_degrees(median(L), median(b))`` --
    the same aggregation ``fitzpatrick._skin_ita`` uses, and the one
    ``estimate_fst`` applies to any real image anyone measures downstream.
    An earlier version of this function computed the MEDIAN OF THE PER-PIXEL
    ANGLES instead, ``median(ita_degrees(L_i, b_i))``. The two are NOT the
    same statistic -- arctan(ratio) is nonlinear, so ``median(f(x))`` and
    ``f(median(x))`` diverge whenever the pixels have real L/b variance,
    which every real image has. This was the actual root cause of a residual
    an independent S-SYNTH validation pass measured end-to-end (median 2,33
    degrees, p95 20,60): isolated down to one textured synthetic case where
    the buggy median-of-angles version of this function disagreed with
    ``estimate_fst``'s measurement of the real, identically shifted pixels by
    over 10 degrees (same pixel count, no subsampling, no border exclusion
    involved -- ruling out every other candidate already investigated).
    Recomputing ``estimate_fst``'s number by hand from
    ``arctan((median_L - 50) / median_b)`` matched to six significant
    figures; recomputing it from the median of per-pixel angles reproduced
    the wrong one. ``_population_shift_fraction`` is a real, separate,
    smaller effect (see its own docstring); this is the dominant one.

    Deliberately does NOT drop pixels that would fall below
    ``fitzpatrick.BORDER_LUMINANCE_THRESHOLD`` at this ``t``: an earlier
    attempt did that per grid point, and it broke the monotonicity the
    bracket search depends on -- a texture-heavy synthetic test showed the
    curve dip and rise non-monotonically as different shadow pixels crossed
    the threshold at different grid points, well before any b*=0 crossing.
    The population this curve is measured over is fixed at ``t = 0`` by
    ``_skin_lab_for_curve`` and held fixed across the whole sweep on purpose.
    """
    shifted = skin_lab.copy()
    shifted[:, 0] = np.clip(skin_lab[:, 0] - t * L_PER_ITA_DEGREE, 0.0, 100.0)
    shifted[:, 2] = skin_lab[:, 2] + t * B_PER_ITA_DEGREE
    srgb = lab_to_srgb(shifted)
    after_lab = srgb_to_lab(srgb)
    l_med = float(np.median(after_lab[:, 0]))
    b_med = float(np.median(after_lab[:, 2]))
    with np.errstate(invalid="ignore", divide="ignore"):
        ita = float(ita_degrees(np.asarray(l_med), np.asarray(b_med)))
    return ita, b_med


def _population_shift_fraction(skin_lab: np.ndarray, t: float) -> float:
    """Fraction of ``skin_lab`` that a post-hoc re-measurement would drop at step ``t``.

    Root cause of a residual an independent S-SYNTH validation pass found
    (median 2,33 degrees, p95 20,60, always realized-lighter-than-predicted):
    this module's curve measures the median ITA over a FIXED pixel
    population, selected once at ``t = 0`` (``_skin_lab_for_curve`` already
    applies ``fitzpatrick.estimate_fst``'s near-black exclusion there, so the
    two agree exactly at the origin). ``tone_augment_lab.augment_skin_tone``
    then shifts that same population's colour, but ANY downstream consumer
    that re-measures the output with ``estimate_fst`` recomputes the border
    exclusion fresh, from the SHIFTED pixels. Removing the pixels that fall
    below the threshold always drops from the dark tail, which always pulls
    a median ITA higher (lighter) -- matching the direction measured on
    S-SYNTH exactly. Confirmed synthetically: a shadow-heavy skin patch
    where 11% of the fixed population newly fell below the threshold at one
    grid point showed a 1,85-degree gap between the fixed-population median
    and the recomputed one, at that same point.

    This is NOT fixed by making the curve itself drop those pixels per grid
    point -- that broke monotonicity instead (see ``_skin_state_at_t``). It
    is instead measured at the solved ``t`` only, so the caller can abstain
    when the two populations would diverge enough to matter, rather than
    silently reporting a target the real output will not land on.
    """
    shifted = skin_lab.copy()
    shifted[:, 0] = np.clip(skin_lab[:, 0] - t * L_PER_ITA_DEGREE, 0.0, 100.0)
    shifted[:, 2] = skin_lab[:, 2] + t * B_PER_ITA_DEGREE
    srgb = lab_to_srgb(shifted)
    newly_excluded = srgb.mean(axis=-1) < BORDER_LUMINANCE_THRESHOLD
    return float(np.mean(newly_excluded))


def _skin_ita_at_t(skin_lab: np.ndarray, t: float) -> float:
    """Median skin ITA after shifting ``skin_lab`` by step ``t``."""
    return _skin_state_at_t(skin_lab, t)[0]


def _reachable_bracket(
    skin_lab: np.ndarray,
    *,
    t_min: float = GRID_T_MIN,
    t_max: float = GRID_T_MAX,
    n_grid: int = GRID_POINTS,
) -> tuple[float, float, float, float] | None:
    """The maximal stretch around ``t = 0`` where the ITA(t) curve is usable.

    "Usable" means finite, non-increasing as ``t`` increases, AND on the same
    side of the b*=0 discontinuity throughout -- checking only the ITA values
    is not enough: a skin b* that crosses zero between two adjacent grid
    points can still look like a plain decrease in ITA at that resolution
    (the angle swings from +90 towards -90 across the crossing), so a sign
    check on b* between EVERY adjacent pair is required, not just a
    monotonicity check on ITA. Measured regression: a synthetic skin tone
    whose grid-adjacent samples carried b* -0,40 and +1,10 passed the
    ITA-only monotonicity check and produced a 70-degree realized error once
    ``brentq`` searched across that hidden crossing.

    This guard measured 0,0% activation on a 48-image S-SYNTH validation
    pass (an independent check, see this module's revision history) -- it
    never fired there. That is a property of S-SYNTH's skin, not evidence
    the guard is unneeded: that cohort's median skin b* (among images that
    resolve) sits comfortably positive (~28), so it is structurally unable
    to exhibit a crossing. The population where this guard matters is the
    opposite tail -- b* near zero, the same ~28% of real dermoscopy that
    already abstains via ``fitzpatrick.MIN_B_FOR_ITA`` for not having an ITA
    at all. Do not remove this check because a dataset without that tail
    never exercises it.

    Returns ``(t_lo, t_hi, ita_at_t_lo, ita_at_t_hi)`` with ``ita_at_t_lo >=
    ita_at_t_hi``, or ``None`` if even ``t = 0`` cannot anchor a stretch.
    """
    t_grid = np.linspace(t_min, t_max, n_grid)
    zero_index = int(np.argmin(np.abs(t_grid)))
    states = [_skin_state_at_t(skin_lab, float(t)) for t in t_grid]
    curve = np.array([state[0] for state in states])
    b_curve = np.array([state[1] for state in states])
    if not np.isfinite(curve[zero_index]):
        return None

    def _no_b_sign_crossing(i: int, j: int) -> bool:
        return np.sign(b_curve[i]) == np.sign(b_curve[j]) and np.sign(b_curve[i]) != 0.0

    lo = zero_index
    while (
        lo > 0
        and np.isfinite(curve[lo - 1])
        and curve[lo - 1] >= curve[lo] - _MONOTONE_TOLERANCE
        and _no_b_sign_crossing(lo - 1, lo)
    ):
        lo -= 1
    hi = zero_index
    while (
        hi < len(t_grid) - 1
        and np.isfinite(curve[hi + 1])
        and curve[hi + 1] <= curve[hi] + _MONOTONE_TOLERANCE
        and _no_b_sign_crossing(hi + 1, hi)
    ):
        hi += 1

    if lo == hi:
        return None
    return float(t_grid[lo]), float(t_grid[hi]), float(curve[lo]), float(curve[hi])


def solve_ita_delta(
    skin_lab: np.ndarray,
    seed: int,
    step: int,
    image_sha256: str,
    *,
    policy_cap_degrees: float | None = DEFAULT_POLICY_CAP_DEGREES,
    xtol: float = BRENTQ_XTOL,
) -> dict | None:
    """Reachable-range eq. 7: sample a target inside this image's own curve, then solve ``t`` exactly.

    ``policy_cap_degrees`` is an explicit, opt-in narrowing of the physically
    reachable bracket for visual-plausibility reasons; ``None`` (the default)
    applies no such cap. See the module docstring for why this is not
    ``tone_augment_lab.MAX_ABS_ITA_DELTA`` by default.

    Returns ``None`` when no monotonic stretch exists, or when a supplied
    ``policy_cap_degrees`` leaves no overlap with it -- the caller abstains in
    both cases with ``fallback_reason="no_reachable_ita_range"``.
    """
    bracket = _reachable_bracket(skin_lab)
    if bracket is None:
        return None
    t_lo, t_hi, ita_at_t_lo, ita_at_t_hi = bracket

    if policy_cap_degrees is None:
        t_lo_final, t_hi_final = t_lo, t_hi
    else:
        t_lo_final = max(t_lo, -policy_cap_degrees)
        t_hi_final = min(t_hi, policy_cap_degrees)
    if t_lo_final >= t_hi_final:
        return None

    ita_hi = _skin_ita_at_t(skin_lab, t_lo_final)  # smaller t -> larger ITA
    ita_lo = _skin_ita_at_t(skin_lab, t_hi_final)
    if not (ita_hi > ita_lo):
        return None

    raw = _hash_unit_interval("tone-augment-lab-v2-objetivo", seed, step, image_sha256)
    ita_objetivo = ita_lo + raw * (ita_hi - ita_lo)

    def _residual(t: float) -> float:
        return _skin_ita_at_t(skin_lab, t) - ita_objetivo

    t_solved = brentq(_residual, t_lo_final, t_hi_final, xtol=xtol)
    return {
        "ita_objetivo": float(ita_objetivo),
        "ita_delta": float(t_solved),
        "reachable_t_lo": t_lo,
        "reachable_t_hi": t_hi,
        "reachable_ita_lo": ita_at_t_lo,
        "reachable_ita_hi": ita_at_t_hi,
        "policy_cap_narrowed_reachable_range": (t_lo_final, t_hi_final) != (t_lo, t_hi),
        "population_shift_fraction": _population_shift_fraction(skin_lab, t_solved),
    }


def _empty_geometry_fields() -> dict:
    """Keys ``tone_augment_lab.summarize_augmentation_records`` expects.

    Every non-applied branch below returns this shape so the frozen module's
    aggregator can summarize v2 records without modification.
    """
    return {
        "ita_delta_requested": 0.0,
        "ita_delta_applied": 0.0,
        "ita_delta_capped": False,
        "feather_sigma": 0.0,
        "skin_fraction": 0.0,
        "lesion_fraction": 0.0,
        "skin_median_l_before": None,
        "skin_median_b_before": None,
        "delta_l_requested": 0.0,
        "delta_l_realized": 0.0,
        "delta_b_requested": 0.0,
        "delta_b_realized": 0.0,
        "gamut_clipped_pixels": 0,
        "skin_median_l_after": None,
        "skin_median_b_after": None,
        "b_sign_flipped_pixels": 0,
    }


def augment_skin_tone_v2(
    img: np.ndarray,
    lesion_mask: np.ndarray,
    seed: int,
    step: int,
    image_sha256: str,
    valid_fov: np.ndarray | None = None,
    *,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
    policy_cap_degrees: float | None = DEFAULT_POLICY_CAP_DEGREES,
    mix_threshold: float = MIX_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """Measure ITA with the mask, solve eq. 7's delta on this image's own reachable range, apply the frozen geometry.

    Five ways this returns identity (no pixel changed, ``applied is False``):

    * ITA is not measurable (``fitzpatrick.estimate_fst`` status other than
      ``"ok"``). ``fallback_reason="ita_unmeasured_<status>"``.
    * ``estimate_fst`` measured ITA via its own fallback mask rather than
      ``lesion_mask``, so this module cannot build a matching curve.
      ``fallback_reason="ita_measured_via_fallback_mask"``.
    * ITA is measurable but the 50/50 mixing draw did not select this image
      (checked before the reachable-range search, to skip that cost on the
      half of images it would discard anyway).
      ``fallback_reason="not_selected_by_mix"``.
    * ITA is measurable and the mixing draw selects this image, but no usable
      monotonic stretch exists for it (optionally narrowed further by
      ``policy_cap_degrees``). ``fallback_reason="no_reachable_ita_range"``.
    * A usable ``t`` was solved, but a large-enough fraction of the solving
      population would fall below the near-black border threshold once
      actually shifted (see ``_population_shift_fraction``) that a post-hoc
      re-measurement would land away from ``ita_objetivo``.
      ``fallback_reason="border_population_shift_too_large"``. ``ita_objetivo``
      and ``ita_delta_raw`` are still reported here, for diagnosis.

    The manifest always carries ``ita_measured``, ``ita_measurement_status``,
    ``ita_objetivo``, ``ita_delta_raw`` and the geometry record's
    ``ita_delta_applied``, ``None`` where a branch has no measurement or no
    solved delta to report.
    """
    img = _check_image_range(img)
    lesion_mask = _as_binary_mask(lesion_mask, "lesion_mask")
    fov = None if valid_fov is None else _as_binary_mask(valid_fov, "valid_fov")
    _validate_key(seed, step, image_sha256)

    # fitzpatrick.estimate_fst's border_mask is True=NOT-skin; valid_fov here
    # is True=IN-FIELD, the opposite polarity, matching how
    # tone_augment_lab._skin_selector treats the same argument.
    border_mask = None if fov is None else ~fov
    measurement = estimate_fst(img, mask=lesion_mask, border_mask=border_mask)

    base = {
        "ita_measured": measurement.ita,
        "ita_measurement_status": measurement.status,
        "ita_measurement_skin_fraction": measurement.skin_fraction,
    }

    def _abstain(reason: str) -> tuple[np.ndarray, dict]:
        return img.copy(), {
            **base,
            **_empty_geometry_fields(),
            "applied": False,
            "fallback_reason": reason,
            "ita_objetivo": None,
            "ita_delta_raw": None,
            "apply_draw": None,
            "selected_for_transform": False,
        }

    if measurement.status != "ok":
        return _abstain(f"ita_unmeasured_{measurement.status}")
    if measurement.fallback_used:
        return _abstain("ita_measured_via_fallback_mask")

    # Mix gate before the reachable-range search: cheap, and the search's
    # cost (a grid sweep plus a root-find) is wasted work on the half of
    # images the mix draw will not select anyway.
    draw = sample_apply_draw(seed, step, image_sha256)
    if draw <= mix_threshold:
        out, record = _abstain("not_selected_by_mix")
        record["apply_draw"] = draw
        return out, record

    skin_lab = _skin_lab_for_curve(img, lesion_mask, border_mask)
    if skin_lab is None:
        out, record = _abstain("no_reachable_ita_range")
        record["apply_draw"] = draw
        return out, record

    solved = solve_ita_delta(
        skin_lab, seed, step, image_sha256, policy_cap_degrees=policy_cap_degrees
    )
    if solved is None:
        out, record = _abstain("no_reachable_ita_range")
        record["apply_draw"] = draw
        return out, record

    # The curve's median was measured over a FIXED pre-transform population;
    # a large fraction of it falling below the border threshold at the
    # solved t means any post-hoc re-measurement (estimate_fst on the real
    # output) will drop those pixels and land away from ita_objetivo -- see
    # _population_shift_fraction. Abstain rather than report an unreachable
    # promise.
    if solved["population_shift_fraction"] > MAX_POPULATION_SHIFT_FRACTION:
        out, record = _abstain("border_population_shift_too_large")
        record["apply_draw"] = draw
        record["ita_objetivo"] = solved["ita_objetivo"]
        record["ita_delta_raw"] = solved["ita_delta"]
        return out, record

    # The bound passed here is whatever bracket the delta was actually solved
    # within, never a separate number: augment_skin_tone's own clip must not
    # re-truncate a value this module already guaranteed is in range.
    geometry_bound = max(abs(solved["reachable_t_lo"]), abs(solved["reachable_t_hi"]))
    # _unified_skin_fov, not the raw fov: makes augment_skin_tone's internal
    # _skin_selector shift exactly the population this module solved and
    # measured over, not the (possibly wider) raw field of view.
    out, geometry_record = augment_skin_tone(
        img,
        lesion_mask,
        _unified_skin_fov(img, border_mask),
        ita_delta=solved["ita_delta"],
        feather_sigma=feather_sigma,
        max_abs_ita_delta=geometry_bound,
    )
    return out, {
        **base,
        **geometry_record,
        "ita_objetivo": solved["ita_objetivo"],
        "ita_delta_raw": solved["ita_delta"],
        "reachable_t_lo": solved["reachable_t_lo"],
        "reachable_t_hi": solved["reachable_t_hi"],
        "policy_cap_narrowed_reachable_range": solved["policy_cap_narrowed_reachable_range"],
        "population_shift_fraction": solved["population_shift_fraction"],
        "apply_draw": draw,
        "selected_for_transform": True,
    }
