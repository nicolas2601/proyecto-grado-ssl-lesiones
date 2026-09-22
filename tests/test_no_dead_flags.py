"""A constant nobody reads is worse than no constant at all.

CONV_GAP_FROM_TIMM declared which backbones must be read at the conv GAP and
nothing consulted it: both runners branched on a literal name. Adding a
backbone to the list would have changed no behaviour, and the model would
have been read at the wrong depth producing a plausible number - the exact
failure the list was written to prevent, after that failure had already
happened once.

That is the worst shape of the bug: a defence written in response to an
incident, never wired, which also closes the subject. Nobody looks there
again because it was "fixed".

The audit that found it was cheap - list module constants, count their
references - so it runs here instead of depending on someone remembering.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCES = sorted(REPO.glob("scripts/*.py")) + sorted((REPO / "src" / "dermssl").rglob("*.py"))

# Constants that exist to be read from outside this tree: the Kaggle
# notebooks and the SLURM scripts import them, so a single in-tree reference
# is expected. Each entry needs a reason; an empty allowlist is the goal.
ALLOWED_SINGLE_USE = {
    "SIMSIAM_CHECKPOINT_URL": "la descarga la hace el notebook generado, no el paquete",
}

CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*[:=]", re.M)


def _corpus() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in SOURCES)


def test_every_declared_constant_is_read_somewhere():
    corpus = _corpus()
    dead = []
    for path in SOURCES:
        for match in CONSTANT.finditer(path.read_text(encoding="utf-8")):
            name = match.group(1)
            if name in ALLOWED_SINGLE_USE:
                continue
            if len(re.findall(rf"\b{name}\b", corpus)) <= 1:
                dead.append(f"{path.relative_to(REPO)}: {name}")

    assert not dead, (
        "constantes declaradas que nadie lee; parecen gobernar algo y no gobiernan "
        "nada:\n  " + "\n  ".join(sorted(dead))
    )


def test_the_conv_gap_list_governs_both_readers():
    """The specific flag that was dead, pinned so it cannot go back."""
    corpus = _corpus()
    assert len(re.findall(r"\bCONV_GAP_FROM_TIMM\b", corpus)) >= 2, (
        "CONV_GAP_FROM_TIMM volvio a ser decorativa"
    )


def test_the_allowlist_entries_still_exist():
    """An allowlist that outlives its constants hides the next dead flag."""
    corpus = _corpus()
    stale = [name for name in ALLOWED_SINGLE_USE if name not in corpus]
    assert not stale, f"la lista permite constantes que ya no existen: {stale}"
