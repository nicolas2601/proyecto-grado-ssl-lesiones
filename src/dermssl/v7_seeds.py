"""The one registry of v7 pretraining seeds, and where each one's files live.

A seed is registered once its single evaluation (``final_eval.json`` and the
probabilities beside it) is committed. Every generator of tables, the model
card and the release reads this, so adding a seed is one line here plus the
regenerated outputs. v7.1 is not part of it: its selection closed over the
s42+s43 soup (56e7670) and keeps that pair fixed.
"""

from __future__ import annotations

from pathlib import Path

REGISTERED_SEEDS: tuple[int, ...] = (42, 43, 44)
BASE_SEED = REGISTERED_SEEDS[0]  # Its files and folders carry no seed suffix.
V71_SOUP_SEEDS: tuple[int, ...] = (42, 43)  # Frozen: the v7.1 selection used this soup.

V7_RESULTS = Path("resultados") / "clasificacion" / "v7"
_WORDS = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}


def _folder(seed: int) -> Path:
    return V7_RESULTS if seed == BASE_SEED else V7_RESULTS / f"s{seed}"


def final_eval_path(seed: int) -> Path:
    """Repo-relative ``final_eval.json`` of one seed."""
    return _folder(seed) / "final_eval.json"


def probas_path(seed: int) -> Path:
    return _folder(seed) / "final_eval.probas.npz"


def final_dir_name(arm: str, seed: int) -> str:
    """Folder of a final under ``salidas/v7`` on the pod."""
    return f"final_{arm}" if seed == BASE_SEED else f"final_{arm}_s{seed}"


def model_names(arms, seeds=REGISTERED_SEEDS) -> tuple[str, ...]:
    return tuple(f"{arm}_s{seed}" for arm in arms for seed in seeds)


def count_word(n: int) -> str:
    return _WORDS.get(n, str(n))


def join_es(items) -> str:
    """``a``, ``a y b``, ``a, b y c``."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]
