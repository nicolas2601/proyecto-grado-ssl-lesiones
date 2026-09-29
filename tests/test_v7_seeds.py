"""The v7 seeds live in one registry; every generator reads it.

s44 joined ``REGISTERED_SEEDS`` with its committed final_eval.json (docs/29
2.5); every generated table is regenerated from it and the freshness tests
check them. v7.1 is a closed selection over the s42+s43 soup (56e7670) and
keeps its own fixed pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dermssl import v7_seeds as vs  # noqa: E402

V7 = Path("resultados") / "clasificacion" / "v7"


def test_the_registry_holds_the_evaluated_seeds_in_order():
    assert vs.REGISTERED_SEEDS == (42, 43, 44)
    assert vs.BASE_SEED == vs.REGISTERED_SEEDS[0]


def test_v71_keeps_its_own_pair_whatever_the_registry_says():
    assert vs.V71_SOUP_SEEDS == (42, 43)


def test_paths_follow_the_committed_layout():
    assert vs.final_eval_path(42) == V7 / "final_eval.json"
    assert vs.final_eval_path(44) == V7 / "s44" / "final_eval.json"
    assert vs.probas_path(43) == V7 / "s43" / "final_eval.probas.npz"


def test_every_registered_seed_has_its_committed_evaluation():
    for seed in vs.REGISTERED_SEEDS:
        assert (ROOT / vs.final_eval_path(seed)).is_file(), seed
        assert (ROOT / vs.probas_path(seed)).is_file(), seed


def test_seed_44_is_registered_with_its_evaluation_probas_and_environment():
    from dermssl import run_environment  # noqa: PLC0415

    assert 44 in vs.REGISTERED_SEEDS
    final = ROOT / vs.final_eval_path(44)
    assert final.is_file() and (ROOT / vs.probas_path(44)).is_file()
    assert run_environment.sidecar_problems(run_environment.sidecar_path(final)) == []


def test_final_dirs_and_model_names():
    assert vs.final_dir_name("A", 42) == "final_A"
    assert vs.final_dir_name("B", 44) == "final_B_s44"
    assert vs.model_names(("A", "B"), (42, 43)) == ("A_s42", "A_s43", "B_s42", "B_s43")


@pytest.mark.parametrize(("seeds", "text"), [((42, 43), "dos"), ((42, 43, 44), "tres")])
def test_seed_count_in_words(seeds, text):
    assert vs.count_word(len(seeds)) == text


@pytest.mark.parametrize(("items", "text"), [(["a"], "a"), (["a", "b"], "a y b"), (["a", "b", "c"], "a, b y c")])
def test_spanish_join(items, text):
    assert vs.join_es(items) == text


def test_the_v71_final_dirs_come_from_its_fixed_pair():
    import v71_io  # noqa: PLC0415

    assert v71_io.FINAL_DIRS == {("A", "s42"): "final_A", ("A", "s43"): "final_A_s43",
                                 ("B", "s42"): "final_B", ("B", "s43"): "final_B_s43"}


def test_release_models_and_copies_follow_the_registry():
    import finalize_release as fr  # noqa: PLC0415

    assert fr.MODELS == ("A_s42", "A_s43", "A_s44", "B_s42", "B_s43", "B_s44")
    assert fr.COPIES["results/s43/final_eval.json"] == ROOT / V7 / "s43" / "final_eval.json"
    assert fr.COPIES["results/s44/final_eval.probas.npz"] == ROOT / V7 / "s44" / "final_eval.probas.npz"
