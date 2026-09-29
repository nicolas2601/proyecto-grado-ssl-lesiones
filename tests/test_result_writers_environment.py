"""Every script that writes an evaluation/result JSON embeds the environment block.

v7.1 recorded no versions or threads, and a T4 reproduction drift of ~1e-3 could
not be attributed. The registry below is the contract: a script is either WIRED
(it calls ``run_environment.snapshot`` for the JSON it writes) or EXEMPT with a
reason. A new JSON-writing script that is in neither fails the discovery test.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# script -> the JSON(s) whose top level carries ``environment``.
WIRED = {
    "scripts/v71_run.py": "units/*/result.json, holdout_grid.json, pretest.json",
    "scripts/v71_val.py": "selection.json, v71_final_eval*.json",
    "scripts/t4_sog_run.py": "t4_holdout_grid.json (via the shared block, was a local copy)",
    "scripts/t4_sog_val.py": "t4_selection.json, t4_final_eval*.json",
    "scripts/t4_repro_compare.py": "the comparison JSON (plus a_/b_environment of the runs)",
    "scripts/v71_extra_contrasts.py": "extra_contrasts.json",
}


def _calls_snapshot(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "snapshot" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "run_environment"):
            return True
    return False


@pytest.mark.parametrize("script", sorted(WIRED))
def test_every_wired_script_embeds_the_environment_block(script):
    assert _calls_snapshot(ROOT / script), f"{script} writes a result JSON without run_environment.snapshot()"
