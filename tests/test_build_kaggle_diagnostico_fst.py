"""build_kaggle_diagnostico_fst debe generar, POR DATASET, un script que
REALMENTE corre fuera de este repositorio (sin poder importar dermssl) - el
mismo tipo de bug que rompio el notebook de entrenamiento en Kaggle real
(PR2: "inlining fitzpatrick without resolving its dermssl.pretrain
import"), pero detectado aca en CI en vez de en una sesion de Kaggle."""

import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import build_kaggle_diagnostico_fst as bkd  # noqa: E402
from dermssl import data_resolver  # noqa: E402


def test_datasets_come_from_data_resolver_not_redefined():
    """Los tres dataset_id y sus slugs de Kaggle tienen que salir de
    dermssl.data_resolver.DATASET_HANDLES (la fuente de verdad que tambien
    usa el notebook real de Nico) - no un diccionario propio que pueda
    desincronizarse, y que en su momento tuvo el slug de ISIC 2019 en
    resolucion completa en vez del resized."""
    assert bkd.DATASETS == tuple(data_resolver.DATASET_HANDLES)
    assert data_resolver.DATASET_HANDLES["isic2019"] == "nischaydnk/isic-2019-jpg-224x224-resized"


@pytest.mark.parametrize("dataset_id", bkd.DATASETS)
def test_metadata_is_private_cpu_only_with_exactly_that_datasets_source(dataset_id):
    m = bkd.metadata("mariapaus", dataset_id)
    assert m["id"] == f"mariapaus/diagnostico-fst-{dataset_id}"
    assert m["kernel_type"] == "script"
    assert m["is_private"] is True
    assert m["enable_gpu"] is False
    assert m["dataset_sources"] == [data_resolver.DATASET_HANDLES[dataset_id]]


def test_metadata_rejects_an_unknown_dataset_id():
    with pytest.raises(ValueError):
        bkd.build_script("no_existe")


@pytest.mark.parametrize("dataset_id", bkd.DATASETS)
def test_build_script_never_imports_dermssl(dataset_id):
    """El script generado corre en un kernel de Kaggle sin este
    repositorio - cualquier ``import dermssl`` que sobreviva al inlinado
    rompe la corrida real (esto es exactamente lo que paso en PR2)."""
    source = bkd.build_script(dataset_id)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("dermssl") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("dermssl")


@pytest.mark.parametrize("dataset_id", bkd.DATASETS)
def test_build_script_compiles(dataset_id):
    compile(bkd.build_script(dataset_id), "<diagnostico_fst>", "exec")


def test_build_script_inlines_only_the_two_lesion_guided_functions_it_needs():
    """El resto de lesion_guided.py (border_ring, sample_centers,
    guided_crop_centers, crop_slices - el sampler multi-crop de SSL v6, sin
    relacion con FST) NO debe terminar en el kernel. Esto es justamente lo
    que se corrigio: antes se inlinaba el archivo entero."""
    source = bkd.build_script("ham10000")
    assert "def otsu_threshold" in source
    assert "def lesion_mask" in source
    for unrelated in ("border_ring", "sample_centers", "guided_crop_centers", "crop_slices", "DINO"):
        assert unrelated not in source


def test_real_flag_sets_smoke_false_in_the_generated_source():
    smoke_source = bkd.build_script("ham10000", real=False)
    real_source = bkd.build_script("ham10000", real=True)
    assert "SMOKE = True" in smoke_source
    assert "SMOKE = False" in real_source
    assert "SMOKE = True" not in real_source


def _write_disk_image(path: Path, skin_rgb, lesion_rgb=(0.30, 0.20, 0.10), size=64, radius=15):
    """Disco de lesion sobre fondo de piel - una imagen de color plano no
    sirve aca: Otsu no puede encontrar un split bimodal en una imagen
    uniforme (degenerate_mask, correctamente documentado en
    test_fitzpatrick.py), y este test quiere ambas corridas en "ok"."""
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.empty((size, size, 3))
    img[:] = skin_rgb
    img[dist <= radius] = lesion_rgb
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(path)


@pytest.mark.parametrize("dataset_id", bkd.DATASETS)
def test_generated_script_runs_end_to_end_against_synthetic_kaggle_input(tmp_path, dataset_id):
    """El test de mas valor de este archivo: ejecuta el script generado de
    UN dataset tal cual correria en Kaggle (sin sys.path a este repo, con
    un solo dataset montado bajo /kaggle/input, como pushea el builder
    ahora), y verifica que escribe su CSV con filas reales. Si el inlinado
    de lesion_guided, data_resolver.index_images o fitzpatrick.py se
    rompe, esto falla aca - no en una sesion de Kaggle."""
    pytest.importorskip("PIL")
    kaggle_input = tmp_path / "kaggle_input"
    kaggle_working = tmp_path / "kaggle_working"
    kaggle_working.mkdir()

    # Anidado como el montaje real de Kaggle (/kaggle/input/datasets/<owner>/<slug>/...),
    # asi el test tambien cubre que index_images encuentre las imagenes ahi.
    folder = kaggle_input / "datasets" / "owner" / data_resolver.DATASET_HANDLES[dataset_id].split("/")[1] / "images"
    folder.mkdir(parents=True)
    for i in range(3):
        _write_disk_image(folder / f"img_{i}.jpg", (0.6, 0.4, 0.3))

    source = bkd.build_script(dataset_id, real=False)
    patched = (
        source
        .replace('Path("/kaggle/input")', f"Path({str(kaggle_input)!r})")
        .replace('Path("/kaggle/working")', f"Path({str(kaggle_working)!r})")
    )

    exec(compile(patched, "<kaggle_kernel_under_test>", "exec"), {})

    out_csv = kaggle_working / f"diagnostico_fst_{dataset_id}.csv"
    assert out_csv.is_file()
    with out_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        assert row["dataset_id"] == dataset_id
        assert row["status_sin_mascara"] == "ok"
        assert row["status_con_mascara"] == "ok"
        assert row["bucket_sin_mascara"] != ""
        assert row["bucket_con_mascara"] != ""

    out_json = kaggle_working / f"diagnostico_fst_{dataset_id}.json"
    assert out_json.is_file()
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["dataset"] == dataset_id
    assert summary["n"] == 3
    assert summary["smoke"] is True


def test_generated_script_exits_cleanly_when_no_images_are_found(tmp_path):
    kaggle_input = tmp_path / "kaggle_input"
    kaggle_input.mkdir()
    kaggle_working = tmp_path / "kaggle_working"
    kaggle_working.mkdir()

    source = bkd.build_script("ham10000", real=False)
    patched = (
        source
        .replace('Path("/kaggle/input")', f"Path({str(kaggle_input)!r})")
        .replace('Path("/kaggle/working")', f"Path({str(kaggle_working)!r})")
    )

    with pytest.raises(SystemExit):
        exec(compile(patched, "<kaggle_kernel_under_test>", "exec"), {})
