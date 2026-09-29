"""The environment block every evaluation JSON embeds (dermssl.run_environment).

No GPU and no torch needed: torch is injected as a fake module object.
"""

import json
from types import SimpleNamespace

from dermssl import run_environment as re_


def _fake_torch(*, cuda_available=True, initialized=True, cudnn_available=True):
    cudnn = SimpleNamespace(deterministic=True, benchmark=False, is_available=lambda: cudnn_available,
                            version=lambda: 90100)
    cuda = SimpleNamespace(is_available=lambda: cuda_available, is_initialized=lambda: initialized,
                           device_count=lambda: 2, get_device_name=lambda i: f"Tesla T4 #{i}")
    return SimpleNamespace(__version__="2.4.1+cu121", version=SimpleNamespace(cuda="12.1"),
                           backends=SimpleNamespace(cudnn=cudnn), cuda=cuda,
                           are_deterministic_algorithms_enabled=lambda: True, get_num_threads=lambda: 4)


def _snap(modules, **kw):
    kw.setdefault("environ", {})
    return re_.snapshot(modules=modules, cpu_count=8, pools=[], **kw)


def test_the_block_names_every_required_field():
    env = _snap({"torch": _fake_torch()})
    assert set(env) == {"schema", "python", "packages", "torch", "env", "threadpools", "cpu_count", "knobs"}
    assert set(env["packages"]) >= {"numpy", "scipy", "scikit-learn", "torch", "torchvision"}
    assert set(env["env"]) >= {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES"}
    assert env["cpu_count"] == 8 and env["python"].count(".") == 2


def test_a_loaded_torch_reports_cuda_cudnn_determinism_threads_and_gpus():
    state = _snap({"torch": _fake_torch()})["torch"]
    assert state == {
        "loaded": True, "version": "2.4.1+cu121", "cuda_runtime": "12.1", "cudnn": 90100,
        "cudnn_deterministic": True, "cudnn_benchmark": False, "deterministic_algorithms": True,
        "num_threads": 4, "cuda_available": True, "cuda_initialized": True,
        "gpus": [{"index": 0, "name": "Tesla T4 #0"}, {"index": 1, "name": "Tesla T4 #1"}]}


def test_gpu_names_are_not_read_unless_cuda_is_already_initialised():
    state = _snap({"torch": _fake_torch(initialized=False)})["torch"]
    assert state["cuda_available"] is True and state["cuda_initialized"] is False and state["gpus"] == []


def test_a_cpu_torch_build_reports_no_cudnn():
    state = _snap({"torch": _fake_torch(cuda_available=False, cudnn_available=False)})["torch"]
    assert state["cudnn"] is None and state["gpus"] == [] and state["cuda_available"] is False


def test_torch_not_loaded_by_the_script_is_reported_and_never_imported():
    modules = {}
    assert _snap(modules)["torch"] == {"loaded": False}
    assert "torch" not in modules


def test_a_loaded_module_version_wins_over_the_installed_one():
    versions = re_.package_versions({"numpy": SimpleNamespace(__version__="9.9.9")})
    assert versions["numpy"] == "9.9.9"


def test_an_absent_package_is_none_not_an_error(monkeypatch):
    monkeypatch.setitem(re_.PACKAGES, "no-such-dist-xyz", "no_such_module_xyz")
    assert re_.package_versions({})["no-such-dist-xyz"] is None


def test_thread_env_vars_are_copied_and_missing_ones_are_none():
    env = _snap({}, environ={"OMP_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})["env"]
    assert env["OMP_NUM_THREADS"] == "1" and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["MKL_NUM_THREADS"] is None


def test_knobs_are_the_parallelism_flags_the_script_defines_plus_extras():
    args = SimpleNamespace(probe_jobs=6, boot_jobs=12, decode_workers=4, seed=0)
    env = _snap({}, args=args, loky_inner_threads=2)
    assert env["knobs"] == {"probe_jobs": 6, "boot_jobs": 12, "decode_workers": 4, "loky_inner_threads": 2}


def test_no_args_means_no_knobs():
    assert _snap({})["knobs"] == {}


def test_real_threadpools_are_listed_with_their_thread_counts():
    import numpy  # noqa: F401 - loads a BLAS so there is at least one pool

    pools = re_.threadpools()
    assert pools and all(set(p) == set(re_.THREADPOOL_KEYS) for p in pools)


def test_the_default_snapshot_is_strict_json():
    json.dumps(re_.snapshot(), allow_nan=False)
