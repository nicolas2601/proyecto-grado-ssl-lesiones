"""Test helper: how many BLAS threads the probe's numpy/scipy would run with, right now."""

NUMPY_BLAS_PREFIXES = ("libscipy_openblas", "libmkl")


def blas_threads(pools=None) -> int:
    """Threads of the BLAS numpy and scipy link against; -1 if it is absent or not uniform.

    Not simply the first BLAS pool: opencv bundles a single-threaded OpenBLAS that no
    limit can raise and the probe never calls, and a worker may have loaded it first.
    """
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    pools = threadpool_info() if pools is None else pools
    ours = {pool["num_threads"] for pool in pools if pool["user_api"] == "blas"
            and pool["filepath"].rsplit("/", 1)[-1].startswith(NUMPY_BLAS_PREFIXES)}
    return ours.pop() if len(ours) == 1 else -1
