"""Label-free image sampling weights for representation-space coverage."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


_CLUSTER_SEED = 1337
_MAX_CLUSTERS = 64
_MIN_WEIGHT = 0.25
_MAX_WEIGHT = 4.0
_COVERAGE_MIX = 0.5


def coverage_probabilities(features, component_ids, domains) -> np.ndarray:
    """Return global weights mixing uniform image sampling and per-source coverage.

    Components represent image identities (e.g. original/transformed views),
    while domains are non-diagnostic source labels such as HAM/PAD.
    """
    vectors = np.asarray(features, dtype=np.float64)
    components = np.asarray(component_ids, dtype=object)
    sources = np.asarray(domains, dtype=object)
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError("features must be a non-empty 2D matrix")
    n = vectors.shape[0]
    if components.shape != (n,) or sources.shape != (n,):
        raise ValueError("features, component_ids, and domains must have equal lengths")
    if not np.isfinite(vectors).all():
        raise ValueError("features must be finite")
    if any(_is_missing(value) for value in components):
        raise ValueError("component_ids must not be empty")
    if any(_is_missing(value) for value in sources):
        raise ValueError("domains must not be empty")

    component_sources = {}
    component_rows = {}
    for index, component in enumerate(components.tolist()):
        component_sources.setdefault(component, set()).add(sources[index])
        component_rows.setdefault(component, []).append(index)
    if any(len(values) != 1 for values in component_sources.values()):
        raise ValueError("each component must belong to exactly one domain")

    means, mean_components, mean_domains = [], [], []
    for component, rows in component_rows.items():
        component_mean = vectors[rows].mean(axis=0)
        norm = np.linalg.norm(component_mean)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("component mean must have a non-zero finite norm")
        means.append(component_mean / norm)
        mean_components.append(component)
        mean_domains.append(next(iter(component_sources[component])))
    means = np.asarray(means, dtype=np.float64)
    mean_components = np.asarray(mean_components, dtype=object)
    mean_domains = np.asarray(mean_domains, dtype=object)

    coverage = np.zeros(n, dtype=np.float64)
    domain_values = sorted(set(mean_domains.tolist()), key=str)
    for domain in domain_values:
        selected = np.flatnonzero(mean_domains == domain)
        domain_features = means[selected]
        domain_components = mean_components[selected]
        cluster_count = min(_MAX_CLUSTERS, len(selected))
        cluster_ids = KMeans(
            n_clusters=cluster_count,
            random_state=_CLUSTER_SEED,
            n_init=1,
            max_iter=100,
        ).fit_predict(domain_features)
        occupancies = np.bincount(cluster_ids, minlength=cluster_count).astype(np.float64)
        median_occupancy = float(np.median(occupancies[occupancies > 0]))
        cluster_weights = np.clip(
            np.sqrt(median_occupancy / occupancies), _MIN_WEIGHT, _MAX_WEIGHT
        )
        component_weights = cluster_weights[cluster_ids]
        component_weights /= component_weights.sum()
        for local_index, component_index in enumerate(selected):
            rows = component_rows[mean_components[component_index]]
            coverage[rows] = component_weights[local_index] / len(rows)

    for domain in domain_values:
        rows = np.flatnonzero(sources == domain)
        domain_total = coverage[rows].sum()
        if domain_total <= 0 or not np.isfinite(domain_total):
            raise ValueError("each domain must have positive finite coverage weight")
        coverage[rows] = 0.5 / len(rows) + 0.5 * coverage[rows] / domain_total
        coverage[rows] /= len(domain_values)
    return coverage


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False
