"""Own deterministic two-level structural-entropy solver (paper-based adaptation)."""
from __future__ import annotations
import heapq
import numpy as np
from dermssl.w3_components import DegenerateCase


def _graph(adjacency):
    a = np.asarray(adjacency, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Finite square nonnegative adjacency required')
    if not np.allclose(a, a.T) or not np.allclose(np.diag(a), 0):
        raise ValueError('Undirected adjacency without self-edges required')
    if a.sum() <= 0:
        raise DegenerateCase('Structural entropy is undefined for zero graph volume')
    return a


def _benefit(degree, internal, volume):
    degree = np.asarray(degree)
    ratio = np.maximum(degree / volume, np.finfo(float).tiny)
    return -np.asarray(internal) / volume * np.log2(ratio)


def structural_entropy(adjacency, labels):
    """H2 = H1 - sum_c [-internal_c/V * log2(volume_c/V)]."""
    a = _graph(adjacency)
    labels = np.asarray(labels)
    if labels.shape != (len(a),) or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError('Nonnegative integer partition required')
    degree = a.sum(axis=1); volume = degree.sum()
    positive = degree > 0
    h1 = -(degree[positive] / volume * np.log2(degree[positive] / volume)).sum()
    benefit = 0.
    for c in np.unique(labels):
        idx = labels == c
        benefit += float(_benefit(degree[idx].sum(), a[np.ix_(idx, idx)].sum(), volume))
    return float(h1 - benefit)


def merge_and_refine(adjacency, *, max_refinement_passes=30, tolerance=1e-10):
    """Greedy positive entropy-benefit merges, then improving node relocations.

    Heap merging avoids a full O(n^3) pair search. This is a Python solver of the
    paper objective, not a claimed bitwise port of the author's Java tie rules.
    """
    a = _graph(adjacency); n = len(a); volume = a.sum()
    if type(max_refinement_passes) is not int or max_refinement_passes < 0 or tolerance < 0:
        raise ValueError('Nonnegative refinement budget and tolerance required')
    degree = a.sum(1); volumes = degree.copy(); inside = np.zeros(n)
    edges = [{int(j): float(a[i, j]) for j in np.flatnonzero(a[i])} for i in range(n)]
    active = set(range(n)); versions = np.zeros(n, dtype=int); members = [{i} for i in range(n)]
    labels = np.arange(n); queue = []
    def push(i, j):
        i, j = sorted((i, j)); w = edges[i].get(j, 0.)
        gain = float(_benefit(volumes[i]+volumes[j], inside[i]+inside[j]+2*w, volume)
                     - _benefit(volumes[i], inside[i], volume) - _benefit(volumes[j], inside[j], volume))
        if gain > tolerance:
            heapq.heappush(queue, (-gain, i, j, int(versions[i]), int(versions[j])))
    for i in range(n):
        for j in edges[i]:
            if i < j: push(i, j)
    merges = 0
    while queue:
        _, i, j, vi, vj = heapq.heappop(queue)
        if i not in active or j not in active or versions[i] != vi or versions[j] != vj:
            continue
        inside[i] += inside[j] + 2 * edges[i].get(j, 0.)
        volumes[i] += volumes[j]; active.remove(j); versions[i] += 1; versions[j] += 1
        for node in members[j]: labels[node] = i
        members[i].update(members[j]); members[j].clear()
        for k, w in list(edges[j].items()):
            if k == i: continue
            edges[i][k] = edges[i].get(k, 0.) + w
            edges[k][i] = edges[i][k]; edges[k].pop(j, None)
        edges[i].pop(j, None); edges[j].clear()
        for k in sorted(edges[i]):
            if k in active: push(i, k)
        merges += 1
    _, labels = np.unique(labels, return_inverse=True)
    trace = [structural_entropy(a, labels)]
    k = int(labels.max()) + 1
    vols = np.bincount(labels, weights=degree, minlength=k)
    internals = np.array([a[np.ix_(labels == c, labels == c)].sum() for c in range(k)])
    for _ in range(max_refinement_passes):
        moves = 0
        for node in range(n):
            old = labels[node]; d = degree[node]
            if d == 0: continue
            links = np.bincount(labels, weights=a[node], minlength=k)
            base = _benefit(vols, internals, volume)
            removed = float(_benefit(max(0., vols[old]-d), max(0., internals[old]-2*links[old]), volume))
            gains = (_benefit(vols+d, internals+2*links, volume) - base) + removed - base[old]
            gains[old] = 0.
            best = int(np.argmax(gains))
            if gains[best] > tolerance:
                vols[old] -= d; vols[best] += d
                internals[old] -= 2*links[old]; internals[best] += 2*links[best]
                internals = np.maximum(internals, 0.)
                labels[node] = best; moves += 1
        current = structural_entropy(a, labels)
        if current > trace[-1] + 1e-8:
            raise RuntimeError('Structural entropy increased unexpectedly')
        trace.append(current)
        if not moves: break
    _, labels = np.unique(labels, return_inverse=True)
    return labels, {'merges': merges, 'entropy_trace': trace, 'solver': 'python_h2_greedy_relocation'}
