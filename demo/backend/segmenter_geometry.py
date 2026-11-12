"""Mask geometry for the lesion segmenter: largest component, outer contour, simplification, PNG I/O.

Pure numpy + stdlib (no OpenCV/scipy/skimage in the image). Points are (x, y) pixel coordinates.
"""

from __future__ import annotations

import io
from collections import deque

import numpy as np
from PIL import Image

MAX_CONTOUR_POINTS = 256
MIN_SIMPLIFY_EPSILON = 0.5
MAX_SIMPLIFY_EPSILON = 4096.0
PNG_THRESHOLD = 128
_FOUR_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBOURS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))  # clockwise from north


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 4-connected foreground component (all False when there is none)."""
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    best_label, best_size, current = 0, 0, 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if labels[y0, x0]:
            continue
        current += 1
        labels[y0, x0] = current
        queue: deque[tuple[int, int]] = deque([(int(y0), int(x0))])
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy, dx in _FOUR_NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if _is_foreground(mask, ny, nx) and not labels[ny, nx]:
                    labels[ny, nx] = current
                    queue.append((ny, nx))
        if size > best_size:
            best_label, best_size = current, size
    if best_label == 0:
        return np.zeros_like(mask)
    return labels == best_label


def _is_foreground(mask: np.ndarray, y: int, x: int) -> bool:
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def outer_contour(mask: np.ndarray) -> list[tuple[int, int]]:
    """Outer boundary of the foreground blob that holds the top-left-most pixel (Moore-neighbour tracing)."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    start = (int(ys[0]), int(xs[0]))
    cur = start
    back = (start[0], start[1] - 1)  # west of the top-left-most pixel is background
    trace = [start]
    for _ in range(8 * mask.size + 8):
        k = _NEIGHBOURS.index((back[0] - cur[0], back[1] - cur[1]))
        found, back_step = None, None
        for step in range(1, 9):
            dy, dx = _NEIGHBOURS[(k + step) % 8]
            if _is_foreground(mask, cur[0] + dy, cur[1] + dx):
                found = (cur[0] + dy, cur[1] + dx)
                back_step = _NEIGHBOURS[(k + step - 1) % 8]
                break
        if found is None:
            break
        back = (cur[0] + back_step[0], cur[1] + back_step[1])
        cur = found
        if cur == start:
            break
        trace.append(cur)
    return [(x, y) for y, x in trace]


def _point_segment_distance(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


def douglas_peucker(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Open-polyline Douglas-Peucker; the first and last points are always kept."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        distances = [_point_segment_distance(points[k], points[i], points[j]) for k in range(i + 1, j)]
        far = int(np.argmax(distances))
        if distances[far] > epsilon:
            index = i + 1 + far
            keep[index] = True
            stack.append((i, index))
            stack.append((index, j))
    return [point for point, kept in zip(points, keep) if kept]


def simplify_contour(points: list[tuple[int, int]], max_points: int = MAX_CONTOUR_POINTS) -> list[tuple[int, int]]:
    """Closed-contour simplification: grow the tolerance until at most ``max_points`` remain."""
    if len(points) <= 3:
        return list(points)
    closed = [*points, points[0]]
    epsilon = MIN_SIMPLIFY_EPSILON
    simplified = douglas_peucker(closed, epsilon)
    while len(simplified) - 1 > max_points and epsilon < MAX_SIMPLIFY_EPSILON:
        epsilon *= 2.0
        simplified = douglas_peucker(closed, epsilon)
    return simplified[:-1]


def encode_mask_png(mask: np.ndarray) -> bytes:
    """8-bit grayscale PNG, 255 = foreground, 0 = background."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def decode_mask_png(payload: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(payload)).convert("L")) >= PNG_THRESHOLD
