"""Fixed-threshold GPU cosine candidates; similarity does not prove patient identity."""
import numpy as np


def cosine_edges(query, reference, threshold=.995, same_collection=False, limit=250000, device="cuda:0"):
    import torch
    import torch.nn.functional as functional
    query, reference = np.asarray(query, dtype=np.float32), np.asarray(reference, dtype=np.float32)
    if (query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]
            or not query.shape[0] or not reference.shape[0] or not query.shape[1]
            or not np.isfinite(query).all() or not np.isfinite(reference).all()
            or np.any(np.linalg.norm(query, axis=1) == 0)
            or np.any(np.linalg.norm(reference, axis=1) == 0)):
        raise ValueError("Expected aligned finite nonzero embedding matrices")
    if not -1 <= threshold <= 1 or limit < 1:
        raise ValueError("Invalid cosine threshold or candidate limit")
    if same_collection and not np.array_equal(query, reference):
        raise ValueError("Self-audit requires identical ordered embedding matrices")
    edges = []
    with torch.inference_mode():
        q = functional.normalize(torch.as_tensor(query, device=device), dim=1)
        r = functional.normalize(torch.as_tensor(reference, device=device), dim=1)
        for start in range(0, len(query), 128):
            scores = (q[start:start + 128] @ r.T).clamp(-1, 1)
            positions = torch.nonzero(scores >= threshold).cpu().tolist()
            for row, column in positions:
                index = start + row
                if same_collection and column <= index:
                    continue
                edges.append({"query_index": index, "reference_index": column,
                              "cosine": float(scores[row, column])})
                if len(edges) > limit:
                    raise ValueError("Cosine candidate limit exceeded; audit incomplete")
    return edges
