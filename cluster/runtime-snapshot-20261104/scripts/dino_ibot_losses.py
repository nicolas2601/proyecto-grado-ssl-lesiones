"""Pure DINO/iBOT helpers; the torch training wiring lives in the runtime, not here."""


def ema_update(teacher, student, momentum):
    return [momentum * t + (1 - momentum) * s for t, s in zip(teacher, student)]


def dino_pair_mean(loss, teacher_views, student_views):
    count = len(teacher_views)
    if count < 2 or count != len(student_views):
        raise ValueError('DINO needs at least two views with matching teacher and student counts')
    pairs = [(i, j) for i in range(count) for j in range(count) if i != j]
    return sum(loss(teacher_views[i], student_views[j]) for i, j in pairs) / len(pairs)
