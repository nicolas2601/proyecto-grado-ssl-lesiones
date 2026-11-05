"""Pure DINO/iBOT/KoLeo objectives on plain lists; torch wiring stays in the runtime."""
import math


def softmax(values):
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(values):
    peak = max(values)
    log_total = peak + math.log(sum(math.exp(v - peak) for v in values))
    return [v - log_total for v in values]


def teacher_probs(logits, center, temp):
    return softmax([(l - c) / temp for l, c in zip(logits, center)])


def update_center(center, batch_mean, momentum):
    return [momentum * c + (1 - momentum) * b for c, b in zip(center, batch_mean)]


def dino_cross_entropy(teacher, student_logits, student_temp):
    log_student = log_softmax([s / student_temp for s in student_logits])
    return -sum(t * l for t, l in zip(teacher, log_student))


def ibot_loss(teacher_patches, student_patches, masked, student_temp):
    positions = [i for i, is_masked in enumerate(masked) if is_masked]
    if not positions:
        raise ValueError('iBOT needs at least one masked patch position')
    losses = [dino_cross_entropy(teacher_patches[i], student_patches[i], student_temp) for i in positions]
    return sum(losses) / len(losses)


def koleo(features):
    if len(features) < 2:
        raise ValueError('KoLeo needs at least two features')
    unit = [[x / math.sqrt(sum(v * v for v in f)) for x in f] for f in features]
    nearest = []
    for i, a in enumerate(unit):
        distances = [math.dist(a, b) for j, b in enumerate(unit) if j != i]
        nearest.append(min(distances))
    return -sum(math.log(d) for d in nearest) / len(nearest)
