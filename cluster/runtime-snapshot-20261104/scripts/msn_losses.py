"""MSN-style objective (estilo MSN: aproximación declarada).

Teacher (unmasked, EMA) prototype assignments, sharpened, are cross-entropy targets for the student (masked view).
A mean-entropy regulariser on the student's average prototype distribution prevents collapse.
"""
import torch
import torch.nn.functional as F


def msn_loss(teacher_logits, student_logits, t_temp=0.04, s_temp=0.1, reg_weight=1.0):
    target = F.softmax(teacher_logits.detach() / t_temp, dim=-1)
    ce = -(target * F.log_softmax(student_logits / s_temp, dim=-1)).sum(-1).mean()
    mean_probs = F.softmax(student_logits / s_temp, dim=-1).mean(0)
    mean_entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum()
    return ce - reg_weight * mean_entropy
