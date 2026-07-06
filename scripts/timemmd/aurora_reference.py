"""Aurora reference values for TimeMMD.

Only Aurora columns are mirrored here: zero-shot and 10% few-shot MSE/MAE.
Source: https://arxiv.org/html/2509.22295v4#Sx5
"""

from __future__ import annotations

from typing import Final

SOURCE_URL: Final = "https://arxiv.org/html/2509.22295v4"

REFERENCE: Final[dict[tuple[str, int, str], dict[str, float]]] = {
    ("Agriculture", 6, "aurora_zero_shot"): {"mse": 0.184, "mae": 0.295},
    ("Agriculture", 8, "aurora_zero_shot"): {"mse": 0.242, "mae": 0.335},
    ("Agriculture", 10, "aurora_zero_shot"): {"mse": 0.297, "mae": 0.365},
    ("Agriculture", 12, "aurora_zero_shot"): {"mse": 0.365, "mae": 0.398},
    ("Agriculture", 6, "aurora_few_shot"): {"mse": 0.127, "mae": 0.233},
    ("Agriculture", 8, "aurora_few_shot"): {"mse": 0.190, "mae": 0.289},
    ("Agriculture", 10, "aurora_few_shot"): {"mse": 0.236, "mae": 0.310},
    ("Agriculture", 12, "aurora_few_shot"): {"mse": 0.295, "mae": 0.340},
    ("Climate", 6, "aurora_zero_shot"): {"mse": 0.859, "mae": 0.747},
    ("Climate", 8, "aurora_zero_shot"): {"mse": 0.858, "mae": 0.746},
    ("Climate", 10, "aurora_zero_shot"): {"mse": 0.868, "mae": 0.748},
    ("Climate", 12, "aurora_zero_shot"): {"mse": 0.875, "mae": 0.753},
    ("Climate", 6, "aurora_few_shot"): {"mse": 0.867, "mae": 0.744},
    ("Climate", 8, "aurora_few_shot"): {"mse": 0.858, "mae": 0.745},
    ("Climate", 10, "aurora_few_shot"): {"mse": 0.863, "mae": 0.744},
    ("Climate", 12, "aurora_few_shot"): {"mse": 0.869, "mae": 0.749},
    ("Economy", 6, "aurora_zero_shot"): {"mse": 0.035, "mae": 0.150},
    ("Economy", 8, "aurora_zero_shot"): {"mse": 0.033, "mae": 0.145},
    ("Economy", 10, "aurora_zero_shot"): {"mse": 0.032, "mae": 0.143},
    ("Economy", 12, "aurora_zero_shot"): {"mse": 0.032, "mae": 0.144},
    ("Economy", 6, "aurora_few_shot"): {"mse": 0.015, "mae": 0.095},
    ("Economy", 8, "aurora_few_shot"): {"mse": 0.015, "mae": 0.099},
    ("Economy", 10, "aurora_few_shot"): {"mse": 0.016, "mae": 0.101},
    ("Economy", 12, "aurora_few_shot"): {"mse": 0.016, "mae": 0.102},
    ("Energy", 12, "aurora_zero_shot"): {"mse": 0.117, "mae": 0.245},
    ("Energy", 24, "aurora_zero_shot"): {"mse": 0.226, "mae": 0.354},
    ("Energy", 36, "aurora_zero_shot"): {"mse": 0.292, "mae": 0.409},
    ("Energy", 48, "aurora_zero_shot"): {"mse": 0.383, "mae": 0.472},
    ("Energy", 12, "aurora_few_shot"): {"mse": 0.097, "mae": 0.212},
    ("Energy", 24, "aurora_few_shot"): {"mse": 0.199, "mae": 0.322},
    ("Energy", 36, "aurora_few_shot"): {"mse": 0.271, "mae": 0.352},
    ("Energy", 48, "aurora_few_shot"): {"mse": 0.352, "mae": 0.431},
    ("Environment", 48, "aurora_zero_shot"): {"mse": 0.281, "mae": 0.380},
    ("Environment", 96, "aurora_zero_shot"): {"mse": 0.284, "mae": 0.382},
    ("Environment", 192, "aurora_zero_shot"): {"mse": 0.270, "mae": 0.375},
    ("Environment", 336, "aurora_zero_shot"): {"mse": 0.269, "mae": 0.377},
    ("Environment", 48, "aurora_few_shot"): {"mse": 0.269, "mae": 0.372},
    ("Environment", 96, "aurora_few_shot"): {"mse": 0.271, "mae": 0.373},
    ("Environment", 192, "aurora_few_shot"): {"mse": 0.269, "mae": 0.374},
    ("Environment", 336, "aurora_few_shot"): {"mse": 0.251, "mae": 0.368},
    ("Health", 12, "aurora_zero_shot"): {"mse": 1.093, "mae": 0.668},
    ("Health", 24, "aurora_zero_shot"): {"mse": 1.572, "mae": 0.849},
    ("Health", 36, "aurora_zero_shot"): {"mse": 1.688, "mae": 0.920},
    ("Health", 48, "aurora_zero_shot"): {"mse": 1.857, "mae": 0.963},
    ("Health", 12, "aurora_few_shot"): {"mse": 0.992, "mae": 0.641},
    ("Health", 24, "aurora_few_shot"): {"mse": 1.332, "mae": 0.796},
    ("Health", 36, "aurora_few_shot"): {"mse": 1.467, "mae": 0.818},
    ("Health", 48, "aurora_few_shot"): {"mse": 1.579, "mae": 0.847},
    ("Security", 6, "aurora_zero_shot"): {"mse": 67.572, "mae": 3.909},
    ("Security", 8, "aurora_zero_shot"): {"mse": 70.576, "mae": 4.013},
    ("Security", 10, "aurora_zero_shot"): {"mse": 74.173, "mae": 4.148},
    ("Security", 12, "aurora_zero_shot"): {"mse": 77.579, "mae": 4.264},
    ("Security", 6, "aurora_few_shot"): {"mse": 64.513, "mae": 3.798},
    ("Security", 8, "aurora_few_shot"): {"mse": 67.828, "mae": 3.930},
    ("Security", 10, "aurora_few_shot"): {"mse": 72.423, "mae": 4.092},
    ("Security", 12, "aurora_few_shot"): {"mse": 75.482, "mae": 4.132},
    ("SocialGood", 6, "aurora_zero_shot"): {"mse": 0.701, "mae": 0.442},
    ("SocialGood", 8, "aurora_zero_shot"): {"mse": 0.804, "mae": 0.493},
    ("SocialGood", 10, "aurora_zero_shot"): {"mse": 0.886, "mae": 0.543},
    ("SocialGood", 12, "aurora_zero_shot"): {"mse": 0.960, "mae": 0.587},
    ("SocialGood", 6, "aurora_few_shot"): {"mse": 0.689, "mae": 0.427},
    ("SocialGood", 8, "aurora_few_shot"): {"mse": 0.784, "mae": 0.461},
    ("SocialGood", 10, "aurora_few_shot"): {"mse": 0.850, "mae": 0.532},
    ("SocialGood", 12, "aurora_few_shot"): {"mse": 0.931, "mae": 0.554},
    ("Traffic", 6, "aurora_zero_shot"): {"mse": 0.154, "mae": 0.285},
    ("Traffic", 8, "aurora_zero_shot"): {"mse": 0.158, "mae": 0.286},
    ("Traffic", 10, "aurora_zero_shot"): {"mse": 0.163, "mae": 0.289},
    ("Traffic", 12, "aurora_zero_shot"): {"mse": 0.168, "mae": 0.294},
    ("Traffic", 6, "aurora_few_shot"): {"mse": 0.149, "mae": 0.292},
    ("Traffic", 8, "aurora_few_shot"): {"mse": 0.155, "mae": 0.284},
    ("Traffic", 10, "aurora_few_shot"): {"mse": 0.160, "mae": 0.287},
    ("Traffic", 12, "aurora_few_shot"): {"mse": 0.165, "mae": 0.296},
}


def get_reference(domain: str, pred_len: int, reference_model: str) -> dict[str, float] | None:
    return REFERENCE.get((domain, int(pred_len), reference_model))
