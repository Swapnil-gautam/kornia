"""Probe which grid_sample argument combinations assert on the MPS backend.

Maps the crash surface for kornia#4032: for each (source shape, grid shape) pair,
report whether F.grid_sample succeeds, raises a catchable error, or would abort.
Run on an Apple Silicon runner; falls back to CPU (everything passes) elsewhere.
"""

import torch
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"probing on device: {device}, torch {torch.__version__}\n")

CASES = [
    # (label, src shape, grid shape)
    ("normal src, normal grid   ", (1, 1, 4, 4), (1, 3, 4, 2)),
    ("normal src, ZERO-H grid   ", (1, 1, 4, 4), (1, 0, 4, 2)),
    ("normal src, ZERO-W grid   ", (1, 1, 4, 4), (1, 3, 0, 2)),
    ("1x1 src,    ZERO-H grid   ", (1, 1, 1, 1), (1, 0, 4, 2)),
    ("1x1 src,    ZERO-W grid   ", (1, 1, 1, 1), (1, 3, 0, 2)),
    ("1x1 src,    1x1 grid      ", (1, 1, 1, 1), (1, 1, 1, 2)),
    ("ZERO-H src, 1x1 grid      ", (1, 1, 0, 4), (1, 1, 1, 2)),
    ("ZERO-H src, ZERO-H grid   ", (1, 1, 0, 4), (1, 0, 4, 2)),
    ("zero-batch src+grid       ", (0, 1, 4, 4), (0, 3, 4, 2)),
]

for label, src_shape, grid_shape in CASES:
    src = torch.zeros(*src_shape, device=device)
    grid = torch.zeros(*grid_shape, device=device)
    try:
        out = F.grid_sample(src, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
        print(f"OK    | {label} -> out shape {tuple(out.shape)}")
    except Exception as e:
        first_line = str(e).splitlines()[0][:110]
        print(f"RAISE | {label} -> {type(e).__name__}: {first_line}")
