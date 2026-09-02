"""Reproduce kornia#4032 on MPS: warp_affine with a zero destination dimension.

Expected on unfixed kornia + MPS: RuntimeError internal assert from grid_sample.
Expected on CPU (any kornia): empty (1, 1, 0, 4) output, no error.
"""

import torch

from kornia.geometry.transform import warp_affine

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"repro on device: {device}, torch {torch.__version__}")

src = torch.rand(1, 1, 4, 4, device=device, requires_grad=True)
M = torch.eye(2, 3, device=device)[None].requires_grad_()

for dsize in [(0, 4), (3, 0)]:
    try:
        out = warp_affine(src, M, dsize)
        out.sum().backward()
        grads = src.grad is not None and M.grad is not None
        print(f"OK    | dsize={dsize} -> shape {tuple(out.shape)}, autograd connected: {grads}")
    except Exception as e:
        first_line = str(e).splitlines()[0][:110]
        print(f"RAISE | dsize={dsize} -> {type(e).__name__}: {first_line}")
