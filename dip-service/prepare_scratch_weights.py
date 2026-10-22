"""Extract only the trained U-Net parameters from the upstream training file."""
from pathlib import Path
import sys

import torch


source, target = map(Path, sys.argv[1:3])
checkpoint = torch.load(source, map_location="cpu", weights_only=True)
state = checkpoint["model_state"]
if state and next(iter(state)).startswith("module."):
    state = {key.removeprefix("module."): value for key, value in state.items()}
torch.save(state, target)
print(f"Prepared {len(state)} model tensors ({target.stat().st_size} bytes)")
