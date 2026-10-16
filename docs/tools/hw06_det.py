# Run a script with PyTorch's deterministic algorithms turned on, without editing it
# (same wrapper as ML2022's docs/tools/hw07_det.py). Used whenever two runs must be compared
# bit for bit: the notebook reference vs the split code, or a tool vs the reference.
# HW06 also needs a fixed torch seed: the notebook only calls random.seed(42), and
# get_peft_model initialises LoRA A before SFTTrainer's set_seed(1126), so without this the
# LoRA init comes from torch's random-at-startup seed and differs on every run.
# HW06_TORCH_SEED (default 0) only fixes that startup seed; the script itself is unchanged.
# usage: CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/tools/hw06_det.py <script.py> [args...]
import os
import runpy
import sys

import torch

torch.use_deterministic_algorithms(True)
torch.manual_seed(int(os.environ.get('HW06_TORCH_SEED', '0')))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
