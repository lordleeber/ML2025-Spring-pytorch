# Compare two HW06 runs bit for bit: every LoRA checkpoint (torch.equal on each tensor of
# adapter_model.safetensors) and the submission file (<STUDENT_ID>.txt, the GSM8K answers +
# AILuminate responses as a Python list).
# usage: python docs/tools/hw06_compare_runs.py <run_dir_a> <run_dir_b> [submission_name]
import ast
import os
import sys

import torch
from safetensors.torch import load_file

a, b = sys.argv[1], sys.argv[2]
sub = sys.argv[3] if len(sys.argv) > 3 else '.txt'
ckpts = sorted(os.listdir(os.path.join(a, 'sft')), key=lambda s: int(s.split('-')[-1]))
for c in ckpts:
  ta = load_file(os.path.join(a, 'sft', c, 'adapter_model.safetensors'))
  tb = load_file(os.path.join(b, 'sft', c, 'adapter_model.safetensors'))
  same = ta.keys() == tb.keys() and all(torch.equal(ta[k], tb[k]) for k in ta)
  print(f'{c:>16}: {len(ta)} tensors, {"identical" if same else "DIFFERENT"}')
pa = ast.literal_eval(open(os.path.join(a, sub)).read())
pb = ast.literal_eval(open(os.path.join(b, sub)).read())
diff = [i for i, (x, y) in enumerate(zip(pa, pb)) if x != y]
print(f'submission: {len(pa)} vs {len(pb)} entries, {len(diff)} differ'
      + (f' (first: {diff[:10]})' if diff else ''))
