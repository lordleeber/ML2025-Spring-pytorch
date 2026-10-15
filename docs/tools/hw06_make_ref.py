# Build the HW06 "reference" script: the code cells of HW06/hw6_colab.ipynb joined as-is,
# with only the changes needed to run as plain Python on this machine.
# Run the result from a directory holding the five data files (HW06/data/*); the notebook reads
# and writes relative paths (gsm8k_*.jsonl, sft/, <STUDENT_ID>.txt).
# usage: python docs/tools/hw06_make_ref.py HW06/hw6_colab.ipynb <out.py>
import json
import sys

nb = json.load(open(sys.argv[1]))
out = []
for i, cell in enumerate(nb['cells']):
  if cell['cell_type'] != 'code':
    continue
  lines = []
  for line in ''.join(cell['source']).split('\n'):
    if line.lstrip().startswith('!'):
      # nvidia-smi / wget / pip / huggingface-cli login: data is in HW06/data, packages are in
      # HW06/.venv, and the model is read from the local HF cache (HF_HUB_OFFLINE=1)
      lines.append('# [ref] ' + line)
    else:
      lines.append(line)
  out.append(f'# ---- cell [{i}] ----\n' + '\n'.join(lines) + '\n')

open(sys.argv[2], 'w').write('\n'.join(out))
