# Trace the first generator calls of a HW06 run (notebook reference or HW06/hw6.py) without
# editing it: same deterministic setup as hw06_det.py, plus a patched
# TextGenerationPipeline.__call__ that logs, for each call, hashes of the torch CPU/CUDA RNG
# states and Python's random state before the call, of the input chats, and of the output.
# Two runs that should match can be diffed line by line to find the first state that differs.
# After --calls calls it writes the log and exits.
# usage: HW06_TORCH_SEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/tools/hw06_trace.py
#        <log.jsonl> <calls> <script.py> [args...]
import hashlib
import json
import os
import random
import runpy
import sys

import torch

torch.use_deterministic_algorithms(True)
torch.manual_seed(int(os.environ.get('HW06_TORCH_SEED', '0')))
log_path, max_calls = sys.argv[1], int(sys.argv[2])
sys.argv = sys.argv[3:]

from transformers.pipelines.text_generation import TextGenerationPipeline  # noqa: E402

orig_call = TextGenerationPipeline.__call__
records = []


def h(x):
    return hashlib.sha256(x if isinstance(x, bytes) else repr(x).encode()).hexdigest()[:12]


def traced(self, text_inputs, **kwargs):
    rec = {'call': len(records), 'torch_cpu': h(torch.get_rng_state().numpy().tobytes()),
           'torch_cuda': h(torch.cuda.get_rng_state().numpy().tobytes()),
           'py_random': h(random.getstate()), 'input': h(text_inputs),
           'model_id': id(self.model), 'model_training': self.model.training}
    out = orig_call(self, text_inputs, **kwargs)
    text = out[0]['generated_text'][-1]['content']
    rec['output'] = h(text)
    rec['output_head'] = text[:60]
    records.append(rec)
    if len(records) >= max_calls:
        with open(log_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        sys.stdout.flush()
        os._exit(0)
    return out


TextGenerationPipeline.__call__ = traced
runpy.run_path(sys.argv[0], run_name='__main__')
