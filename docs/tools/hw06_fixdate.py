# Run a HW06 script with the chat template's "Today Date" pinned, without editing the script.
# Llama 3.2's template writes `Today Date: {strftime_now("%d %b %Y")}` into the system header of
# every evaluation prompt, so the same run gives different answers on different days.
# strftime_now is a local function that calls datetime.now(), so this wrapper replaces the
# `datetime` that transformers.utils.chat_template_utils imported with one whose now() returns
# the fixed date, then behaves like hw06_det.py (deterministic algorithms, fixed torch seed).
# usage: HW06_DATE=2026-10-11 HW06_TORCH_SEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
#        python docs/tools/hw06_fixdate.py <script.py> [args...]
import datetime as dt
import os
import runpy
import sys

import torch
import transformers.utils.chat_template_utils as ctu

DATE = dt.datetime.fromisoformat(os.environ['HW06_DATE'])


class FixedDatetime(dt.datetime):
  @classmethod
  def now(cls, tz=None):
    return DATE


ctu.datetime = FixedDatetime
torch.use_deterministic_algorithms(True)
torch.manual_seed(int(os.environ.get('HW06_TORCH_SEED', '0')))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
