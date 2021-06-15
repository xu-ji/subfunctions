from scripts.slurm.cifar.constants import *
from scripts.global_constants import *
from scripts.analysis.print_results import data_pretty, models_pretty
import os.path as osp
import torch
import numpy as np
from util.general import cleanstr
from collections import defaultdict

def print_hyperparams():

  for data in datasets:
    for model, _ in models_ps:
      printstr = "%s & %s " % (data_pretty[data], models_pretty[model])

      p_counts = defaultdict(int)
      delta_counts = defaultdict(int)

      for seed in seeds:
        fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
          "%s_%s_%s_%s_%s_results" % (data, seed, "subfunctions", model, "")))

        if osp.exists(fname):
          results = torch.load(fname)
          p = results["winning_method_variables"]["p"] # rho
          delta = results["winning_method_variables"]["delta"].cpu().numpy() # delta
          p_counts[str(p)] += 1
          delta_counts[str(delta)] += 1

      for hdict in [p_counts, delta_counts]:
        printstr += " & "
        i = 0
        for k, v in sorted(list(hdict.items()), key = lambda t: -t[1]):
          printstr += " %s (\#: %s)" % (k, v)
          if i < len(hdict) - 1:
            printstr += ","
          i += 1

      printstr += "\\\\"
      print(printstr)


if __name__ == "__main__":
  print_hyperparams()