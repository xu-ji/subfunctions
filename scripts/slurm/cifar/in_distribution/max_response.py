import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY
from scripts.slurm.cifar.constants import *

template = "python -m scripts.in_distribution --data %s --model %s --seed %d --cuda max_response"

for data in datasets:
  for model, _ in models_ps:
    for seed in seeds:
      cmd = template % (data, model, seed)
      cmd = augment_command(cmd, "slurm", suff="generic.sh")

      print("%s" % cmd)
      if not PRINT_COMMANDS_ONLY:
        os.system(cmd)
