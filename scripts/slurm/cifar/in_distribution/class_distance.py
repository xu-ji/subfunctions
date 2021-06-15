import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY
from scripts.slurm.cifar.constants import *

template = "python -m scripts.in_distribution --data %s --model %s --seed %d --cuda --batch_size " \
           "%s class_distance --balance_data"

for data in datasets:
  for model, _ in models_ps:
    for seed in seeds:
      cmd = template % (data, model, seed, class_distance_batch_szs[model])
      cmd = augment_command(cmd, "slurm", suff="generic_large_mem.sh")

      print("%s" % cmd)
      if not PRINT_COMMANDS_ONLY:
        os.system(cmd)
