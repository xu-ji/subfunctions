import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY
from scripts.slurm.cifar.constants import *

template = "python -m scripts.in_distribution --data %s --model %s --seed %d --cuda subfunctions " \
           "--search_ps %s"

for data in datasets:
  for model, ps in models_ps:
    for seed in seeds:
      cmd = template % (data, model, seed, ps)
      cmd = augment_command(cmd, "slurm", suff="generic.sh")
      print("%s" % cmd)
      if not PRINT_COMMANDS_ONLY:
        os.system(cmd)
