import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY
from scripts.slurm.cifar.constants import *

# Ensure subfunctions_pre is run already

template = "python -m scripts.in_distribution --data %s --model %s --seed %s --cuda " \
           "ensemble_subfunctions --search_ps %s"

for data in datasets:
  for model, ps in models_ps:
    cmd = template % (data, model, ensemble_seeds, ps)
    cmd = augment_command(cmd, "slurm", suff="generic.sh")

    print("%s" % cmd)
    if not PRINT_COMMANDS_ONLY:
      os.system(cmd)
