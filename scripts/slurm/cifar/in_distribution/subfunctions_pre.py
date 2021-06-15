import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY
from scripts.slurm.cifar.constants import *

template = "python -m scripts.in_distribution --data %s --model %s --seed %d --cuda subfunctions " \
           "--search_ps %s --precompute --precompute_p_i %d --pattern_batch_sz %d"

for data in datasets:
  for model, ps in models_ps:
    model_precompute_p_is = list(range(len(ps.split(" ")))) # list(range(9))
    for seed in seeds:
      for p_i_i, precompute_p_i in enumerate(model_precompute_p_is):
        cmd = template % (data, model, seed, ps, precompute_p_i, subfunctions_pattern_batch_sz)
        cmd = augment_command(cmd, "slurm", suff="generic_large.sh")
        print("%s" % cmd)
        if not PRINT_COMMANDS_ONLY:
          os.system(cmd)
