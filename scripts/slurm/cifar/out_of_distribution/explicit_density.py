import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY
from scripts.slurm.cifar.constants import *

template = "python -m scripts.out_of_distribution --new_distr_data %s --new_distr_data_root %s " \
           "--data %s --model %s --seed %d --cuda explicit_density"

for data in datasets:
  for model, _ in models_ps:
    for ood_dataset, ood_path in ood_datasets[data]:
      for seed in seeds:
        cmd = template % (ood_dataset, ood_path, data, model, seed)
        cmd = augment_command(cmd, "slurm", suff="generic.sh")

        print("%s" % cmd)
        if not PRINT_COMMANDS_ONLY:
          os.system(cmd)
