import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY
from scripts.slurm.cifar.constants import *

template = "python -m scripts.out_of_distribution --new_distr_data %s --new_distr_data_root %s " \
           "--data %s --model %s --seed %s --cuda ensemble_subfunctions --search_ps %s"

for data in datasets:
  for model, ps in models_ps:
    for ood_dataset, ood_path in ood_datasets[data]:
      cmd = template % (ood_dataset, ood_path, data, model, ensemble_seeds, ps)
      cmd = augment_command(cmd, "slurm", suff="generic.sh")

      print("%s" % cmd)
      if not PRINT_COMMANDS_ONLY:
        os.system(cmd)
