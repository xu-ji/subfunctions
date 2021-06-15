import os
from util.general import augment_command
from scripts.global_constants import PRINT_COMMANDS_ONLY


datasets_settings = [("cifar100",
                      "--lr 0.1 --epochs 200 --lr_sched_epoch_gap 60 --lr_sched_mult 0.2 "
                      "--weight_decay 5e-4"),
                     ("cifar10",
                      "--lr 0.1 --epochs 100 --lr_sched_epoch_gap 30 --lr_sched_mult 0.1 "
                      "--weight_decay 5e-4")]

models = ["vgg16_bn", "resnet50model"]

template = "python -m scripts.train_models --data %s --model %s --seed %d %s --cuda"

seeds = list(range(5))

for data, settings in datasets_settings:
  for model in models:
    for seed in seeds:
      cmd = template % (data, model, seed, settings)
      cmd = augment_command(cmd, "slurm", suff="generic_large.sh")
      print("%s" % cmd)
      if not PRINT_COMMANDS_ONLY:
        os.system(cmd)
