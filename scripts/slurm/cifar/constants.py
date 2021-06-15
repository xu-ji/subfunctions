from scripts.global_constants import CIFAR_DATA_ROOT, SVHN_DATA_ROOT

# In distribution
datasets = ["cifar100", "cifar10"]

models_ps = [("vgg16_bn", "32.0 48.0 64.0 98.0 128.0 192.0 256.0 384.0 512.0"), # use exactly 1 whitespace gaps
             ("resnet50model", "4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0")
]

seeds = list(range(5))

ensemble_seeds = ""
for seed in seeds:
  ensemble_seeds += "%s " % seed

# Subfunctions only
subfunctions_pattern_batch_sz = 16

# Class distance only
class_distance_batch_szs = {"vgg16_bn": 32,
                            "resnet50model": 4}

# Out of distribution only
ood_datasets = {"cifar100":
                  [("cifar10", CIFAR_DATA_ROOT),
                   ("svhn", SVHN_DATA_ROOT)],
                "cifar10":
                  [("cifar100", CIFAR_DATA_ROOT),
                   ("svhn", SVHN_DATA_ROOT)],
                }
