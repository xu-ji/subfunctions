# Set this to true to make scripts/slurm/cifar/* print commands only and not run (recommended first)
PRINT_COMMANDS_ONLY = False

# Set this to true to avoid requiring tensorflow and edward2 libraries (if not running GP baseline)
AVOID_GP_LIBRARIES = False

# Set this to true the first time running test
TRAIN_MNIST_MODEL_IN_TEST = True

CIFAR_DATA_ROOT = "/scratch/shared/nfs1/xuji/datasets/CIFAR"

SVHN_DATA_ROOT = "/scratch/shared/nfs1/xuji/datasets/SVHN"

MNIST_DATA_ROOT = "/scratch/shared/nfs1/xuji/datasets"

DEFAULT_MODELS_ROOT = "/scratch/shared/nfs1/xuji/generalization/models"

"""
If you want to run the residual flows density baseline for in_distribution or out_of_distribution:

RESIDUAL_FLOWS_PATH is the path to your copy of https://github.com/xu-ji/residual-flows
This is used by util/methods/explicit_density

RESIDUAL_FLOWS_MODEL_PATH is the path to your trained density model
This is used by the scripts' arguments

Your model must have been produced by the train_img.py script from the xu-ji repo, e.g. 
python train_img.py --data cifar10 --actnorm True --save experiments/cifar10
python train_img.py --data cifar100 --actnorm True --save experiments/cifar100

You can either train models from scratch as above
Or use my pretrained models (linked in README)
Or download the cifar10 model from the original residual flows authors and just reformat the file 
with the xu-ji train_img.py script with SAVE_ONLY set to True

"""
RESIDUAL_FLOWS_PATH = "/users/xuji/residual-flows"

RESIDUAL_FLOWS_MODEL_PATH_PATT = "/scratch/shared/nfs1/xuji/generalization/resflow_models" \
                                 "/%s_resflow_full_model.pytorch"
