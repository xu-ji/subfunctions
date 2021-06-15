from scripts.global_constants import *
import os

# subfunction error

cmd = "export CUDA_VISIBLE_DEVICES=3 && python -m scripts.analysis.label_distribution --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 0 --cuda subfunctions --search_ps 4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0"
cmd = cmd % CIFAR_DATA_ROOT
print("Executing %s" % cmd)
os.system(cmd)

# entropy

cmd = "export CUDA_VISIBLE_DEVICES=3 && python -m scripts.analysis.label_distribution --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 0 --cuda entropy"
cmd = cmd % CIFAR_DATA_ROOT
print("Executing %s" % cmd)
os.system(cmd)

# max response

cmd = "export CUDA_VISIBLE_DEVICES=2 && python -m scripts.analysis.label_distribution --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 0 --cuda max_response"
cmd = cmd % CIFAR_DATA_ROOT
print("Executing %s" % cmd)
os.system(cmd)

# margin

cmd = "export CUDA_VISIBLE_DEVICES=2 && python -m scripts.analysis.label_distribution --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 0 --cuda margin"
cmd = cmd % CIFAR_DATA_ROOT
print("Executing %s" % cmd)
os.system(cmd)

# GP

cmd = "export CUDA_VISIBLE_DEVICES=3 && python -m scripts.analysis.label_distribution --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 0 --cuda gaussian_process"
cmd = cmd % CIFAR_DATA_ROOT
print("Executing %s" % cmd)
os.system(cmd)

# residual flows density

cmd = "export CUDA_VISIBLE_DEVICES=2 && python -m scripts.analysis.label_distribution --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 0 --cuda explicit_density"
cmd = cmd % CIFAR_DATA_ROOT
print("Executing %s" % cmd)
os.system(cmd)

# class distance

cmd = "sbatch scripts/slurm/generic_large_mem.sh \"python -m scripts.analysis.label_distribution --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 0 --cuda --batch_size 4 class_distance --balance_data\" "
cmd = cmd % CIFAR_DATA_ROOT
print("Executing %s" % cmd)
os.system(cmd)
