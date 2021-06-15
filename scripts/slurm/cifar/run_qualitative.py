from scripts.global_constants import *
import os

cmd = "export CUDA_VISIBLE_DEVICES=2 && python -m scripts.analysis.qualitative --num_render_imgs 16 --new_distr_data cifar100 --new_distr_data_root %s --data cifar10 --model resnet50model --seed 1 --cuda subfunctions --search_ps 4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0 "
cmd = cmd % CIFAR_DATA_ROOT

print("Executing %s" % cmd)
os.system(cmd)