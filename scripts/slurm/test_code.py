import os
from scripts.global_constants import MNIST_DATA_ROOT, TRAIN_MNIST_MODEL_IN_TEST

if TRAIN_MNIST_MODEL_IN_TEST:
  train_model_cmd = "export CUDA_VISIBLE_DEVICES=3 && python -m scripts.train_models --data mnist " \
                    "--data_root %s --model_args 8 8 --seed 0 --model mlp --lr 0.1 --epochs 50 " \
                    "--cuda" % MNIST_DATA_ROOT
  print(train_model_cmd)
  os.system(train_model_cmd)

test_code_cmd = "export CUDA_VISIBLE_DEVICES=3 && python -m scripts.in_distribution --data " \
                "mnist --data_root %s --seed 0 --model mlp --cuda subfunctions --search_ps 1.0 " \
                "--precompute --precompute_p_i 0 --pattern_batch_sz 1000 --test_code_brute_force" \
                % MNIST_DATA_ROOT
print(test_code_cmd)
os.system(test_code_cmd)
