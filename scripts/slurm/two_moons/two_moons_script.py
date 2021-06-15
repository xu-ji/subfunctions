import os
from sys import stdout
import argparse

args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument("--seed", type=int, default=4)
args.add_argument("--gpu", type=int, default=3)
args.add_argument("--radius_mult", type=float, default=3.0)

args = args.parse_args()

train_model = True
precomp = True
postcomp = True

norm_data_str = " --two_moons_norm_data "
search_ps = "1.0 2.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0"

if train_model:
  train_model_cmd = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.train_models --data " \
                    "two_moons --model mlp --model_args 32 32 --seed %d --lr 0.1 --epochs 8 " \
                    "--cuda --batch_size 100 --workers 1 %s"
  train_model_cmd = train_model_cmd % (args.gpu, args.seed, norm_data_str)

  print("Executing training %s" % train_model_cmd)
  stdout.flush()
  os.system(train_model_cmd)

if precomp:
  precomp_str = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.two_moons --data two_moons " \
                "--model mlp --seed %d --cuda --batch_size 100 --workers 1 %s subfunctions " \
                "--search_ps %s --precompute --precompute_p_i %d"

  for p_i in range(11):
    precomp_cmd = precomp_str % (args.gpu, args.seed, norm_data_str, search_ps, p_i)

    print("Executing precomp %s" % precomp_cmd)
    stdout.flush()
    os.system(precomp_cmd)

if postcomp:
  postcomp_str = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.two_moons --data two_moons " \
                 "--model mlp --seed %d --cuda --batch_size 100 --workers 1 %s --radius_mult %s " \
                 "subfunctions --search_ps %s"
  postcomp_cmd = postcomp_str % (args.gpu, args.seed, norm_data_str, args.radius_mult, search_ps)
  print("Executing postcomp %s" % postcomp_cmd)
  stdout.flush()
  os.system(postcomp_cmd)
