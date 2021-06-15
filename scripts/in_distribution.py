import argparse
from util.data import *
from util.general import *
from util.methods import *
import arch
from datetime import datetime
from sys import stdout
from scripts.global_constants import *


def in_distribution():
  # ------------------------------------------------------------------------------------------------
  # Arguments
  # ------------------------------------------------------------------------------------------------

  config = argparse.ArgumentParser(allow_abbrev=False)

  config.add_argument("--data", type=str, choices=["cifar10", "cifar100", "mnist"], required=True)
  config.add_argument("--data_root", type=str, default=CIFAR_DATA_ROOT)
  config.add_argument("--batch_size", type=int, default=200)
  config.add_argument("--val_pc", type=float,
                      default=0.15)  # must match one used for train_models.py
  config.add_argument("--workers", type=int, default=1)
  config.add_argument("--model", type=str, default="")
  config.add_argument("--models_root", type=str, default=DEFAULT_MODELS_ROOT)
  config.add_argument("--seed", type=int, nargs="+",
                      required=True)  # to load the corresponding model, and for reproducibility
  config.add_argument("--cuda", default=False, action="store_true")
  config.add_argument("--suff", type=str, default="")

  config.add_argument("--threshold_divs", type=int, default=1000)

  subparsers = config.add_subparsers(dest="method")

  subfunctions_config = subparsers.add_parser("subfunctions")
  ensemble_subfunctions_config = subparsers.add_parser("ensemble_subfunctions")

  for subconfig in [subfunctions_config, ensemble_subfunctions_config]:
    subconfig.add_argument("--search_deltas", type=float, nargs="+",
                           default=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    subconfig.add_argument("--search_ps", type=float, nargs="+", required=True)
    subconfig.add_argument("--precompute", default=False, action="store_true")
    subconfig.add_argument("--precompute_p_i", type=int, default=-1)
    subconfig.add_argument("--pattern_batch_sz", type=int,
                           default=-1)  # set to -1 to do whole dataset at once

    subconfig.add_argument("--no_bound", default=False, action="store_true")
    subconfig.add_argument("--no_log", default=False, action="store_true")
    subconfig.add_argument("--dist_fn", type=str, default="gaussian", choices=["gaussian"])
    subconfig.add_argument("--select_on_AUROC", default=False, action="store_true")

    subconfig.add_argument("--test_code_brute_force", default=False, action="store_true")

  class_distance_config = subparsers.add_parser("class_distance")
  class_distance_config.add_argument("--search_eps", type=float, nargs="+",
                                     default=[0.001, 0.005, 0.01, 0.05, 0.1])
  class_distance_config.add_argument("--balance_data", default=False, action="store_true")

  explicit_density_config = subparsers.add_parser("explicit_density")
  explicit_density_config.add_argument("--density_model_path_pattern", type=str,
                                       default=RESIDUAL_FLOWS_MODEL_PATH_PATT)

  gaussian_process_config = subparsers.add_parser("gaussian_process")
  gaussian_process_config.add_argument("--gp_hidden_dim", type=int, default=1024)
  gaussian_process_config.add_argument("--gp_scales", type=float, nargs="+", default=[1., 2., 4., 8.])

  _ = subparsers.add_parser("max_response")
  _ = subparsers.add_parser("entropy")
  _ = subparsers.add_parser("margin")
  _ = subparsers.add_parser("ensemble_var")
  _ = subparsers.add_parser("ensemble_max_response")

  config = config.parse_args()
  print("Config: %s" % config)

  # ------------------------------------------------------------------------------------------------
  # Script
  # ------------------------------------------------------------------------------------------------

  start_time = datetime.now()

  set_seed(config.seed[0])  # for reproducibility
  train_loader, val_loader, test_loader = get_data(config, val_pc=config.val_pc, training=False)

  for loader in [train_loader, val_loader, test_loader]:
    print_first_labels(loader)  # sanity

  model = [
    torch.load(osp.join(config.models_root, "%s_%d_%s.pytorch" % (config.data, s, config.model)))[
      "model"].eval() for s in config.seed]
  if len(config.seed) == 1:
    config.seed = config.seed[0]
    model = model[0]
    print("original acc: %s" % torch.load(
      osp.join(config.models_root, "%s_%d_%s.pytorch" % (config.data, config.seed, config.model)))[
      "acc"])

  # Store precomputations if not stored already and find val data hyperparameters if any. Also
  # adapt model if necessary.
  model, method_variables = globals()["%s_pre" % config.method](config, model, train_loader,
                                                                val_loader)

  # Run through test data batches, pass each batch to metric method along with needed params,
  # get metrics back, store with ground truth
  unreliability = []
  corrects = []
  for batch_i, (imgs, targets) in enumerate(test_loader):
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    unreliability_i, corrects_i = globals()["%s_metric" % config.method](config, method_variables,
                                                                         model, imgs, targets)
    unreliability.append(unreliability_i)
    corrects.append(corrects_i)

  unreliability = torch.cat(unreliability)
  corrects = torch.cat(corrects)

  store_fname_prefix = "%s_%s_%s_%s_%s" % (
  config.data, config.seed, config.method, config.model, config.suff)
  eval_and_store(config, unreliability, corrects, method_variables, store_fname_prefix)
  print("Took time: %s" % (datetime.now() - start_time))


if __name__ == "__main__":
  in_distribution()
