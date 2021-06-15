import argparse
from util.data import *
from util.general import *
from util.methods import *
import arch
from datetime import datetime
from sys import stdout
from scripts.global_constants import *


# If this is run after prediction_filtering.py, no need to recompute precomputations (i.e. run
# straight away without precompute flag)
# Else, you need to do the precomputations with precompute flag

def out_of_distribution():
  # ------------------------------------------------------------------------------------------------
  # Arguments
  # ------------------------------------------------------------------------------------------------

  config = argparse.ArgumentParser(allow_abbrev=False)

  config.add_argument("--data", type=str, choices=["cifar10", "cifar100", "mnist"], required=True)
  config.add_argument("--data_root", type=str, default=CIFAR_DATA_ROOT)

  config.add_argument("--new_distr_data", type=str, choices=["svhn", "cifar10", "cifar100"],
                      default="svhn")
  config.add_argument("--new_distr_data_root", type=str, required=True)

  config.add_argument("--batch_size", type=int, default=200)
  config.add_argument("--val_pc", type=float, default=0.15)
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
  train_loader, val_loader, old_test_loader = get_data(config, val_pc=config.val_pc, training=False)

  # now, set config.data to ood data
  old_data, old_data_root = config.data, config.data_root
  config.data = config.new_distr_data
  config.data_root = config.new_distr_data_root
  _, _, new_distr_loader_ood = get_data(config, val_pc=0., training=False)
  config.data = old_data
  config.data_root = old_data_root

  print("Doing ood experiment: %s (%s) -> %s (%s)" % (config.data, config.data_root,
                                                      config.new_distr_data,
                                                      config.new_distr_data_root))

  # merge data in balanced way and make new labels
  new_distr_loader = merge_data(config, new_distr_loader_ood, old_test_loader)

  for loader in [old_test_loader, new_distr_loader_ood, new_distr_loader]:
    print_first_labels(loader)
    print(len(loader.dataset))

  model = [
    torch.load(osp.join(config.models_root, "%s_%d_%s.pytorch" % (config.data, s, config.model)))[
      "model"].eval() for s in config.seed]
  if len(config.seed) == 1:
    config.seed = config.seed[0]
    model = model[0]
    print("original acc: %s" % torch.load(
      osp.join(config.models_root, "%s_%d_%s.pytorch" % (config.data, config.seed, config.model)))[
      "acc"])

  # Don't use new distr data to choose hyperparameters
  model, method_variables = globals()["%s_pre" % config.method](config, model, train_loader,
                                                                val_loader)

  # Run through test data batches, pass each batch to metric method along with needed params,
  # get metrics back, store with ground truth
  unreliability = []
  corrects = []
  for batch_i, (imgs, targets) in enumerate(new_distr_loader):
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    unreliability_i, _ = globals()["%s_metric" % config.method](config, method_variables, model,
                                                                imgs, targets.to(torch.long))
    unreliability.append(unreliability_i)

    # correct = whether it belongs to original class, which is dictated by targets
    assert targets.dtype == torch.bool
    corrects.append(targets)

  unreliability = torch.cat(unreliability)
  corrects = torch.cat(corrects)

  store_fname_prefix = "ood_%s_to_%s_%s_%s_%s_%s" % (
  config.data, config.new_distr_data, config.seed, config.method, config.model, config.suff)
  eval_and_store(config, unreliability, corrects, method_variables, store_fname_prefix)
  print("Took time: %s" % (datetime.now() - start_time))


if __name__ == "__main__":
  out_of_distribution()
