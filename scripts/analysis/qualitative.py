import torch
import os.path as osp
from scripts.global_constants import *
import argparse
from util.general import *
from util.data import *
from util.methods import *
from scripts.analysis.print_results import data_pretty


def qualitative():
  # These arguments are just for this rendering script
  config = argparse.ArgumentParser(allow_abbrev=False)
  config.add_argument("--num_render_imgs", type=int, default=16)
  config.add_argument("--print_scores", default=False, action="store_true")

  config.add_argument("--data", type=str, choices=["cifar10", "cifar100", "mnist"], required=True)
  config.add_argument("--data_root", type=str, default=CIFAR_DATA_ROOT)

  config.add_argument("--new_distr_data", type=str, choices=["svhn", "cifar10", "cifar100"],
                      default="svhn")
  config.add_argument("--new_distr_data_root", type=str, required=True)

  config.add_argument("--batch_size", type=int, default=200)
  config.add_argument("--val_pc", type=float, default=0.15)
  config.add_argument("--workers", type=int, default=2)
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

  set_seed(config.seed[0])  # for reproducibility
  train_loader, val_loader, old_test_loader = get_data(config, val_pc=config.val_pc, training=False)

  # now, set config.data to ood data
  old_data, old_data_root = config.data, config.data_root
  config.data = config.new_distr_data
  config.data_root = config.new_distr_data_root
  _, _, new_distr_loader_ood = get_data(config, val_pc=0., training=False)
  config.data = old_data
  config.data_root = old_data_root

  print("Ood experiment: %s (%s) -> %s (%s)" % (config.data, config.data_root,
                                                      config.new_distr_data,
                                                      config.new_distr_data_root))

  # merge data in balanced way and make new labels
  new_distr_loader = merge_data(config, new_distr_loader_ood, old_test_loader)

  print("Doing ranking rendering: %s (%s) -> %s (%s)" % (config.data, config.data_root,
                                                      config.new_distr_data,
                                                      config.new_distr_data_root))

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
  unreliability_all = []
  imgs_all = []
  corrects_all = []
  for batch_i, (imgs, targets) in enumerate(new_distr_loader):
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    unreliability_i, _ = globals()["%s_metric" % config.method](config, method_variables, model,
                                                                imgs, targets.to(torch.long))
    unreliability_all.append(unreliability_i)
    imgs_all.append(imgs.cpu())
    # correct = whether it belongs to original class, which is dictated by targets
    assert targets.dtype == torch.bool
    corrects_all.append(targets)

  unreliability_all = torch.cat(unreliability_all)
  corrects_all = torch.cat(corrects_all)
  imgs_all = torch.cat(imgs_all, dim=0)

  imgs_all = imgs_all - imgs_all.min()
  imgs_all = imgs_all / imgs_all.max() # [0, 1]

  # rank unreliability for old distr and new distr, getting best and worst for either
  orig_data = corrects_all.nonzero(as_tuple=True)
  new_data = (~corrects_all).nonzero(as_tuple=True)
  assert len(orig_data) == 1 and orig_data[0].shape == (len(old_test_loader.dataset),)
  assert len(new_data) == 1 and new_data[0].shape == (len(new_distr_loader_ood.dataset),)
  print("data sizes old %s, new %s" % (len(old_test_loader.dataset), len(new_distr_loader_ood.dataset)))
  orig_data = orig_data[0]
  new_data = new_data[0]

  orig_imgs = imgs_all[orig_data]
  orig_unreliability = unreliability_all[orig_data]

  new_imgs = imgs_all[new_data]
  new_unreliability = unreliability_all[new_data]

  side = int(config.num_render_imgs ** 0.5)

  for orig_or_new, (unreliability_curr, imgs_curr) in enumerate([(orig_unreliability, orig_imgs),
                                                                 (new_unreliability, new_imgs)]) :
    reliable_inds = np.random.choice(unreliability_curr.shape[0]//5, size=config.num_render_imgs, replace=False)
    unreliable_inds = np.random.choice(unreliability_curr.shape[0]//5, size=config.num_render_imgs, replace=False) + (4 * unreliability_curr.shape[0] // 5)
    #reliable_inds = np.sort(reliable_inds)
    #unreliable_inds = np.sort(unreliable_inds)

    ranked_unreliability_inds = torch.argsort(unreliability_curr) # smallest to largest

    for reliable_or_unreliable, inds in enumerate([reliable_inds, unreliable_inds]):
      fig, ax = plt.subplots(side, side, figsize=(2 * side, 2 * side))
      for j, ii in enumerate(inds):
        row, col = divmod(j, side)
        i = ranked_unreliability_inds[ii]
        img = imgs_curr[i]
        img = img.permute(1, 2, 0).cpu().numpy()
        print(("img min max", img.min(), img.max()))
        unreliability = unreliability_curr[i].item()

        ax[row, col].imshow(img)
        if config.print_scores:
          ax[row, col].set_ylabel("%.3f" % unreliability)
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        print("adding unreliability %s" % (unreliability))

      if not config.print_scores:
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

      plots_fname = osp.join(DEFAULT_MODELS_ROOT,
                             "%s.png" % cleanstr(
                               "render_%s_%s_to_%s_%s_%s_orig_%d_reliable_%d" % (
                                 config.method, config.data, config.new_distr_data, config.model,
                                 config.seed, orig_or_new, reliable_or_unreliable)))
      print("saving to %s" % plots_fname)
      fig.savefig(plots_fname, bbox_inches="tight")

  plt.close("all")



if __name__ == "__main__":
  qualitative()
