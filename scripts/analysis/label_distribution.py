import torch
import os.path as osp
from scripts.global_constants import *
import argparse
from util.general import *
from util.data import *
from util.methods import *
from scripts.analysis.print_results import data_pretty


CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak', 'orange', 'orchid', 'otter', 'palm', 'pear',
    'pickup_truck', 'pine', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman',
    'worm'
]

# for: vehicles, mammals, birds and frogs
CIFAR100_LIKE_CIFAR10 = [
  'bear', 'leopard', 'lion', 'tiger', 'wolf', # large carnivores
  'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', # large omnivores and herbivores
  'fox', 'porcupine', 'possum', 'raccoon', 'skunk', # medium sized mammals
  'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', # small mammals
  'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', # vehicles 1 incl bicycles (not really vehicles...)
  'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor', # vehicles 2 incl lawn mowers and rockets (not really vehicles...)
]


for l in CIFAR100_LIKE_CIFAR10:
  if not l in CIFAR100_LABELS_LIST:
    print(l)
    assert False

def qualitative():
  # These arguments are just for this rendering script
  config = argparse.ArgumentParser(allow_abbrev=False)
  config.add_argument("--render_imgs", default=False, action="store_true")

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

  assert config.new_distr_data == "cifar100"

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

  new_distr_loader = new_distr_loader_ood

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
  targets_all = []
  for batch_i, (imgs, targets) in enumerate(new_distr_loader):
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    unreliability_i, _ = globals()["%s_metric" % config.method](config, method_variables, model,
                                                                imgs, targets.to(torch.long))
    unreliability_all.append(unreliability_i)
    imgs_all.append(imgs.cpu())
    targets_all.append(targets)

  unreliability_all = torch.cat(unreliability_all)
  targets_all = torch.cat(targets_all)
  imgs_all = torch.cat(imgs_all, dim=0)
  imgs_all = imgs_all - imgs_all.min()
  imgs_all = imgs_all / imgs_all.max()

  ranked_unreliability_inds = torch.argsort(unreliability_all) # small (reliable) to larger
  assert len(ranked_unreliability_inds.shape) == 1

  all_ranks = []
  ranks_names = []
  medians = []
  colours = []
  #orig_colours = [np.array(hsv_to_rgb(hue, 0.6, 0.7)) for hue in np.linspace(0., 1., num=2)]
  orig_colours = ["tab:purple", "tab:green"]

  print("orig colours")
  print(orig_colours)

  for c in np.sort(targets_all.unique().cpu().numpy()):
    c_inds = (targets_all == c).nonzero(as_tuple=True)[0]
    c_name = CIFAR100_LABELS_LIST[c]

    # render one img with class name and ind just to check
    if config.render_imgs:
      fig, ax = plt.subplots(1, figsize=(3, 3))
      img = imgs_all[c_inds[0]]
      ax.imshow(img.permute(1, 2, 0).cpu().numpy())
      fig.suptitle("%s, class %d" % (c_name, c))
      fig.tight_layout()
      plots_fname = osp.join(DEFAULT_MODELS_ROOT,
                             "%s.png" % cleanstr(
                               "ood_%s_to_%s_%s_distribution_img_%d" % (config.data, config.new_distr_data, config.model, c)))
      print("saving to %s" % plots_fname)
      fig.savefig(plots_fname)
      plt.close("all")

    # for each class's ranks, get median, mean, quartiles, and limits
    c_ranks = []
    for ind in c_inds:
      rank = (ranked_unreliability_inds == ind).nonzero(as_tuple=True)[0]
      assert rank.shape == (1,)
      rank = rank[0]
      c_ranks.append(rank.item())
    c_ranks = np.array(c_ranks)

    all_ranks.append(c_ranks)
    ranks_names.append(c_name)
    medians.append(np.median(c_ranks))
    colours.append(orig_colours[int(c_name in CIFAR100_LIKE_CIFAR10)])

  medians = np.array(medians)
  ordering = np.argsort(medians) # small (reliable) to unreliable)

  all_ranks = [all_ranks[i] for i in ordering]
  ranks_names = [ranks_names[i] for i in ordering]
  colours = [colours[i] for i in ordering]

  print("colours")
  print(colours)

  medianprops = dict(linestyle='-.', linewidth=3, color='black')

  fig, ax = plt.subplots(1, figsize=(12, 3))
  #ax.set_title("")
  bplot = ax.boxplot(all_ranks, labels=ranks_names, patch_artist=True, medianprops=medianprops)

  for item in ['boxes', 'whiskers', 'fliers', 'caps']: #  'medians',
    plt.setp(bplot[item], color="tab:gray")

  for patch, col in zip(bplot['boxes'], colours):
    patch.set_facecolor(col)

  #ax.set_xlabel("%s class" % data_pretty[config.data])
  ax.set_ylabel("Unreliability ranking")
  plt.xticks(rotation=90)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  #ax.spines['bottom'].set_visible(False)
  #ax.spines['left'].set_visible(False)

  fig.tight_layout()
  plots_fname = osp.join(DEFAULT_MODELS_ROOT,
                         "%s.png" % cleanstr(
                           "ood_%s_%s_to_%s_%s_distribution_boxplots" % (
                           config.method, config.data, config.new_distr_data, config.model)))
  print("saving to %s" % plots_fname)
  fig.savefig(plots_fname)
  plt.close("all")


if __name__ == "__main__":
  qualitative()
