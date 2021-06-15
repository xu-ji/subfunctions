import argparse
import os.path as osp
import torch
from util.data import *
from util.general import *
from arch import MLP
from torch.nn import CrossEntropyLoss
from torch import optim
from datetime import datetime
from sys import stdout
from scripts.global_constants import *

from util.methods.subfunctions import subfunctions_pre, subfunctions_metric, \
  bool_tensor_content_hash
from util.two_moons import render_two_moons
from PIL import Image
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# This prints the pretty plots.

def two_moons():
  # ------------------------------------------------------------------------------------------------
  # Arguments
  # ------------------------------------------------------------------------------------------------

  config = argparse.ArgumentParser(allow_abbrev=False)

  config.add_argument("--threshold_divs", type=int, default=100)
  config.add_argument("--data", type=str, choices=["two_moons"], default="two_moons")
  config.add_argument("--data_root", type=str, default="")
  config.add_argument("--batch_size", type=int, default=256)
  config.add_argument("--workers", type=int, default=1)
  config.add_argument("--models_root", type=str, default=DEFAULT_MODELS_ROOT)
  config.add_argument("--seed", type=int, nargs="+",
                      required=True)  # to load the corresponding model, and for reproducibility
  config.add_argument("--cuda", default=False, action="store_true")
  config.add_argument("--suff", type=str, default="")

  config.add_argument("--model", type=str, default="")

  # for this two moons script only
  config.add_argument("--two_moons_norm_data", default=False, action="store_true")
  config.add_argument("--radius_mult", type=float, default=1.5)

  subparsers = config.add_subparsers(dest="method")

  subfunctions_config = subparsers.add_parser("subfunctions")
  for subconfig in [subfunctions_config]:
    subconfig.add_argument("--search_deltas", type=float, nargs="+",
                           default=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    subconfig.add_argument("--search_ps", type=float, nargs="+",
                           default=[1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0])
    subconfig.add_argument("--precompute", default=False, action="store_true")
    subconfig.add_argument("--precompute_p_i", type=int, default=-1)
    subconfig.add_argument("--pattern_batch_sz", type=int,
                           default=-1)  # set to -1 to do whole dataset at once

    subconfig.add_argument("--no_bound", default=False, action="store_true")
    subconfig.add_argument("--no_log", default=False, action="store_true")
    subconfig.add_argument("--dist_fn", type=str, default="gaussian", choices=["gaussian"])
    subconfig.add_argument("--select_on_AUROC", default=False, action="store_true")

  config = config.parse_args()
  config.test_code_brute_force = False
  print("Config: %s" % config)

  set_seed(config.seed[0])  # for reproducibility
  train_loader, val_loader, test_loader = get_data(config, val_pc=0.15, training=False)

  model = [
    torch.load(osp.join(config.models_root, "%s_%d_%s.pytorch" % (config.data, s, config.model)))[
      "model"].eval() for s in config.seed]
  acc = [
    torch.load(osp.join(config.models_root, "%s_%d_%s.pytorch" % (config.data, s, config.model)))[
      "acc"] for s in config.seed]
  if len(config.seed) == 1:
    config.seed = config.seed[0]
    model = model[0]
    acc = acc[0]
  else:
    raise NotImplementedError

  inspect_weights(model)

  # make our special val/test set - with much larger stdev, because on regular data it gets 100% acc
  # this is used to pick delta/p
  val_noise = 0.05
  num_val = len(val_loader.dataset)
  x_val, y_val = sk_datasets.make_moons(n_samples=num_val, shuffle=True, noise=val_noise,
                                        random_state=config.seed)
  x_val, y_val = torch.tensor(x_val, dtype=torch.float), torch.tensor(y_val)
  val_data = [(x_val[i], y_val[i]) for i in range(num_val)]

  max_abs = max([tup[0].abs().max() for tup in train_loader.dataset])
  rad = int(np.ceil(max_abs * config.radius_mult))  # some padding, 1.5
  print("grid is anchored: [%s, %s]" % (-rad, rad))

  num_test_side = 501  # 101 # evenly sampled across entire surface of grid
  x_test = -rad + 2 * rad * torch.arange(num_test_side,
                                         dtype=torch.float) / num_test_side  # -rad to rad
  x_test_0 = x_test.unsqueeze(0).repeat(num_test_side, 1).unsqueeze(2)
  x_test_1 = x_test.unsqueeze(1).repeat(1, num_test_side).unsqueeze(2)
  x_test = torch.cat([x_test_0, x_test_1], dim=2)  # side, side, 2
  x_test = x_test.view(num_test_side ** 2, 2)

  x_test_inds = torch.arange(num_test_side, dtype=torch.float)
  x_test_inds_0 = x_test_inds.unsqueeze(0).repeat(num_test_side, 1).unsqueeze(2)
  x_test_inds_1 = x_test_inds.unsqueeze(1).repeat(1, num_test_side).unsqueeze(2)
  x_test_inds = torch.cat([x_test_inds_0, x_test_inds_1], dim=2)
  x_test_inds = x_test_inds.view(num_test_side ** 2, 2)

  x_all = torch.cat([x_test, x_test_inds], dim=1)
  assert (x_all.shape == (
  num_test_side ** 2, 4))  # [0, 0] in inds corresponds to [-rad, -rad] in data (coords)
  test_data = [(x_all[i], -1) for i in range(num_test_side ** 2)]

  val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size,
                                           shuffle=False, num_workers=config.workers,
                                           pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                            shuffle=False, num_workers=config.workers,
                                            pin_memory=True)

  # render val
  render_two_moons(config, [val_loader], model_suff="", suff="val")

  # Store precomputations
  model, method_variables = globals()["%s_pre" % config.method](config, model, train_loader,
                                                                val_loader)

  # Run through test data batches, pass each batch to metric method along with needed params,
  # get metrics back, store with ground truth
  polytope_colours = {}
  polytope_img = np.zeros((num_test_side, num_test_side, 3),
                          dtype=np.uint8)  # x axis is first dim, as usual. 0,0 = bottom left
  unreliability_img = np.zeros((num_test_side, num_test_side), dtype=np.float)
  for batch_i, (data, targets) in enumerate(test_loader):
    print("batch %d / %d, %s" % (batch_i, len(test_loader), datetime.now()))
    stdout.flush()

    inputs = data[:, :2]
    inds = data[:, 2:]
    inputs, targets = inputs.to(device(config.cuda)), targets.to(device(config.cuda))

    if config.method == "subfunctions":
      unreliability_i, corrects_i, polytopes_i = globals()["%s_metric" % config.method](config,
                                                                                        method_variables,
                                                                                        model,
                                                                                        inputs,
                                                                                        targets,
                                                                                        get_polytope_ids=True)
    else:
      raise NotImplementedError
    # unreliability.append(unreliability_i)
    # corrects.append(corrects_i)

    for j in range(data.shape[0]):
      polytope_str = bool_tensor_content_hash(polytopes_i[j])

      if not polytope_str in polytope_colours:
        polytope_colours[polytope_str] = random_colour()
      assert (inds[j][0] == int(inds[j][0]) and inds[j][1] == int(inds[j][1]))

      # flip the axes - x axis is actually second dimension of image!
      polytope_img[int(inds[j][1]), int(inds[j][0]), :] = polytope_colours[polytope_str]
      unreliability_img[int(inds[j][1]), int(inds[j][0])] = unreliability_i[j].item()
  print("polytope colours sz: %s" % len(polytope_colours))

  print("unreliability range: %s, %s" % (unreliability_img.max(), unreliability_img.min()))

  # unreliability = torch.cat(unreliability)
  # corrects = torch.cat(corrects)

  # ------------------------------------------------------------------------------------------------
  # Draw the plots
  # ------------------------------------------------------------------------------------------------

  render_space = 1.25

  save_prefix = "%s_%s_%s_%s" % (config.data, config.seed, config.method, config.suff)

  # 1. Original labelled data rendered (2D). Green & purple?
  fig, ax = plt.subplots(1, figsize=(4, 4))
  ax.set_xlim(-rad, rad)
  ax.set_ylim(-rad, rad)
  xs_0 = [train_loader.dataset[j][0][0].item() for j in range(len(train_loader.dataset))]
  xs_1 = [train_loader.dataset[j][0][1].item() for j in range(len(train_loader.dataset))]
  ys = [train_loader.dataset[j][1].item() for j in range(len(train_loader.dataset))]
  colours = [["tab:green", "tab:purple"][c] for c in ys]
  ax.scatter(xs_0, xs_1, c=colours)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlabel("Input dim 0", fontsize=14)
  ax.set_ylabel("Input dim 1", fontsize=14)
  fig.tight_layout()
  fig.savefig(osp.join(config.models_root, "%s.png" % cleanstr("%s_results_0" % save_prefix)))

  # 2. Original labelled data underneath original flat model error - single layer contour plot (3D)
  fig, ax = plt.subplots(1, figsize=(4, 4), subplot_kw={"projection": "3d"})

  xx, yy = np.meshgrid(np.linspace(0, num_test_side - 1, num_test_side),
                       np.linspace(0, num_test_side - 1, num_test_side))  # -1 in middle!!
  X = xx
  Y = yy
  whole_model_metric = (1. - acc)
  if not config.no_bound:
    whole_model_metric += torch.sqrt(
      torch.log(2. / method_variables["delta"]) / (2. * method_variables["m"])).item()
    if not config.no_log:
      whole_model_metric = np.log(whole_model_metric)

  Z_unreliability = whole_model_metric * np.ones(X.shape)

  print("whole_model_metric %s" % whole_model_metric)
  print("Z_unreliability %s" % np.unique(Z_unreliability))

  print("orig model acc: %s" % acc)
  ax.plot_surface(X, Y, Z_unreliability, color="tab:blue") #cmap="Blues")  # linewidth=0, shade=False
  ax.scatter(rescale(xs_0, rad, num_test_side), rescale(xs_1, rad, num_test_side),
             [Z_unreliability.max() * render_space] * len(xs_0), c=colours)
  ax.set_xticks([])
  ax.set_yticks([])
  # ax.set_zticks([])
  ax.set_xlabel("Input dim 0", labelpad=0, fontsize=14)
  ax.set_ylabel("Input dim 1", labelpad=0, fontsize=14)

  if (not config.no_bound) and (not config.no_log):
    # ax.text2D(0.05, 0.95, "(logscale)", transform=ax.transAxes)
    ax.set_zlabel("true error bound \n(log)", labelpad=20, fontsize=12)
  else:
    ax.set_zlabel("true error bound", labelpad=20, fontsize=12)

  # ax.tick_params(axis='z', labelrotation=45)
  ax.grid(False)
  # fig.tight_layout()
  # plt.autoscale()
  plt.subplots_adjust(left=0., right=0.8, bottom=0.1, top=0.95)  # as pc of full figure size!
  fig.savefig(osp.join(config.models_root, "%s.png" % cleanstr("%s_results_1" % save_prefix)))

  # 3. Original data rendered with polytope identity (2D).
  # black data dots. polytopes colourful, imshow nearest - origin LOWER
  fig, ax = plt.subplots(1, figsize=(4, 4))
  ax.imshow(polytope_img, origin="lower",
            interpolation="antialiased")  # axis from 0 to num_test_side - 1

  grey = ["grey" for c in ys]
  # ax.scatter(rescale(xs_0, rad, num_test_side), rescale(xs_1, rad, num_test_side), c=grey,
  # alpha=0.01) # no shadow
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlabel("Input dim 0", fontsize=14)
  ax.set_ylabel("Input dim 1", fontsize=14)
  fig.tight_layout()
  fig.savefig(osp.join(config.models_root, "%s.png" % cleanstr("%s_results_2" % save_prefix)))

  # 4. Original labelled data underneat polytope error (3D) - colourful contour plot
  fig, ax = plt.subplots(1, figsize=(4, 4), subplot_kw={"projection": "3d"})
  plt.grid(b=None)

  xx, yy = np.meshgrid(np.linspace(0, num_test_side - 1, num_test_side),
                       np.linspace(0, num_test_side - 1, num_test_side))
  X = xx
  Y = yy
  assert (X.shape == unreliability_img.shape)

  # unreliability_img = - unreliability_img # invert
  ax.plot_surface(X, Y, unreliability_img, cmap=cm.coolwarm, linewidth=0)
  ax.scatter(rescale(xs_0, rad, num_test_side), rescale(xs_1, rad, num_test_side),
             [-unreliability_img.max() * render_space] * len(xs_0),
             c=colours)
  # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
  ax.set_xticks([])
  ax.set_yticks([])
  # ax.set_zticks([])
  ax.set_xlabel("Input dim 0", labelpad=0, fontsize=14)
  ax.set_ylabel("Input dim 1", labelpad=0, fontsize=14)

  if (not config.no_bound) and (not config.no_log):
    # ax.text2D(0.05, 0.95, "(logscale)", transform=ax.transAxes)
    ax.set_zlabel("true error bound \n (log)", labelpad=20, fontsize=12)
  else:
    ax.set_zlabel("true error bound", labelpad=20, fontsize=12)

  ax.grid(False)
  # fig.tight_layout()
  # plt.autoscale()
  plt.subplots_adjust(left=0., right=0.8, bottom=0.1, top=0.95)  # as pc of full figure size!
  fig.savefig(osp.join(config.models_root, "%s.png" % cleanstr("%s_results_3" % save_prefix)))

  # 5. 4 but 2D.
  # print four corners values
  # No shadow!

  print("corners:")
  print((unreliability_img[0, 0],
         unreliability_img[0, unreliability_img.shape[1] - 1],
         unreliability_img[unreliability_img.shape[1] - 1, 0],
         unreliability_img[unreliability_img.shape[1] - 1, unreliability_img.shape[1] - 1]))

  fig, ax = plt.subplots(1, figsize=(4, 4))
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)

  # cmap=cm.coolwarm
  im = ax.imshow(unreliability_img, origin="lower",
                 interpolation="antialiased")  # axis from 0 to num_test_side - 1
  # ax.scatter(rescale(xs_0, rad, num_test_side), rescale(xs_1, rad, num_test_side), c=grey,
  # alpha=0.01)
  cbar = fig.colorbar(im, cax=cax, orientation='vertical')

  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlabel("Input dim 0", fontsize=14)
  ax.set_ylabel("Input dim 1", fontsize=14)

  if (not config.no_bound) and (not config.no_log):
    # ax.set_title("(logscale)")
    cbar.ax.set_ylabel("true error bound (log)", rotation=90, labelpad=10, fontsize=14)
  else:
    cbar.ax.set_ylabel("true error bound", rotation=90, labelpad=10, fontsize=14)

  fig.tight_layout()
  # plt.autoscale()
  figstr = osp.join(config.models_root, "%s.png" % cleanstr("%s_results_4" % save_prefix))
  fig.savefig(figstr)

  plt.close("all")
  print("Saved to: %s*" % osp.join(config.models_root, save_prefix))


def rescale(x, rad, num_test_side):
  if isinstance(x, list): x = np.array(x)
  x = (x + rad) / (2 * rad)  # [0, 1]
  return x * num_test_side


if __name__ == "__main__":
  two_moons()
