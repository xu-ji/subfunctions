import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
from util.general import cleanstr


def render_two_moons(config, loaders, model_suff="", suff=""):
  assert config.data == "two_moons"
  all_colours = [["tab:red", "tab:green"],
                 ["tab:pink", "tab:olive"]]  # e.g. [train colours, test colours]

  fig, ax = plt.subplots(1, figsize=(4, 4))

  for i, loader in enumerate(loaders):
    min_x = np.inf
    max_x = -np.inf
    for x, _ in loader.dataset:
      min_x = min(x.min().item(), min_x)
      max_x = max(x.max().item(), max_x)
    print("(render_two_moons) min x %s max x %s" % (min_x, max_x))

    dataset = loader.dataset
    xs_0 = [dataset[j][0][0].item() for j in range(len(dataset))]
    xs_1 = [dataset[j][0][1].item() for j in range(len(dataset))]

    ys = [dataset[j][1].item() for j in range(len(dataset))]
    colours = [all_colours[i][c] for c in ys]
    ax.scatter(xs_0, xs_1, c=colours)

  fig.savefig(osp.join(config.models_root, "%s.png" % cleanstr(
    "%s_%s_%s_%s_render_training" % (config.data, config.seed, model_suff, suff))))
  plt.close("all")
