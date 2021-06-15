from scripts.global_constants import *
from scripts.slurm.cifar.constants import *
from scripts.analysis.print_results import suff, methods_pretty, models_pretty, data_pretty
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from colorsys import hsv_to_rgb
import torch
import os.path as osp
import numpy as np
import argparse
from util.general import set_seed, cleanstr, sort_results

# ood, auroc
plt.rcParams['font.size'] = 16

print_methods = ["gaussian_process", "explicit_density", "class_distance", "margin", "max_response", "entropy", "subfunctions"]


def show_legend_plot(data, new_distr_data, model):
  return data == "cifar10" and new_distr_data == "svhn" and model == "vgg16_bn"

np.random.seed(0)
num_divs = 1000 # for averaging the contour lines
metric_range = np.linspace(0., 1., num=num_divs, endpoint=True)
turn_on_scatter = False
num_stddevs = 1
transparency = 0.1 #0.15 if num_stddevs == 1 else 0.1

orig_colours = [np.array(hsv_to_rgb(hue, 0.6, 0.7)) for hue in np.linspace(0., 1., num=(len(print_methods) + 6))]
perm = [  11, 6, 4, 10, 2, 1, 8, 7,  9,  3,  0,  5, 12] #np.random.permutation(len(orig_colours))
colours = [orig_colours[i] for i in perm]

plot_ind = 0
for data in datasets:
  for new_i, (new_distr_data, _) in enumerate(ood_datasets[data]):
    for model, _ in models_ps:

      fig, ax = plt.subplots(1, figsize=(4.15, 4))

      for method_i, method in enumerate(print_methods):
        y_all = None

        for seed in seeds:
          fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
            "ood_%s_to_%s_%s_%s_%s_%s_results" % (
              data, new_distr_data, seed, method, model, suff)))

          if osp.exists(fname):
            results = torch.load(fname)
            xs_orig, ys_orig = results["fprs"], results["tprs"]
            xs_orig, ys_orig = sort_results(xs_orig, ys_orig)

            if turn_on_scatter: ax.scatter(xs_orig, ys_orig, color=colours[method_i])

            f = interpolate.interp1d(xs_orig, ys_orig)
            ys_new = f(metric_range)

            if y_all is None:
              y_all = np.expand_dims(ys_new, axis=0)
            else:
              y_all = np.concatenate([y_all, np.expand_dims(ys_new, axis=0)], axis=0)

        ys = y_all.mean(axis=0)
        ys_std = y_all.std(axis=0)
        xs = metric_range

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        ax.plot(xs, ys, "-", color=colours[method_i], label=methods_pretty[method], linewidth=2)
        ax.set_xlabel("FPR") # fontsize=14
        ax.set_ylabel("TPR") # , fontsize=14
        ax.set_xlim([0., 1.])
        ax.set_ylim([0., ys.max()])

        # plot stddevs
        for std in range(1, num_stddevs + 1): #3 + 1):
          ax.fill_between(xs, ys - std * ys_std,
                          ys + std * ys_std, alpha=((1 + num_stddevs - std) * transparency),
                          edgecolor=colours[method_i], facecolor=colours[method_i])

      #if show_legend_plot(data, new_distr_data, model):
      #  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
      #ax.legend(loc="lower right", fontsize=14)

      fig.suptitle(r"$\rightarrow$%s, %s" % (data_pretty[new_distr_data], models_pretty[model]), y=0.97) # , fontsize=14 ,  y=1.01
      #fig.tight_layout()
      plt.subplots_adjust(bottom=0.15, left=0.19, right=0.925, top=0.9)

      plots_fname = osp.join(DEFAULT_MODELS_ROOT,
                             "%s.png" % cleanstr("ood_%s_to_%s_%s_summary_plots" % (data, new_distr_data, model)))
      print("saving to %s" % plots_fname)
      fig.savefig(plots_fname)
      plt.close("all")

      plot_ind += 1


fig_leg, ax_leg = plt.subplots(1, figsize=(2.8, 2.5))
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', fontsize=12)
ax_leg.axis('off')
legend_fname = osp.join(DEFAULT_MODELS_ROOT, "%s.png" % cleanstr("ood_summary_legend"))
fig_leg.savefig(legend_fname)
plt.close("all")
