from scripts.slurm.cifar.constants import *
import os.path as osp
from scripts.global_constants import *
import torch
import numpy as np
from decimal import Decimal
from collections import defaultdict
from util.general import cleanstr, cleanstr_latex
from collections import OrderedDict


def encode(tup):
  return str(tup)

verbose = True

methods_pretty = {
  "subfunctions": "Subfunctions (ours)",
  "max_response": "Max response",
  "entropy": "Entropy",
  "margin": "Margin",
  "class_distance": "Class distance",
  "explicit_density": "Residual flows density",
  "gaussian_process": "GP",
  "tack_et_al": "Cluster distance",
  "dropout": "MC dropout"
}

models_pretty = {"vgg16_bn": "VGG16", "resnet50model": "ResNet50"}
data_pretty = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "svhn": "SVHN"}


# cols: ood auroc, in distr aucea, in distr auroc, average
# values: performance relative to best performer (highest best)

# average over datasets (incl src/target for OOD) and models

methods = ["explicit_density", "gaussian_process", "class_distance", "margin", "entropy", "max_response", "dropout", "tack_et_al", "subfunctions"]
task_metrics = {"in_distr": ["AUROC"], "out_of_distr": "AUROC"} # "AUCEA",
suff = ""


# Work out best results for each exps set ----------------------------------------------------------

bests = {} # exps_set name -> value
bests_names = {} # exps_set name -> method

for task_type, metric in [("out_of_distr", "AUROC"), ("in_distr", "AUROC")]: # , ("in_distr", "AUCEA")

  for method in methods:

    if task_type == "out_of_distr":

      # get OOD average
      for data in datasets:
        for new_i, (new_distr_data, _) in enumerate(ood_datasets[data]):
          for model, _ in models_ps:  # col 2 2nd

            exps_set = encode((task_type, metric, data, new_distr_data, model)) # not incl model

            if verbose: print("(Bests) Considering OOD exps %s" % (exps_set))

            # get average over seeds
            metric_results = []
            for seed in seeds:
              fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
                "ood_%s_to_%s_%s_%s_%s_%s_results" % (
                  data, new_distr_data, seed, method, model, suff)))

              if osp.exists(fname):
                results = torch.load(fname)
                metric_results.append(results[metric])

            result = np.array(metric_results).mean()

            if not exps_set in bests or result > bests[exps_set]:
              bests[exps_set] = result
              bests_names[exps_set] = method

    elif task_type == "in_distr":

      for model, _ in models_ps:  # col 1st

        exps_set = encode((task_type, metric, data, model))

        if verbose: print("(Bests) Considering ID exps %s" % (exps_set))

        metric_results = []
        for seed in seeds:
          fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
            "%s_%s_%s_%s_%s_results" % (data, seed, method, model, suff)))

          if osp.exists(fname):
            results = torch.load(fname)
            metric_results.append(results[metric])

        result = np.array(metric_results).mean()

        if not exps_set in bests or result > bests[exps_set]:
          bests[exps_set] = result
          bests_names[exps_set] = method

if verbose:
  print("(Bests) Best methods")
  print(bests_names)
  print()
  print("(Bests) Best results")
  print(bests)


print("\n-------------------------\n")



# Work out main table contents ---------------------------------------------------------------------

print_grid = defaultdict(list) # method, cols

best_method_per_col = []

for task_type, metric in [("out_of_distr", "AUROC"), ("in_distr", "AUROC")]: # cols # , ("in_distr", "AUCEA")

  curr_best = - np.inf
  curr_best_method = None

  for method in methods:

    metric_results = []

    if task_type == "out_of_distr":

      # get OOD average
      for data in datasets:
        for new_i, (new_distr_data, _) in enumerate(ood_datasets[data]):
          for model, _ in models_ps:  # col 2 2nd

            exps_set = encode((task_type, metric, data, new_distr_data, model)) # not incl model

            if verbose: print("Considering OOD exps %s" % (exps_set))

            avg_metric = []
            # get average over seeds
            for seed in seeds:
              fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
                "ood_%s_to_%s_%s_%s_%s_%s_results" % (
                  data, new_distr_data, seed, method, model, suff)))

              if osp.exists(fname):
                results = torch.load(fname)

                avg_metric.append(results[metric])
                # subtract from the best for this exps set
                # metric_results.append(results[metric] - bests[exps_set])

            avg_metric = np.array(avg_metric).mean() # average perf over seeds

            metric_results.append(avg_metric - bests[exps_set])

    elif task_type == "in_distr":

      for model, _ in models_ps:  # col 1st

        exps_set = encode((task_type, metric, data, model))

        if verbose: print("Considering ID exps %s" % (exps_set))

        avg_metric = []
        for seed in seeds:
          fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
            "%s_%s_%s_%s_%s_results" % (data, seed, method, model, suff)))

          if osp.exists(fname):
            results = torch.load(fname)

            avg_metric.append(results[metric])
            # subtract from the best for this exps set
            #metric_results.append(results[metric] - bests[exps_set])
        avg_metric = np.array(avg_metric).mean()  # average perf over seeds

        metric_results.append(avg_metric - bests[exps_set])


    metric_results = np.array(metric_results)
    if len(metric_results) > 0:
      #if not len(metric_results) == len(seeds):
      #  print("Fewer values than expected")
      #  raise NotImplementedError

      result = metric_results.mean()
      print_grid[method].append((result, metric_results.std()))
      # "%.3f $\pm$ %.0E" % (result, Decimal("%.16f" % metric_results.std()))

      # check if it's the best in its column
      if result > curr_best:
        curr_best = result
        curr_best_method = method

    else:
      print("No metrics")
      raise NotImplementedError

  best_method_per_col.append(curr_best_method)


# add average metric for all methods (over all existing columns)
curr_best = - np.inf
curr_best_method = None
for method in methods:
  avg_mean, avg_std = [], []
  for m, s in print_grid[method]:
    avg_mean.append(m)
    avg_std.append(s)

  result = np.array(avg_mean).mean()
  print_grid[method].append((result, np.array(avg_std).mean()))

  if result > curr_best:
    curr_best = result
    curr_best_method = method

best_method_per_col.append(curr_best_method)


# print print_grid row by row (method), highlighting the best in each column

print_str = ""
for task_type, metric in [("out_of_distr", "AUROC"), ("in_distr", "AUROC")]: # cols # ("in_distr", "AUCEA")
  print_str += "& %s %s " % (task_type, metric)
print_str += "& average "
print_str += "\\\\"

print(print_str)

for method in methods:
  print_str = "%s" % methods_pretty[method]

  for i, (m, s) in enumerate(print_grid[method]):
    print_str_curr = "%.3f $\pm$ %.0E" % (m, Decimal("%.16f" % s))

    if method == best_method_per_col[i]:
      print_str_curr = "\\textbf{%s}" % print_str_curr

    print_str += " & %s " % print_str_curr

  print_str += "\\\\"

  print(print_str)












