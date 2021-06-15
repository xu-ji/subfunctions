from scripts.slurm.cifar.constants import *
import os.path as osp
from scripts.global_constants import *
import torch
import numpy as np
from decimal import Decimal
from collections import defaultdict
from util.general import cleanstr, cleanstr_latex

models_pretty = {"vgg16_bn": "VGG16", "resnet50model": "ResNet50"}
data_pretty = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "svhn": "SVHN"}

verbose = False
print_chosen_variables = False
print_original_accs = True

methods = ["explicit_density", "gaussian_process", "class_distance", "margin", "entropy", "max_response", "subfunctions"]
task_metrics = {"in_distr": ["AUCEA", "AUROC"], "out_of_distr": "AUROC"}
suff = ""

methods_pretty = {
  "subfunctions": "Subfunctions (ours)",
  "max_response": "Max response",
  "entropy": "Entropy",
  "margin": "Margin",
  "class_distance": "Class distance",
  "explicit_density": "Residual flows density",
  "gaussian_process": "GP"
}

ood_strings = {"cifar10": defaultdict(list), "cifar100": defaultdict(list)} # data -> model, method string -> results string (made of ood datasets)

def print_results():
  print("\n")
  print("metrics: %s" % str(task_metrics))

  # original accuracies of trained models
  if print_original_accs:
    for data in datasets:
      for model, _ in models_ps:
        # original
        printstr = "%s & %s " % (data_pretty[data], models_pretty[model])
        acc_results = []
        for seed in seeds:
          fname = osp.join(DEFAULT_MODELS_ROOT, "%s_%d_%s.pytorch" % (data, seed, model))
          if osp.exists(fname):
            model_file = torch.load(fname)
            acc_results.append(model_file["acc"])
        acc_results = np.array(acc_results)
        #print(acc_results)
        printstr += "& %.3f $\pm$ %.0E " % (acc_results.mean(), Decimal("%.16f" % acc_results.std()))
        printstr += "\\\\"
        print(printstr)

  # OOD results

  metric = task_metrics["out_of_distr"]
  for data in datasets:
    print("\nOOD table for %s \n" % data)

    printstr = ""
    for new_i, (new_distr_data, _) in enumerate(ood_datasets[data]):  # col 1st
      printstr += " & \multicolumn{2}{c}{$\\rightarrow$ %s} " % data_pretty[new_distr_data]
    printstr += "\\\\"
    print(printstr)

    print("\\midrule")

    printstr = ""
    for new_i, (new_distr_data, _) in enumerate(ood_datasets[data]):  # col 1st
      for model, _ in models_ps:  # col 2nd
        printstr += " & %s " % models_pretty[model]
    printstr += "\\\\"
    print(printstr)

    print("\\midrule")

    print_grid = defaultdict(list)
    best_score_per_col = defaultdict(int)
    best_method_per_col = defaultdict(str)

    for method in methods: # row
      print_grid[method].append("%s" % methods_pretty[method])

      col_ind = 1

      for new_i, (new_distr_data, _) in enumerate(ood_datasets[data]): # col 1st
        for model, _ in models_ps: # col 2 2nd

          # get average over seeds
          metric_results = []
          for seed in seeds:
            fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
              "ood_%s_to_%s_%s_%s_%s_%s_results" % (
                data, new_distr_data, seed, method, model, suff)))

            if osp.exists(fname):
              results = torch.load(fname)

              if "p" in results["winning_method_variables"] and print_chosen_variables:
                print("p: %s" % str(results["winning_method_variables"]["p"]))

              metric_results.append(results[metric])

          metric_results = np.array(metric_results)
          if len(metric_results) > 0:
            note = ""
            if verbose or not len(metric_results) == len(seeds):
              note = "(%d)" % len(metric_results)

            result = metric_results.mean()
            print_grid[method].append("%.3f $\pm$ %.0E %s" % (
              result, Decimal("%.16f" % metric_results.std()), note))

            if result > best_score_per_col[col_ind]:
              best_score_per_col[col_ind] = result
              best_method_per_col[col_ind] = method
          else:
            print_grid[method].append("-")

          col_ind += 1

    if verbose:
      print("best_method_per_col")
      print(best_method_per_col)

    # print table including bolding the best in each column
    for method in methods:
      printstr = ""
      for col_ind, col in enumerate(print_grid[method]):
        if col_ind > 0:
          printstr += " & "

        if best_method_per_col[col_ind] == method: printstr += " \\textbf{%s} " % col
        else: printstr += col

      printstr += "\\\\"
      print(printstr)

  # In distribution results

  for data in datasets:
    print("\nIn distribution table for %s \n" % data)

    printstr = ""
    for model, _ in models_ps: # col 1st
      printstr += " & \multicolumn{2}{c}{%s} " % models_pretty[model]
    printstr += "\\\\"
    print(printstr)

    print("\\midrule")

    printstr = ""
    for model, _ in models_ps:  # col 1st
      for metric in task_metrics["in_distr"]: # col 2nd
        printstr += " & %s " % metric
    printstr += "\\\\"
    print(printstr)

    print("\\midrule")

    print_grid = defaultdict(list)
    best_score_per_col = defaultdict(int)
    best_method_per_col = defaultdict(str)

    for method in methods: # row
      print_grid[method].append("%s" % methods_pretty[method])

      col_ind = 1

      for model, _ in models_ps:  # col 1st
        for metric in task_metrics["in_distr"]: # col 2nd
          metric_results = []

          for seed in seeds:
            fname = osp.join(DEFAULT_MODELS_ROOT, "%s.pytorch" % cleanstr(
              "%s_%s_%s_%s_%s_results" % (data, seed, method, model, suff)))

            if osp.exists(fname):
              results = torch.load(fname)

              if "p" in results["winning_method_variables"] and print_chosen_variables:
                print("p: %s" % str(results["winning_method_variables"]["p"]))

              metric_results.append(results[metric])

          metric_results = np.array(metric_results)
          if len(metric_results) > 0:
            note = ""
            if verbose or not len(metric_results) == len(seeds):
              note = "(%d)" % len(metric_results)

            result = metric_results.mean()
            print_grid[method].append("%.3f $\pm$ %.0E %s" % (
              result, Decimal("%.16f" % metric_results.std()), note))

            if result > best_score_per_col[col_ind]:
              best_score_per_col[col_ind] = result
              best_method_per_col[col_ind] = method
          else:
            print_grid[method].append("-")

          col_ind += 1

    if verbose:
      print("best_method_per_col")
      print(best_method_per_col)

    # print table including bolding the best in each column
    for method in methods:
      printstr = ""
      for col_ind, col in enumerate(print_grid[method]):
        if col_ind > 0:
          printstr += " & "

        if best_method_per_col[col_ind] == method:
          printstr += " \\textbf{%s} " % col
        else:
          printstr += col

      printstr += "\\\\"
      print(printstr)


if __name__ == "__main__":
  print_results()


