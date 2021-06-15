import os.path as osp
import pickle
from collections import OrderedDict

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
from sklearn import metrics
from colorsys import hsv_to_rgb
from datetime import datetime
import arch
import random
import os

def print_first_labels(loader, n=10):
  for xs, ys in loader:
    print("(print_first_labels) %s" % ys[:n])
    break


def set_seed(seed):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def device(cuda):
  if cuda:
    return torch.device("cuda:0")
  else:
    return torch.device("cpu")


def sort_and_extend(xs, ys):
  # order by xs
  xs, ys = sort_results(xs, ys)

  # extend xs range to [0, 1] with nearest interp
  if xs[0] > 0:
    xs = np.concatenate([np.array([0.]), xs])
    ys = np.concatenate([np.array([ys[0]]), ys])
  if xs[-1] < 1:
    xs = np.concatenate([xs, np.array([1.])])
    ys = np.concatenate([ys, np.array([ys[-1]])])

  return xs, ys


def sort_results(xs, ys):
  # order by xs
  assert len(xs.shape) == 1 and len(ys.shape) == 1 and (ys >= 0).all()
  order = np.argsort(xs)
  xs = xs[order]
  ys = ys[order]

  return xs, ys


def compute_area(xs, ys, mode, ys_max=None):
  # Summarize model skill using trapezium rule
  # xs should be ordered, and same range across methods for fairness. ys are non-negative
  # sklearn.metrics.auc: area under ys = area to 0. xs range is unchanged.
  assert isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray)

  xs, ys = sort_results(xs, ys)

  assert xs[0] == 0. and xs[-1] == 1.

  if mode == "under":
    return metrics.auc(x=xs, y=ys)
  elif mode == "over":
    ys = ys_max - ys  # flip the curve vertically
    return metrics.auc(x=xs, y=ys)


def interpolate_acc(accs, coverages, coverage_val):
  # find closest coverage vals on either side of coverage_val and linearly interpolate acc
  if coverages.max() >= coverage_val and coverages.min() < coverage_val:
    coverage_is = np.argsort(coverages)
    coverages_ord = coverages[coverage_is]
    fst_gte_ii = np.argmax(coverages_ord >= coverage_val)
    left_i = coverage_is[fst_gte_ii - 1]
    right_i = coverage_is[fst_gte_ii]
    assert coverages[left_i] <= coverages[right_i]
    acc = (coverage_val - coverages[left_i]) / (coverages[right_i] - coverages[left_i]) * (
    accs[right_i] - accs[left_i]) + accs[left_i]
  else:
    # if only one side found, assign the acc of single closest i.e. the found side.
    print("(interpolate_acc) Warning, doing closest rather than linear interpolation")
    closest_i = (np.abs(coverages - coverage_val)).argmin()
    acc = accs[closest_i]

  return acc


def plot_graph(axarr, i, xs, ys, xlabel, ylabel, title="", color="tab:blue", label="", marker="o-"):
  # scatter, and range of x should be [0, 1]
  assert (xs >= 0.).all() and (xs <= 1.).all()
  # axarr[i].scatter(xs, ys)
  order = np.argsort(xs)
  xs = xs[order]
  ys = ys[order]
  axarr[i].plot(xs, ys, marker, color=color, label=label)
  axarr[i].set_title(title)
  axarr[i].set_xlabel(xlabel)
  axarr[i].set_ylabel(ylabel)
  axarr[i].set_xlim([0., 1.])
  axarr[i].set_ylim([0., ys.max()])


def minimal(config, method_variables):
  new_method_variables = {}
  for k, v in method_variables.items():
    if config.data != "mnist":
      if not (isinstance(v, int) or isinstance(v, float)):
        continue
    new_method_variables[k] = v
  return new_method_variables


def evaluate(config, model, test_loader):
  model.eval()
  correct = 0
  total = 0
  for batch_i, (imgs, targets) in enumerate(test_loader):
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
    preds = model(imgs)
    preds_flat = preds.argmax(dim=1)

    correct += preds_flat.eq(targets).sum().item()
    total += imgs.shape[0]

  acc = correct / float(total)
  # print("Test acc: %f" % acc)
  model.train()
  return acc


def lr_sched_maker(lr_sched_epoch_gap, lr_sched_mult, epoch):
  # if epoch % 30 == 0: return 0.1
  if epoch % lr_sched_epoch_gap == 0:
    return lr_sched_mult
  else:
    return 1.


def add_hooks_sizes(model, curr_name):
  for i, (name, m) in enumerate(model.named_children()):
    full_name = "%s_%s" % (curr_name, name)
    if hasattr(m, "named_children") and len(list(m.named_children())) > 0:
      add_hooks_sizes(m, full_name)  # recurse
    else:
      m.register_forward_hook(print_sz_fn_maker(full_name))


def print_sz_fn_maker(full_name):
  def print_sz(module, input, output):
    if isinstance(input, tuple):
      assert (len(input) == 1)
      input = input[0]
    if isinstance(output, tuple):
      assert (len(output) == 1)
      output = output[0]
    print("(%s) %s -> %s" % (full_name, input.shape, output.shape))

  return print_sz


"""
def sum_p2(p, N):
  i = 0
  last_term = p * np.power(1 - p, i) * comb(N, i)
  print(("should be equal:", last_term, p))
  for i in range(1, N):
    part = last_term * (1. - p) * (N - i + 1.) / i
    last_term += part
    print((i, last_term))
    if not np.isfinite(last_term): break
    
"""


def cleanstr(s):
  assert isinstance(s, str)
  s = re.sub(r"[^\w\s]", "", s)  # remove non alphanumeric
  s = re.sub(r"\s+", "_", s)  # whitespace
  return s


def cleanstr_latex(s):
  assert isinstance(s, str)
  s = re.sub(r"_", " ", s)  # into whitespace
  return s


# [list((np.array(hsv_to_rgb(hue, 0.5, 0.8)) * 255.).astype(np.uint8)) for hue in hues]
def random_colour(integer=True):  # in rgb!
  hue = np.random.rand()
  if integer:
    return (np.array(hsv_to_rgb(hue, 0.7, 0.7)) * 255.).astype(np.uint8)  # 0.6, 0.7
  else:
    return np.array(hsv_to_rgb(hue, 0.7, 0.7))


def inspect_weights(model):
  print("Inspecting model")
  # print the avg norm of the incoming weights for each node
  # print the bias for each layer
  for n, p in model.named_parameters():
    if "bias" in n:
      print("%s & $%s$ & & & %.3f & \\\\" %
            (n.replace("_", "\\_"), list(torch.tensor(p.data.shape).numpy()),
             torch.norm(p.data, p=2).item()))
    elif "weight" in n:
      assert (len(p.data.shape) == 2)
      norms = torch.norm(p.data, dim=1, p=2)
      assert (norms.shape == (p.data.shape[0],))
      print("%s & $%s$ & %.3f & %.3f & %.3f & %.3f \\\\" %
            (n.replace("_", "\\_"), list(torch.tensor(p.data.shape).numpy()),
             norms.min().item(), norms.max().item(), norms.mean().item(), norms.std().item()))
    else:
      raise NotImplementedError


def augment_command(cmd, compute, gpu=None, suff="", pre_suff=""):
  if compute == "slurm":
    cmd = ("sbatch %s scripts/slurm/%s \"" % (pre_suff, suff)) + cmd + "\""
  elif compute == "local":
    cmd = ("export CUDA_VISIBLE_DEVICES=%d && nohup " % gpu) + cmd + (" > %s_.out &" % suff)

  return cmd


def check_load(fname, thresh=datetime(2021, 4, 11, 0, 0, 0, 0)):
  d = datetime.fromtimestamp(osp.getmtime(fname))
  if d < thresh: print(fname)
  assert d >= thresh
  return torch.load(fname)


def add_spectral_norm_last(model):
  if isinstance(model, arch.ResNetModel) or isinstance(model, arch.VGG):

    if isinstance(model, arch.ResNetModel):
      print("Adding spectral norm to resnet model")
      old_seq = model.linear_feats
    else:
      print("Adding spectral norm to vgg model")
      old_seq = model.classifier

    next_ind = 0
    new_seq = []
    for ind, mod in old_seq._modules.items():
      print((ind, mod.__class__))
      assert ind == str(next_ind)
      next_ind += 1
      new_mod = mod
      if hasattr(mod, "weight"):
        new_mod = torch.nn.utils.spectral_norm(mod, name="weight")
        print("(added)")
      new_seq.append(new_mod)

    assert len(new_seq) == len(old_seq)

    if isinstance(model, arch.ResNetModel):
      model.linear_feats = torch.nn.Sequential(*new_seq)
      model.fc = torch.nn.utils.spectral_norm(model.fc, name="weight")
    else:
      model.classifier = torch.nn.Sequential(*new_seq)

  elif isinstance(model, arch.ResNetModel):
    print("Adding spectral norm to resnet model")
    model.fc = torch.nn.utils.spectral_norm(model.fc, name="weight")

  else:
    raise NotImplementedError

  return model


def get_norms(model, verbose=True):
  metric_feats = []
  metric_feats_convs = []
  metric_lin = []
  metric_all = []
  still_feats = True
  for n, p in model.named_parameters():
    # if verbose: print(n)
    data = p.data
    if still_feats and (len(p.data.shape) == 2):
      still_feats = False
      if verbose: print("moved into linear layers! at %s" % n)

    if still_feats:
      res = metric_feats
    else:
      res = metric_lin

    if len(data.shape) > 1:
      data = data.flatten(start_dim=1)
    else:
      data = data.unsqueeze(1)

    per_out_norms = torch.norm(data, dim=1, p=2, keepdim=False)  # one per output dim
    assert per_out_norms.shape == (data.shape[0],)
    # curr_std = torch.std(per_out_norms) # stddev across output nodes
    curr_metric = per_out_norms.mean()
    assert len(curr_metric.shape) == 0

    res.append(curr_metric)
    metric_all.append(curr_metric)
    if still_feats and len(p.data.shape) == 4: metric_feats_convs.append(curr_metric)

  metric_feats = torch.stack(metric_feats).mean().item()
  metric_lin = torch.stack(metric_lin).mean().item()
  metric_all = torch.stack(metric_all).mean().item()
  metric_feats_convs = torch.stack(metric_feats_convs).mean().item()

  return metric_feats, metric_lin, metric_all, metric_feats_convs


def eval_and_store(config, unreliability, corrects, method_variables, store_fname_prefix):
  # For every threshold (n divisions from min to max), compute all metrics. Obviously ignore inf
  # bounds when finding max thresh.
  unreliability_finite = unreliability[unreliability.isfinite()]
  if unreliability_finite.shape[0] == 0:
    print("All test samples had unseen patterns. Aborting.")
    exit(0)

  min_unrel = unreliability_finite.min().item()
  max_unrel = unreliability_finite.max().item()
  print("Threshold range: %s" % str((min_unrel, max_unrel)))

  # loop through min and max finite bounds, and also +/- infinity to tie graphs to [0, 1] corners
  tprs, fprs = np.zeros(config.threshold_divs + 1 + 2, dtype=np.float), np.zeros(
    config.threshold_divs + 1 + 2, dtype=np.float)
  coverages = np.zeros(config.threshold_divs + 1 + 2, dtype=np.float)
  effective_acc_or_precisions = np.zeros(config.threshold_divs + 1 + 2, dtype=np.float)
  recalls = np.zeros(config.threshold_divs + 1 + 2, dtype=np.float)

  for t_i in range(config.threshold_divs + 1 + 2):
    if t_i == 0:
      t = -np.inf
    elif t_i <= config.threshold_divs + 1:
      t = min_unrel + ((t_i - 1) / float(config.threshold_divs)) * (max_unrel - min_unrel)
    else:
      assert t_i == config.threshold_divs + 2
      t = np.inf

    accepts = unreliability <= t

    # 2 variables for prediction: whether it was accepted, and whether it was correct
    tp = (accepts * corrects).sum().item()
    fp = (accepts * (~corrects)).sum().item()
    tn = ((~accepts) * (~corrects)).sum().item()
    fn = ((~accepts) * corrects).sum().item()

    tpr = tp / max(1., float(tp + fn))  # how many were accepted within correct model predictions
    fpr = fp / max(1., float(fp + tn))  # how many were accepted within incorrect model predictions
    coverage = (tp + fp) / float(tp + fp + tn + fn)  # how many we accepted
    effective_acc_or_precision = tp / max(1., float(
      tp + fp))  # how many were correct model predictions within the accepted
    recall = tpr  # tp / max(1., float(tp + fn))

    # The ranges of the xs used in the graphs need to have the same range across methods for
    # fairness.
    tprs[t_i] = tpr

    # should hit 0 for *just under* first t (first t close to 0). Max: 1, if accept all incorrect
    #  predictions i.e. for last t.
    fprs[t_i] = fpr

    # should hit 0 for *just under* first t (first t close to 0). Max: 1, for last t.
    coverages[t_i] = coverage

    # should hit 0 for *just under* first t (first t close to 0). Max: 1, if out of all accepted,
    #  all were correct. Probably will (for small t), but may not hit.
    effective_acc_or_precisions[t_i] = effective_acc_or_precision

    recalls[t_i] = recall


  AUCEA = compute_area(xs=coverages, ys=effective_acc_or_precisions, mode="under")
  AUROC = compute_area(xs=fprs, ys=tprs, mode="under")

  # Store
  results_fname = osp.join(config.models_root,
                           "%s.pytorch" % cleanstr("%s_results" % store_fname_prefix))

  winning_method_variables = {}
  if config.method == "subfunctions":
    for k in ["p", "p_i", "delta", "delta_results", "global_prob_norm",
              "best_metric"]:
      winning_method_variables[k] = method_variables[k]

  results = {"AUCEA": AUCEA, "AUROC": AUROC,

             "tprs": tprs, "fprs": fprs, "coverages": coverages,
             "effective_acc_or_precisions": effective_acc_or_precisions, "recalls": recalls,
             "max_unrel": max_unrel, "min_unrel": min_unrel,

             "winning_method_variables": winning_method_variables}
  torch.save(results, results_fname)

  fig, axarr = plt.subplots(2, figsize=(4, 2 * 4))
  plot_graph(axarr, 0, xs=coverages, ys=effective_acc_or_precisions, xlabel="Coverage",
             ylabel="Eff. Accuracy or Precision", title="Area under: %f" % AUCEA)
  plot_graph(axarr, 1, xs=fprs, ys=tprs, xlabel="FP rate", ylabel="TP rate",
             title="Area under: %f" % AUROC)

  fig.tight_layout(rect=[0, 0.03, 1, 0.95])
  plots_fname = osp.join(config.models_root, "%s.png" % cleanstr("%s_plots" % store_fname_prefix))
  fig.savefig(plots_fname)
  plt.close("all")

  print("AUCEA %s, AUROC %s" % (AUCEA, AUROC))
  print("Saved results to %s" % results_fname)
  print("Saved plots to %s" % plots_fname)
