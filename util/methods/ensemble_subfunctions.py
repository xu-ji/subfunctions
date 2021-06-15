from util.general import *
from util.methods.subfunctions import start_name, get_keywords, pop_maps_tensor, STEB, \
  add_hooks_tensor
import torch.nn.functional as F


def ensemble_subfunctions_pre(config, model, train_loader, val_loader):
  # This relies on precomputations from running the subfunctions method first. Do that method first.

  assert len(config.seed) > 1
  method_variables = {}
  base_method = "subfunctions"

  for model_curr in model:
    names, hook_handles = [], []
    add_hooks_tensor(model_curr, start_name, get_keywords(model_curr), names, hook_handles)

  for seed in config.seed:
    metric_values = np.zeros(len(config.search_ps), dtype=np.float)
    for p_i in range(len(config.search_ps)):
      saved_method_variables = torch.load(osp.join(config.models_root, "%s.pytorch" % cleanstr(
        "%s_%s_%s_%s_%s_method_variables_p_%d" % (
          config.data, seed, base_method, config.model, config.suff,
          p_i))))
      metric_values[p_i] = saved_method_variables["best_metric"]
    best_p_i = metric_values.argmax()

    if not np.isfinite(metric_values.max()):
      print("(subfunctions_pre) No hyperparameter was successful. Aborting.")
      exit(0)

    print("(subfunctions_pre) best i %s, all %s" % (best_p_i, list(metric_values)))

    method_variables_curr = torch.load(osp.join(config.models_root, "%s.pytorch" % cleanstr(
      "%s_%s_%s_%s_%s_method_variables_p_%d" % (
        config.data, seed, base_method, config.model, config.suff,
        best_p_i))))

    method_variables[seed] = method_variables_curr

  return model, method_variables


def ensemble_subfunctions_metric(config, method_variables, model, imgs, targets):
  assert isinstance(model, list) and len(model) > 1
  # get average over seeds

  unreliabilities = []
  res = []
  for seed in config.seed:
    method_variables_curr = method_variables[seed]

    with torch.no_grad():
      preds = model[seed](imgs)

    softmax_preds = F.softmax(preds, dim=1)
    res.append(softmax_preds)

    curr_patterns = pop_maps_tensor(config, model[seed],
                                    method_variables_curr["pattern_layer_names"])
    unreliability_curr = STEB(config, curr_patterns, method_variables_curr,
                              torch.tensor([method_variables_curr["delta"]],
                                           device=device(config.cuda))).squeeze(1)  # num samples
    unreliabilities.append(unreliability_curr)

  res = torch.stack(res, dim=0)  # num models, num samples, num classes
  avg_preds = res.mean(dim=0)  # num_samples, classes
  assert len(avg_preds.shape) == 2
  top_classes_preds, top_classes = avg_preds.max(dim=1)  # num_samples
  assert len(top_classes.shape) == 1 and len(top_classes_preds.shape) == 1
  correct = top_classes.eq(targets)

  unreliability = torch.stack(unreliabilities, dim=0)
  unreliability = unreliability.mean(dim=0)
  assert unreliability.shape == (imgs.shape[0],)

  return unreliability, correct
