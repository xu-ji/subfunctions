import torch
import torch.nn.functional as F
from util.methods.subfunctions import start_name, get_keywords, fits_keyword
from sys import stdout
from datetime import datetime
from util.general import device
import numpy as np
from util.methods.tack_et_al import tack_et_al_pattern_keywords
from util.general import compute_aucea, device

def dropout_pre(config, model, train_loader, val_loader):
  # construct coreset with size 10% of samples
  method_variables = {}

  assert config.model in ["vgg16_bn", "resnet50model"]

  # find the best p
  best_aucea = -np.inf
  best_p = None
  for p in config.dropout_ps:
    method_variables["dropout_p"] = p

    names, hook_handles = [], []
    add_dropout_hooks(method_variables, model, start_name,
                      get_keywords(model, tack_et_al_pattern_keywords),
                      names, hook_handles)

    all_unreliability = []
    all_correct = []
    for batch_i, (imgs, targets) in enumerate(val_loader):
      imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
      unreliability, correct = dropout_metric(config, method_variables, model, imgs, targets)
      all_unreliability.append(unreliability)
      all_correct.append(correct)

    all_unreliability = torch.cat(all_unreliability, dim=0)
    all_correct = torch.cat(all_correct, dim=0)

    print("Shapes %s %s" % (all_unreliability.shape, all_correct.shape))

    aucea = compute_aucea(config, all_unreliability, all_correct)
    print("p %s aucea %s" % (p, aucea))
    if aucea > best_aucea:
      best_aucea = aucea
      best_p = p

    for h in hook_handles:
      h.remove()

  print("chose: %s" % best_p)
  method_variables["dropout_p"] = best_p

  names, hook_handles = [], []
  add_dropout_hooks(method_variables, model, start_name,
                    get_keywords(model, tack_et_al_pattern_keywords),
                    names, hook_handles)

  return model, method_variables


def dropout_metric(config, method_variables, model, imgs, targets):

  # run sample through n times and get variance in most probable class

  all = []
  for i in range(config.dropout_iterations):
    with torch.no_grad():
      preds = model(imgs)
    softmax_preds = F.softmax(preds, dim=1)
    preds_flat = preds.argmax(dim=1)
    all.append(softmax_preds[torch.arange(imgs.shape[0], device=device(config.cuda)), preds_flat])
    if i == 0:
      correct = preds_flat.eq(targets)

  all = torch.stack(all, dim=0)
  unreliability = all.var(dim=0)

  return unreliability, correct


def add_dropout_hooks(method_variables, model, curr_name, keywords, names, hook_handles, explore=True):
  for i, (name, m) in enumerate(model.named_children()):  # same order each time
    full_name = "%s_%s" % (curr_name, name)
    if explore: print("(add hooks) at %s, %s" % (
    full_name, m.__class__))  # use this to find the names to set pattern_keywords

    found = False
    for k_info in keywords:
      if fits_keyword(m, full_name, k_info):
        assert (not found)  # only one keyword allowed per module
        found = True
        names.append(full_name)
        m.recorded = None  # initialize storage field
        hook_handles.append(m.register_forward_hook(add_dropout_maker(method_variables["dropout_p"])))

    if hasattr(m, "named_children") and len(list(m.named_children())) > 0:
      add_dropout_hooks(method_variables, m, full_name, keywords, names, hook_handles)  # recurse


def add_dropout_maker(dropout_p):
  def add_dropout(module, input, output):
    as_tuple = isinstance(output, tuple)
    if as_tuple:
      assert (len(output) == 1)
      output = output[0]

    output = F.dropout(output, p=dropout_p, training=True, inplace=False)

    if as_tuple:
      output = (output,)

    return output

  return add_dropout


