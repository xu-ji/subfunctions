from util.methods.subfunctions import start_name, get_keywords, fits_keyword
from util.general import device
import torch
from util.data import classes_per_dataset
import numpy as np
from sys import stdout
from datetime import datetime

from sklearn.linear_model import LogisticRegressionCV
from sklearn import covariance

# unlike our method, the features don't need to be straight after a nonlinearity
class_distance_pattern_keywords = {
  "VGG": [("fieldname_exact", "root_features_%d" % i, "out") for i in [13, 23, 33, 43]], # for vgg16
  "ResNetModel": [("fieldname_exact", "root_conv5_x", "out"), ("fieldname_exact", "root_linear_feats", "out"), ("fieldname_exact", "root_fc", "out")] # for resnet50model
  # using earlier layers for resnet50 requires huge RAM for this method due to feature matrix inversion, >200GB... (they used smaller resnet34 in their code)
}


def class_distance_pre(config, model, train_loader, val_loader):
  # Get mean and cov per class and per layer - store
  method_variables = {}

  method_variables["num_classes"] = classes_per_dataset[config.data]

  # Add hooks to model
  assert config.model in ["vgg16_bn", "resnet50model"]
  names, hook_handles = [], []
  add_class_distance_hooks(model, start_name, get_keywords(model, class_distance_pattern_keywords),
                           names, hook_handles)
  print("(class_distance_pre) layer names: %s" % names)

  method_variables["pattern_layer_names"] = names
  method_variables["num_layers"] = len(names)

  # Compute mean
  means = [None for _ in range(method_variables["num_layers"])]  # layer -> class, feat len
  counts = torch.zeros(method_variables["num_classes"], dtype=torch.int,
                       device=device(config.cuda))  # class -> counts
  for batch_i, (imgs, targets) in enumerate(train_loader):
    if batch_i % max(1, len(train_loader) // 100) == 0:
      print("(class_distance_pre) training data means: %d / %d, %s" % (
      batch_i, len(train_loader), datetime.now()))
      stdout.flush()

    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
    with torch.no_grad():
      _ = model(imgs)

    curr_feats = pop_class_distance_feats(config, model, method_variables,
                                          class_distance_pattern_keywords)  # num layers: [num samples,
    # feature len]
    for l in range(method_variables["num_layers"]):
      if means[l] is None: means[l] = torch.zeros(
        (method_variables["num_classes"], curr_feats[l].shape[1]), dtype=torch.float,
        device=device(config.cuda))

      for c in range(method_variables["num_classes"]):
        feats_class = curr_feats[l][targets == c, :]
        assert (len(feats_class.shape) == 2)
        means[l][c] += feats_class.sum(dim=0)

    for c in range(method_variables["num_classes"]):
      counts[c] += (targets == c).sum()

  for l in range(method_variables["num_layers"]):
    means[l] /= counts.unsqueeze(1)

  # Compute (tied across classes) covariance. Like normal cov except subtract class-specific mean
  #covs = [None for _ in range(method_variables["num_layers"])]
  #layer_feats = [[] for _ in range(method_variables["num_layers"])]
  inv_covs = []

  for l in range(method_variables["num_layers"]): # making this outer loop for memory reasons
    total_count = 0
    layer_feats_l = []
    for batch_i, (imgs, targets) in enumerate(train_loader):
      if batch_i % max(1, len(train_loader) // 100) == 0:
        print("(class_distance_pre) l %d training data covs: %d / %d, %s" % (
          l, batch_i, len(train_loader), datetime.now()))
        stdout.flush()

      imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
      with torch.no_grad():
        _ = model(imgs)

        curr_feats = pop_class_distance_feats(config, model, method_variables,
                                              class_distance_pattern_keywords)  # num layers: [num samples,
        # feature len]
        if batch_i == 0:
          print("curr feats size: %s" % [curr_feats[l].shape for l in
                                         range(method_variables["num_layers"])])
          stdout.flush()

        for c in range(method_variables["num_classes"]):
          feats_class = curr_feats[l][targets == c, :]
          assert (len(feats_class.shape) == 2)
          curr_feat_demean = feats_class - means[l][c].unsqueeze(0)  # num samples, feat len

          layer_feats_l.append(curr_feat_demean.cpu())
          total_count += (targets == c).sum()

    assert total_count == len(train_loader.dataset)

    layer_feats_l = torch.cat(layer_feats_l, dim=0).cpu().numpy()
    print("(class_distance_pre) doing inv for layer %s" % l)
    stdout.flush()
    cov_solver = covariance.EmpiricalCovariance(store_precision=True, assume_centered=False)
    cov_solver.fit(layer_feats_l)
    inv_cov = torch.from_numpy(cov_solver.precision_).float().cuda()
    assert(inv_cov.shape == (layer_feats_l.shape[1], layer_feats_l.shape[1]))
    inv_covs.append(inv_cov)

  method_variables["means"] = means
  method_variables["inv_covs"] = inv_covs

  # for each eps:
  #   train logistic regressor (on all train data layer unreliabilities w/ eps) to predict true
  # result
  #   measure how well the regressor does (on all val data layer unreliabilities w/ eps)
  # save best eps and regressor model

  best_regressor_goodness = -np.inf
  best_regressor = None
  best_eps = None
  for eps in config.search_eps:
    regressor_goodness, regressor = eval_eps(config, eps, model, train_loader, val_loader,
                                             method_variables)

    print("(class_distance_pre) eps %s, regressor goodness %s, %s" % (
    eps, regressor_goodness, datetime.now()))
    stdout.flush()

    if regressor_goodness > best_regressor_goodness:
      best_regressor_goodness = regressor_goodness
      best_regressor = regressor
      best_eps = eps

  method_variables["eps"] = best_eps
  method_variables["regressor"] = best_regressor
  method_variables["regressor_goodness"] = best_regressor_goodness

  print("(class_distance_pre) best eps %s, best regressor goodness %s" % (
  best_eps, best_regressor_goodness))

  return model, method_variables


def class_distance_metric(config, method_variables, model, imgs, targets):
  # compute layer unreliabilities w/ eps, run through regressor, return (1 - regressor result)

  closeness = torch.zeros(imgs.shape[0], method_variables["num_layers"],
                          dtype=torch.float, device=device(config.cuda))
  corrects = torch.zeros(imgs.shape[0], dtype=torch.bool, device=device(config.cuda))

  get_class_dists(config, imgs, targets, method_variables["eps"], model, method_variables,
                  0, closeness, corrects)

  closeness = closeness.cpu().numpy()
  preds = method_variables["regressor"].predict_proba(closeness)
  assert (preds.shape == (imgs.shape[0], 2))
  assert (preds >= 0).all() and (preds <= 1).all()
  unreliability = preds[:, 0]
  unreliability = torch.tensor(unreliability, dtype=torch.float, device=device(config.cuda))

  return unreliability, corrects


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def add_class_distance_hooks(model, curr_name, keywords, names, hook_handles, explore=True):
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
        hook_handles.append(m.register_forward_hook(store_features_maker(k_info[2])))

    if hasattr(m, "named_children") and len(list(m.named_children())) > 0:
      add_class_distance_hooks(m, full_name, keywords, names, hook_handles)  # recurse


def store_features_maker(in_or_out):
  def store_features(module, input, output):
    features = {"in": input, "out": output}[in_or_out]

    if isinstance(features, tuple):
      assert (len(features) == 1)
      features = features[0]

    assert module.recorded is None
    module.recorded = features.flatten(start_dim=1)

  return store_features


def pop_class_distance_feats(config, model, method_variables, keywords, no_grad=True):
  # returns list of tensors. num layers: [num samples, feature len]
  maps = dict([(old_name, None) for old_name in method_variables["pattern_layer_names"]])
  layer_names = []
  get_distance_feats_helper(model, curr_name=start_name,
                            keywords=get_keywords(model, keywords),
                            names=layer_names,
                            maps=maps)
  for k, v in maps.items():
    assert v is not None

  assert (layer_names == method_variables["pattern_layer_names"])  # same order
  maps_list = [maps[layer_name] for layer_name in method_variables["pattern_layer_names"]]
  if no_grad:
    maps_list = [maps.detach() for maps in maps_list]

  return maps_list


def get_distance_feats_helper(model, curr_name, keywords, names, maps):
  # return maps as list of tensors. num layers: [num samples, feature len]

  # Rely on using the same iterative order every time for template indices to hold
  for i, (name, m) in enumerate(model.named_children()):  # ordered
    full_name = "%s_%s" % (curr_name, name)

    found = False
    for k_info in keywords:
      if fits_keyword(m, full_name, k_info):
        assert (not found)  # only one keyword allowed per module
        names.append(full_name)
        assert (isinstance(m.recorded, torch.Tensor))
        maps[full_name] = m.recorded
        found = True
        m.recorded = None  # clear model storage

    if hasattr(m, "named_children") and len(list(m.named_children())) > 0:
      get_distance_feats_helper(m, full_name, keywords, names, maps)  # recurse


def eval_eps(config, eps, model, train_loader, val_loader, method_variables):
  # for each sample x layer (train loader):
  #   use means and covs to get distance from sample to closest class (for that layer!)
  #   get gradient of closest distance^ wrt to feature, binarize it, * eps, add to feature
  #   recompute the closest class, then negate to get -ve of smallest distance

  train_closeness = torch.zeros(len(train_loader.dataset), method_variables["num_layers"],
                                dtype=torch.float, device=device(config.cuda))
  train_corrects = torch.zeros(len(train_loader.dataset), dtype=torch.bool,
                               device=device(config.cuda))
  batch_ind_start = 0
  for batch_i, (imgs, targets) in enumerate(train_loader):
    if batch_i % max(1, len(train_loader) // 50) == 0:
      print("(eval_eps) %s, collect train, batch %d / %d, %s " % (
      eps, batch_i, len(train_loader) - 1, datetime.now()))
      stdout.flush()

    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    get_class_dists(config, imgs, targets, eps, model, method_variables,
                    batch_ind_start, train_closeness, train_corrects)

    batch_ind_start += imgs.shape[0]
  assert (batch_ind_start == len(train_loader.dataset))
  assert (train_corrects == 0).sum() + (train_corrects == 1).sum() == train_corrects.shape[0]

  if config.balance_data:
    corrects_i = train_corrects.nonzero(as_tuple=False).squeeze(1)
    incorrects_i = (~train_corrects).nonzero(as_tuple=False).squeeze(1)
    print("(eval_eps) corrects sz %s, incorrects sz %s" % (corrects_i.shape, incorrects_i.shape))

    min_sz = min(corrects_i.shape[0], incorrects_i.shape[0])
    corrects_i, incorrects_i = corrects_i[:min_sz], incorrects_i[:min_sz]
    print(
      "(eval_eps) new corrects sz %s, incorrects sz %s" % (corrects_i.shape, incorrects_i.shape))

    assert (len(corrects_i.shape) == 1 and len(incorrects_i.shape) == 1)
    selected_inds = torch.cat((corrects_i, incorrects_i), dim=0)
    selected_inds = selected_inds[
      torch.tensor(np.random.permutation(selected_inds.shape[0]), device=device(config.cuda))]

    train_corrects = train_corrects[selected_inds]
    train_closeness = train_closeness[selected_inds]
    print("(eval_eps) final train corrects, closeness szs %s %s" % (
    train_corrects.shape, train_closeness.shape))

  # train: for each sample, tensor of -ves of closest distances -> sigmoid
  # loss: labels y are 0 or 1. avg of: -y * log(sigm) -(1 - y) log(1 - sigm)
  # because want to max log likelihood of individual probs = min their negation
  # y=1 indicates sample was correctly classified (confidence. Unreliability is 1 - this)

  train_closeness = train_closeness.cpu().numpy()
  train_corrects = train_corrects.cpu().numpy()
  regressor = LogisticRegressionCV(n_jobs=-1, multi_class="ovr", cv=5).fit(train_closeness,
                                                                           train_corrects)

  # do same distance tensor computation per sample in val loader, to obtain confidence
  # take > 0.5 to be confident, < 0.5 to be unconfident, and report accuracy (unlike our val,
  # which uses area under graph)
  val_closeness = torch.zeros(len(val_loader.dataset), method_variables["num_layers"],
                              dtype=torch.float, device=device(config.cuda))
  val_corrects = torch.zeros(len(val_loader.dataset), dtype=torch.bool, device=device(config.cuda))
  batch_ind_start = 0
  for batch_i, (imgs, targets) in enumerate(val_loader):
    if batch_i % max(1, len(val_loader) // 50) == 0:
      print("(eval_eps) %s, collect val, batch %d / %d, %s " % (
      eps, batch_i, len(val_loader) - 1, datetime.now()))
      stdout.flush()

    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    get_class_dists(config, imgs, targets, eps, model, method_variables,
                    batch_ind_start, val_closeness, val_corrects)

    batch_ind_start += imgs.shape[0]
  assert (batch_ind_start == len(val_loader.dataset))

  val_closeness = val_closeness.cpu().numpy()
  val_corrects = val_corrects.cpu().numpy()

  val_results = regressor.predict_proba(val_closeness)  # class 1 = p(accept)
  assert (val_results.shape == (val_closeness.shape[0], 2))
  assert (val_results >= 0).all() and (val_results <= 1).all()

  val_reliabilities = val_results[:, 1]
  accepts = val_reliabilities > 0.5
  # print("val proportion accepted %s out of %s, %s" % (accepts.sum(), val_corrects.shape[0],
  # val_reliabilities.shape[0]))
  # print("val proportion correct %s out of %s, %s" % (val_corrects.sum(), val_corrects.shape[0],
  #  val_reliabilities.shape[0]))
  regressor_acc = (accepts == val_corrects).sum() / float(
    val_corrects.shape[0])  # "wrong" = was correct, rejected or was incorrect, accepted

  return regressor_acc, regressor


def get_class_dists(config, inputs, targets, eps, model, method_variables,
                    batch_ind_start, closeness_out, corrects_out):
  # closeness_out shape: num samples, num layers
  for l in range(method_variables["num_layers"]):
    imgs_l = inputs.clone().requires_grad_(True)
    preds = model(imgs_l)

    if l == 0:  # take opportunity to store whether it was correctly predicted
      corrects_out[batch_ind_start:(batch_ind_start + inputs.shape[0])] = (
      preds.argmax(dim=1).eq(targets))

    curr_feats = pop_class_distance_feats(config, model, method_variables,
                                          class_distance_pattern_keywords,
                                          no_grad=False)  # num layers: [num samples, feature len]

    dists_l = get_feats_class_dists_l(config, curr_feats, l,
                                      method_variables)  # num samples, num classes
    assert (dists_l.requires_grad)

    closest_class_dists, closest_class = dists_l.min(dim=1)  # no grad
    imgs_grads = torch.autograd.grad(dists_l[torch.arange(dists_l.shape[0], device=device(
      config.cuda)), closest_class].sum(), imgs_l)[0]
    assert imgs_grads.shape == imgs_l.shape

    # have to produce a new set of images per layer - hence this expensive loop
    imgs_l = imgs_l.detach() - eps * imgs_grads  # min dist to closest class with noise eps

    with torch.no_grad():
      _ = model(imgs_l)

    curr_feats_final = pop_class_distance_feats(config, model, method_variables,
                                                class_distance_pattern_keywords)  # num layers: [num samples,
    # feature len]
    dists_l_final = get_feats_class_dists_l(config, curr_feats_final, l,
                                            method_variables)  # num samples, num classes

    closest_class_dists_final, _ = dists_l_final.min(dim=1)
    assert (closest_class_dists_final.shape == (inputs.shape[0],))

    closeness_out[batch_ind_start:(batch_ind_start + inputs.shape[0]),
    l] = - closest_class_dists_final


def get_feats_class_dists_l(config, curr_feats, l, method_variables):
  feats_l = curr_feats[l].unsqueeze(1)  # num samples, 1, feat len
  feats_l_x_classes = feats_l - method_variables["means"][l].unsqueeze(
    0)  # 1, num classes, feat len -> num samples, num classes, feat len
  assert feats_l_x_classes.shape == (
  curr_feats[l].shape[0], method_variables["num_classes"], curr_feats[l].shape[1])

  # for each sample, for each class, get distance
  inv_cov_l = method_variables["inv_covs"][l].unsqueeze(0)  # 1, feat len, feat len
  assert (inv_cov_l.shape == (1, curr_feats[l].shape[1], curr_feats[l].shape[1]))

  dists_part = torch.matmul(feats_l_x_classes, inv_cov_l)
  assert (dists_part.shape == (
  curr_feats[l].shape[0], method_variables["num_classes"], curr_feats[l].shape[1]))
  dists = (feats_l_x_classes * dists_part).sum(dim=2)  # dot product
  assert (dists.shape == (curr_feats[l].shape[0], method_variables["num_classes"]))

  return dists
