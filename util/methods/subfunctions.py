from util.general import *
from sys import stdout
from scipy.special import comb
from datetime import datetime
import sys

# We only consider post relu layers (not e.g. post maxpool layers, though this is also doable)
pattern_keywords = {
  "MLP": [("fieldname", "relu", "out")],
  "VGG": [("fieldname_exact", "root_classifier_4", "out")],
  "ResNetModel": [("fieldname_exact", "root_linear_feats", "out")],
}

start_name = "root"


def subfunctions_pre(config, model, train_loader, val_loader):
  """
  Two usages of this function depending on config.precompute: 
  to precompute normalization terms (for given kernel width p), 
  or to load precomputed terms for best p
  """

  config.search_deltas = torch.tensor(config.search_deltas)
  print("(subfunctions_pre) search deltas: %s, search ps: %s" % (
  config.search_deltas, config.search_ps))

  names, hook_handles = [], []
  add_hooks_tensor(model, start_name, get_keywords(model), names, hook_handles)

  if config.precompute:
    print("(subfunctions_pre) precomputing p_i %d" % config.precompute_p_i)
    # Do precompute_p_i th value of p
    method_variables = {}

    train_ind_to_patt = None
    train_patt_to_ind = {}

    mask = None
    print("(subfunctions_pre) layer names: %s" % names)
    method_variables["pattern_layer_names"] = names
    method_variables["subsample_inds"] = None

    # Run training data through and get counts and accs
    num_train_samples = len(train_loader.dataset)
    method_variables["m"] = num_train_samples
    train_corrects = torch.zeros(num_train_samples, dtype=torch.int32).to(device(config.cuda))
    train_sample_counts = torch.zeros(num_train_samples, dtype=torch.int32).to(device(config.cuda))
    train_indexer_i = 0
    for batch_i, (imgs, targets) in enumerate(train_loader):
      if batch_i % max(1, len(train_loader) // 100) == 0:
        print("(subfunctions_pre) training data: %d / %d" % (batch_i, len(train_loader)))
        stdout.flush()

      imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
      with torch.no_grad():
        preds = model(imgs)
      preds_flat = preds.argmax(dim=1)
      correct = preds_flat.eq(targets)

      if mask is None:
        mask = get_patt_mask_tensor(model, names)
        if train_ind_to_patt is None:
          train_ind_to_patt = torch.zeros(num_train_samples, mask.shape[0], dtype=torch.bool).to(
            device(config.cuda))

      curr_patterns = pop_maps_tensor(config, model, names)
      curr_patterns = format_patterns(curr_patterns, method_variables["subsample_inds"])
      for j in range(imgs.shape[0]):
        patt_str = bool_tensor_content_hash(curr_patterns[j])
        if not patt_str in train_patt_to_ind:
          train_ind_to_patt[train_indexer_i, :] = curr_patterns[j]
          train_patt_to_ind[patt_str] = train_indexer_i
          train_indexer_i += 1

        train_corrects[train_patt_to_ind[patt_str]] += correct[j].item()
        train_sample_counts[train_patt_to_ind[patt_str]] += 1

    train_corrects = train_corrects[:train_indexer_i]
    train_sample_counts = train_sample_counts[:train_indexer_i]
    train_ind_to_patt = train_ind_to_patt[:train_indexer_i, :]

    method_variables["train_ind_to_patt"] = train_ind_to_patt
    method_variables["train_patt_to_ind"] = train_patt_to_ind
    method_variables["corrects"] = train_corrects
    method_variables["sample_counts"] = train_sample_counts
    print("(subfunctions_pre) num populated polytopes: %s" % train_ind_to_patt.shape[0])

    N = mask.sum().item()
    print("(subfunctions_pre) N: %s" % N)  # number of nodes = size of "V" in the paper
    stdout.flush()
    method_variables["N"] = N
    method_variables["mask"] = mask
    method_variables["search_deltas"] = config.search_deltas
    method_variables["best_metric"] = -np.inf

    if config.pattern_batch_sz > 0:
      pattern_batch_sz = config.pattern_batch_sz
    else:
      pattern_batch_sz = train_ind_to_patt.shape[0]  # all
    method_variables["pattern_batch_sz"] = pattern_batch_sz

    # Compute terms for our p
    p = config.search_ps[config.precompute_p_i]
    print("(subfunctions_pre) doing p %d (%s), %s" % (config.precompute_p_i, p, datetime.now()))
    stdout.flush()

    global_prob_norm = compute_prob_norm(num_train_samples, N, p, config.dist_fn)
    if not np.isfinite(global_prob_norm) or global_prob_norm > sys.maxsize:
      print("(subfunctions_pre) skipping %s (global_prob_norm: %s)" % (p, global_prob_norm))
      method_variables["best_metric"] = -np.inf

    else:
      print("(subfunctions_pre) p %s global_prob_norm: %s" % (p, global_prob_norm))

      method_variables["global_prob_norm"] = global_prob_norm
      method_variables["p"] = p

      train_patt_factors = compute_norm_terms(config, train_ind_to_patt, N, global_prob_norm,
                                              train_sample_counts, p,
                                              pattern_batch_sz)  # num_train_samples array of floats
      method_variables["train_patt_factors"] = train_patt_factors

      best_delta, metric_per_delta, isfinite = \
        find_best_delta_tensor(config, val_loader, model, method_variables, names, STEB)

      if not isfinite:
        method_variables["best_metric"] = -np.inf
      else:
        curr_metric = metric_per_delta.max()
        print(
          "(subfunctions_pre) currently considering p %s, its best delta %s and metric %s" % (
          p, best_delta, curr_metric))
        stdout.flush()

        method_variables["best_metric"] = curr_metric
        method_variables["delta"] = best_delta
        method_variables["delta_results"] = dict(
          [(config.search_deltas[i], metric_per_delta[i]) for i in
           range(config.search_deltas.shape[0])])

    torch.save(method_variables,
               osp.join(config.models_root, "%s.pytorch" % cleanstr(
                 "%s_%s_%s_%s_%s_method_variables_p_%d" % (
                 config.data, config.seed, config.method, config.model,
                 config.suff, config.precompute_p_i))))
    print("(subfunctions_pre) finished computations for this p")

    if config.test_code_brute_force: test_subfunctions(config, method_variables)
    exit(0)

  else:
    print("(subfunctions_pre) selecting best p")
    metric_values = np.zeros(len(config.search_ps), dtype=np.float)
    for p_i in range(len(config.search_ps)):
      saved_method_variables = torch.load(osp.join(config.models_root, "%s.pytorch" % cleanstr(
        "%s_%s_%s_%s_%s_method_variables_p_%d" % (
        config.data, config.seed, config.method, config.model, config.suff,
        p_i))))
      metric_values[p_i] = saved_method_variables["best_metric"]
    best_p_i = metric_values.argmax()

    if not np.isfinite(metric_values.max()):
      print("(subfunctions_pre) No hyperparameter was successful. Aborting.")
      exit(0)

    print("(subfunctions_pre) best i %s, all %s" % (best_p_i, list(metric_values)))

    method_variables = torch.load(osp.join(config.models_root, "%s.pytorch" % cleanstr(
      "%s_%s_%s_%s_%s_method_variables_p_%d" % (
        config.data, config.seed, config.method, config.model, config.suff,
        best_p_i))))

    method_variables["p_i"] = best_p_i
    print(
      "(subfunctions_pre) p %s (ind %s), best delta %s, all %s, global prob norm %s, area %s" % (
      method_variables["p"],
      method_variables["p_i"],
      method_variables["delta"],
      method_variables["delta_results"],
      method_variables["global_prob_norm"],
      method_variables["best_metric"]))
    stdout.flush()
    return model, method_variables


def subfunctions_metric(config, method_variables, model, imgs, targets, get_polytope_ids=False):
  # Inference function for unreliability

  with torch.no_grad():
    preds = model(imgs)
  preds_flat = preds.argmax(dim=1)
  correct = preds_flat.eq(targets)

  curr_patterns = pop_maps_tensor(config, model, method_variables["pattern_layer_names"])
  unreliability = STEB(config, curr_patterns, method_variables,
                       torch.tensor([method_variables["delta"]],
                                    device=device(config.cuda))).squeeze(1)

  if not get_polytope_ids:
    return unreliability, correct
  else:
    return unreliability, correct, curr_patterns


# --------------------------------------------------------------------------------------------------
# Major helpers
# --------------------------------------------------------------------------------------------------


def STEB(config, query_patterns, method_variables, deltas):
  assert isinstance(query_patterns, torch.Tensor)
  assert isinstance(deltas, torch.Tensor) and len(deltas.shape) == 1

  query_patterns = format_patterns(query_patterns, method_variables["subsample_inds"])

  closeness = compute_closeness(config, query_patterns,
                                method_variables)  # num queries x num training samples
  res = compute_emprisk(closeness, method_variables).unsqueeze(1).expand(query_patterns.shape[0],
                                                                         deltas.shape[
                                                                           0])  # num queries,
  # num deltas

  if not config.no_bound:
    smooth_prob = (closeness * method_variables["sample_counts"].unsqueeze(0)).sum(dim=1) / \
                  method_variables["global_prob_norm"]  # num queries
    gen_gap_bound = torch.sqrt(torch.log(2. / deltas.unsqueeze(0)) / (
    2. * float(method_variables["m"]))) / smooth_prob.unsqueeze(1)  # num queries, num deltas

    res = res.clone() + gen_gap_bound
    if not config.no_log:
      res = torch.log(res)

  return res


def compute_closeness(config, query_patterns, method_variables):
  num_shared_bits = []
  assert method_variables["pattern_batch_sz"] > 0
  num_pop_polytopes = method_variables["train_ind_to_patt"].shape[0]
  num_batches = int(np.ceil(num_pop_polytopes / float(method_variables["pattern_batch_sz"])))

  for batch_l in range(num_batches):
    start_l, end_l = batch_l * method_variables["pattern_batch_sz"], min(num_pop_polytopes,
                                                                         (batch_l + 1) *
                                                                         method_variables[
                                                                           "pattern_batch_sz"])
    patterns = method_variables["train_ind_to_patt"][start_l: end_l,
               :]  # batch sz or less, patt len
    num_shared_bits.append(
      (query_patterns.unsqueeze(1) == patterns.unsqueeze(0)).sum(dim=2))  # num queries, batch size
  num_shared_bits = torch.cat(num_shared_bits, dim=1)

  assert num_shared_bits.shape == (query_patterns.shape[0], num_pop_polytopes)
  closeness = close_par(config, num_shared_bits.view(-1), method_variables["N"],
                        method_variables["p"])
  closeness = closeness.view(query_patterns.shape[0],
                             num_pop_polytopes)  # num queries, num train samples
  return closeness


def compute_emprisk(closeness, method_variables):
  # closeness: num queries, num train samples
  emprisk = ((closeness * (
  method_variables["sample_counts"] - method_variables["corrects"]).unsqueeze(0)) /
             method_variables["train_patt_factors"].unsqueeze(0)).sum(dim=1) / float(
    method_variables["m"])
  return emprisk  # num queries


def compute_prob_norm(num_train_samples, N, p, dist_fn):
  if dist_fn == "geometric":
    prev_term = p
  elif dist_fn == "gaussian":
    prev_term = 1.

  total = prev_term
  for i in range(1, N + 1):
    if dist_fn == "geometric":
      new_term = ((1. - p) * (N - i + 1) / float(
        i)) * prev_term  # doesn't use cumulative sum but prev term!
    elif dist_fn == "gaussian":
      new_term = (np.exp((-2. * i + 1) / 2. * (p ** 2.)) * (N - i + 1) / float(i)) * prev_term

    if new_term < sys.float_info.epsilon: break

    total += new_term
    prev_term = new_term

  prob_norm = total * num_train_samples

  return prob_norm


def find_best_delta_tensor(config, val_loader, model, method_variables, names, metric_function_par):
  # Find the best delta in parallel

  unreliability_per_delta = []
  val_corrects = []
  for batch_i, (imgs, targets) in enumerate(val_loader):
    if batch_i % max(1, len(val_loader) // 100) == 0:
      print("(find_best_delta_tensor) val data: %d / %d" % (batch_i, len(val_loader)))
      stdout.flush()

    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
    with torch.no_grad():
      preds = model(imgs)
    preds_flat = preds.argmax(dim=1)
    correct = preds_flat.eq(targets)

    curr_patterns = pop_maps_tensor(config, model, names)
    unreliability_per_delta.append(metric_function_par(config, curr_patterns, method_variables,
                                                       config.search_deltas.to(device(
                                                         config.cuda))))  # num patterns, num deltas
    val_corrects.append(correct)

  unreliability_per_delta = torch.cat(unreliability_per_delta, dim=0)
  val_corrects = torch.cat(val_corrects)

  assert (unreliability_per_delta.shape == (val_corrects.shape[0], config.search_deltas.shape[0]))

  if not unreliability_per_delta.isfinite().any():
    print("(find_best_delta_tensor) all densities are 0 for this arch x dataset. Not picking this.")
    best_delta = -1
    metric_per_delta = None
    isfinite = False
  else:
    print(
      "(find_best_delta_tensor) val data risk bounds info (of finite): mean, std, min, max, "
      "proportion finite:")
    for i in range(config.search_deltas.shape[0]):
      unreliability_finite = unreliability_per_delta[unreliability_per_delta[:, i].isfinite(), i]
      if unreliability_finite.shape[0] > 0:
        print((unreliability_finite.mean(dim=0).item(), unreliability_finite.std(dim=0).item(),
               unreliability_finite.min(dim=0)[0].item(), unreliability_finite.max(dim=0)[0].item(),
               unreliability_finite.shape[0], unreliability_per_delta.shape[0]))
      else:
        print("(find_best_delta_tensor) Empty")

    metric_per_delta, rejected = \
      compute_metric_per_delta(config, unreliability_per_delta, val_corrects)
    best_delta_i = np.argmax(metric_per_delta)
    assert not (rejected[best_delta_i])
    best_delta = config.search_deltas[best_delta_i]
    isfinite = True

  return best_delta, metric_per_delta, isfinite


def compute_norm_terms(config, train_ind_to_patt, N, global_prob_norm, train_sample_counts, p,
                       pattern_batch_sz):
  """
  This function computes normalization terms for every populated subfunction.
  Batching is used for tractability.
  """

  # Computing the lookup table
  max_nonnegligible_dist = find_biggest_non_negligible_dist(config, N, p)
  print("(compute_norm_terms) max_nonnegligible_dist: %s" % max_nonnegligible_dist)

  # work out inner sum for each [0...N] - in batches
  num_shared_bits_all = torch.arange(N + 1, dtype=torch.int, device=device(config.cuda))

  all_combs = compute_all_combs(N, max_nonnegligible_dist, config.cuda)  # N + 1, nonneg

  factors_parts_all = []
  num_factor_parts_batches = int(np.ceil(num_shared_bits_all.shape[0] / float(pattern_batch_sz)))
  for shared_bits_batch_i in range(num_factor_parts_batches):
    print("(compute_norm_terms) computing factor parts all %d / %d, %s" % (
    shared_bits_batch_i, num_factor_parts_batches - 1, datetime.now()))
    stdout.flush()

    start = shared_bits_batch_i * pattern_batch_sz
    end = min((shared_bits_batch_i + 1) * pattern_batch_sz, num_shared_bits_all.shape[0])
    factors_parts = compute_factor_part_par(config, num_shared_bits_all[start:end], N, p,
                                            max_nonnegligible_dist, all_combs)
    factors_parts_all.append(factors_parts)
  factors_parts_all = torch.cat(factors_parts_all, dim=0)
  assert factors_parts_all.shape == (N + 1,)

  # Using the lookup table
  num_samples = train_ind_to_patt.shape[0]
  num_batches = int(np.ceil(num_samples / float(pattern_batch_sz)))
  factors = torch.zeros(num_samples, dtype=torch.float, device=device(config.cuda))
  for batch_k in range(num_batches):
    print("(compute_norm_terms) %d (%d) out of %d, %s" % (
    batch_k, batch_k * pattern_batch_sz, num_batches - 1, datetime.now()))
    sys.stdout.flush()
    start_k, end_k = batch_k * pattern_batch_sz, min(num_samples, (batch_k + 1) * pattern_batch_sz)
    h_k = train_ind_to_patt[start_k: end_k, :]  # batch sz or less, patt len

    for batch_l in range(num_batches):
      start_l, end_l = batch_l * pattern_batch_sz, min(num_samples,
                                                       (batch_l + 1) * pattern_batch_sz)
      h_l = train_ind_to_patt[start_l: end_l, :]  # batch sz or less, patt len
      h_l_count = train_sample_counts[start_l: end_l]

      # eq not mult
      num_shared_bits = (h_k.unsqueeze(1) == h_l.unsqueeze(0)).sum(dim=2).to(
        torch.long)  # batch_sz (k), batch_sz (l). ints
      assert (num_shared_bits.shape == (end_k - start_k, end_l - start_l))

      factors_k_l = factors_parts_all[num_shared_bits.view(-1)].view(end_k - start_k,
                                                                     end_l - start_l)
      factors_k_l = (h_l_count.unsqueeze(0) * factors_k_l).sum(dim=1)
      factors[start_k: end_k] += factors_k_l

  factors /= global_prob_norm

  return factors


def compute_all_combs(N, max_nonnegligible_dist, is_cuda):
  # Compute combinations metrics in parallel
  # Returns size N + 1, max_nonnegligible_dist+1 matrix

  curr = torch.ones(N + 1, max_nonnegligible_dist + 1, dtype=torch.int64,
                    device=device(is_cuda))  # b = 0 (fix first column)
  a_mat = torch.arange(N + 1, device=device(is_cuda)).unsqueeze(1).expand(-1,
                                                                          max_nonnegligible_dist
                                                                          + 1)  # corresponds to row
  b_mat = torch.arange(max_nonnegligible_dist + 1, device=device(is_cuda)).unsqueeze(0).expand(
    N + 1, -1)  # corresponds to col
  assert (a_mat.shape == curr.shape) and (b_mat.shape == curr.shape)
  incr_numer = a_mat - b_mat + 1
  incr_denom = b_mat
  for b in range(1, max_nonnegligible_dist + 1):  # 1... max_nonnegligible_dist
    curr[:, b] = (curr[:, b - 1] * incr_numer[:, b]) // incr_denom[:, b]
  curr[b_mat > a_mat] = 0  # upper right triangle is 0, choosing more than available
  return curr


def compute_factor_part_par(config, num_shared_bits, N, p, max_nonnegligible_dist, all_combs):
  assert len(num_shared_bits.shape) == 1

  # first axis is i, second is j
  i_mat = torch.arange(max_nonnegligible_dist + 1, device=device(config.cuda)).unsqueeze(1).expand(
    -1, max_nonnegligible_dist + 1)  # b
  j_mat = i_mat.transpose(0, 1)  # c

  close_mat1 = close_par(config, (N - i_mat).unsqueeze(0), N, p).expand(num_shared_bits.shape[0],
                                                                        -1,
                                                                        -1)  # sample len,
  # nonneg, nonneg (same for all samples)
  close_mat2 = close_par(config, num_shared_bits.unsqueeze(1).unsqueeze(1) + (
  j_mat - (i_mat - j_mat)).unsqueeze(0), N, p)  # sample len, nonneg, nonneg

  # num shared bits is contiguous ascending order
  comb_mat1 = comb_par_efficient((N - num_shared_bits)[-1], (N - num_shared_bits)[0], j_mat,
                                 all_combs).flip(0)  # sample len, nonneg+1, nonneg+1
  comb_mat2 = comb_par_efficient(num_shared_bits[0], num_shared_bits[-1], i_mat - j_mat,
                                 all_combs)  # sample len, nonneg+1, nonneg+1

  comb_mat1[(close_mat1 == 0.)] = 0.  # 0. times nan is still nan, remove
  comb_mat1[(close_mat2 == 0.)] = 0.

  comb_mat2[(close_mat1 == 0.)] = 0.
  comb_mat2[(close_mat2 == 0.)] = 0.

  for mat in [close_mat1, close_mat2, comb_mat1, comb_mat2]:
    assert mat.shape == (
    num_shared_bits.shape[0], max_nonnegligible_dist + 1, max_nonnegligible_dist + 1)

  res = close_mat1 * comb_mat1 * comb_mat2 * close_mat2

  # only take the elements that matter
  res[:, j_mat > i_mat] = 0.
  res[(j_mat.unsqueeze(0) > (N - num_shared_bits).unsqueeze(1).unsqueeze(1))] = 0.
  res[(i_mat - j_mat).unsqueeze(0) > num_shared_bits.unsqueeze(1).unsqueeze(1)] = 0.

  assert res.shape == (
  num_shared_bits.shape[0], max_nonnegligible_dist + 1, max_nonnegligible_dist + 1)

  res = res.sum(dim=[1, 2])
  assert res.isfinite().all()

  return res


def close_par(config, num_shared_material_bits, N, p):  # num_shared_material_bits is mat
  # either geometric or exp
  assert ((num_shared_material_bits >= 0).logical_or(num_shared_material_bits <= N)).all()
  # if config.dist_fn == "geometric":
  #  res = p * torch.pow(1. - p, N - num_shared_material_bits)
  if config.dist_fn == "gaussian":
    res = torch.exp(- (((N - num_shared_material_bits) ** 2) / (2 * (p ** 2))))
  else:
    raise NotImplementedError

  return res


def comb_par_efficient(n_start, n_end_incl, i_mat, all_combs):
  assert n_start <= n_end_incl
  assert len(i_mat.shape) == 2

  res = all_combs[n_start:(n_end_incl + 1), i_mat.flatten()].view(
    (n_end_incl - n_start + 1,) + i_mat.shape)
  return res


def comb_par(n, i_mat):
  # Another version of comb_par_efficient
  assert len(n.shape) == 1 and len(i_mat.shape) == 2

  curr = torch.ones((n.shape[0],) + i_mat.shape, dtype=torch.int64,
                    device=device(i_mat.is_cuda))  # for j=0.
  for j in range(1, i_mat.max() + 1):  # scalar j
    # advance those to j that need it
    # curr[:, j <= i_mat] is 2D (collapse last 2 axis to 1). 2nd axis is variable length.
    curr[:, j <= i_mat] = curr[:, j <= i_mat] * (n - j + 1).unsqueeze(
      1) // j  # no remainder. Szs: num samples, * and num samples, 1

  # where i > n, set to 0 (n choose i = 0)
  curr[n.unsqueeze(1).unsqueeze(1) < i_mat.unsqueeze(0)] = 0
  return curr


def find_biggest_non_negligible_dist(config, N, p):
  # only need to consider subset of [0, N] for i, because beyond a threshold the closeness metric
  # becomes negligible
  for d in range(N + 1):
    metric = close_par(config, torch.tensor([N - d]), N, p)
    stop = metric <= sys.float_info.epsilon

    if stop:
      assert d >= 1
      return d - 1
  return N


# --------------------------------------------------------------------------------------------------
# Minor helpers
# --------------------------------------------------------------------------------------------------


def bool_tensor_content_hash(curr_pattern):
  curr_pattern = curr_pattern.cpu().numpy()
  return "".join([str(x) for x in curr_pattern])


def format_patterns(query_patterns, subsample_inds):
  if not subsample_inds is None:
    assert len(subsample_inds.shape) == 1 and len(query_patterns.shape) == 2
    res = query_patterns[:, subsample_inds]
    assert res.shape == (query_patterns.shape[0], subsample_inds.shape[0])
    return res
  else:
    return query_patterns


def get_patt_mask_tensor(model, old_names):
  # returns uint8 tensor
  module_masks = []
  layer_names = []
  get_mask_tensor_helper(model, curr_name=start_name, keywords=get_keywords(model),
                         names=layer_names,
                         module_masks=module_masks)
  assert (layer_names == old_names)  # same order
  mask = torch.cat(module_masks, dim=0)
  assert len(mask.shape) == 1
  return mask


def get_mask_tensor_helper(model, curr_name, keywords, names, module_masks):
  # Rely on using the same iterative order every time for template indices to hold
  for i, (name, m) in enumerate(model.named_children()):  # ordered
    full_name = "%s_%s" % (curr_name, name)

    found = False
    for k_info in keywords:
      if fits_keyword(m, full_name, k_info):
        assert (not found)  # only one keyword allowed per module
        names.append(full_name)
        # print((m.mask, m.mask.__class__))
        assert (isinstance(m.mask, torch.Tensor))
        module_masks.append(m.mask)
        found = True

    if hasattr(m, "named_children") and len(list(m.named_children())) > 0:
      get_mask_tensor_helper(m, full_name, keywords, names, module_masks)  # recurse


def add_hooks_tensor(model, curr_name, keywords, names, hook_handles, explore=True):
  for i, (name, m) in enumerate(model.named_children()):  # same order each time
    full_name = "%s_%s" % (curr_name, name)
    if explore: print("(add_hooks_tensor) at %s, %s" % (
    full_name, m.__class__))  # use this to find the names to set pattern_keywords

    # everything gets tested for keyword, regardless of if it has children or not
    found = False
    for k_info in keywords:
      if fits_keyword(m, full_name, k_info):
        assert (not found)  # only one keyword allowed per module
        found = True
        names.append(full_name)
        m.recorded = None  # initialize storage field
        m.mask = None
        hook_handles.append(m.register_forward_hook(store_tensor_pattern_fn_maker(k_info[2])))

    if hasattr(m, "named_children") and len(list(m.named_children())) > 0:
      add_hooks_tensor(m, full_name, keywords, names, hook_handles)  # recurse


def store_tensor_pattern_fn_maker(in_or_out):
  def store_pattern_fn(module, input, output):
    features = {"in": input, "out": output}[in_or_out]

    if isinstance(features, tuple):
      assert (len(features) == 1)
      features = features[0]

    assert (features >= 0).all()  # post relu

    discrete = features > 0.
    discrete = discrete.detach().flatten(
      start_dim=1)  # bools. Flatten from channels. Size: nun samples, feature len

    # list of uint8 arrays, one per sample
    assert module.recorded is None
    module.recorded = discrete

    if module.mask is None:
      # this is a bit outdated, don't really need to store a mask now we're storing full bit
      # pattern as tensor
      material_bits = torch.ones(discrete[0].shape, dtype=torch.bool,
                                 device=device(discrete.is_cuda))  # Size: feature len
      module.mask = material_bits

  return store_pattern_fn


def pop_maps_tensor(config, model, old_names):
  # Return num samples, pattern len
  # Also clears map storage
  maps = []  # ordered list of sz: num samples, layer pattern len
  layer_names = []
  get_maps_store_tensor(model, curr_name=start_name, keywords=get_keywords(model),
                        names=layer_names, maps=maps)

  assert (layer_names == old_names)  # same order
  maps_final = torch.cat(maps, dim=1)
  assert len(maps_final.shape) == 2 and maps_final.shape[0] == maps[0].shape[0]
  return maps_final


def get_maps_store_tensor(model, curr_name, keywords, names, maps):
  # Rely on using the same iterative order every time for template indices to hold
  for i, (name, m) in enumerate(model.named_children()):  # ordered
    full_name = "%s_%s" % (curr_name, name)

    found = False
    for k_info in keywords:
      if fits_keyword(m, full_name, k_info):
        assert (not found)  # only one keyword allowed per module
        names.append(full_name)
        assert (isinstance(m.recorded, torch.Tensor) and len(m.recorded.shape) == 2)
        maps.append(m.recorded)
        found = True
        m.recorded = None  # clear model storage

    if hasattr(m, "named_children") and len(list(m.named_children())) > 0:
      get_maps_store_tensor(m, full_name, keywords, names, maps)  # recurse


def get_keywords(model, keywords=pattern_keywords):
  return keywords[model.__class__.__name__.split(".")[-1]]


def fits_keyword(m, full_name, k_info):
  mode, name, in_or_out = k_info
  if mode == "classname":
    return name in m.__class__.__name__  # .lower()
  elif mode == "fieldname":
    return name in full_name
  elif mode == "fieldname_exact":
    return name == full_name


def compute_metric_per_delta(config, unreliability_per_param, corrects):
  """
  Metric for choosing the best kernel width p (AUCEA). Mirrors evaluation in main script.
  """
  assert (len(unreliability_per_param.shape) == 2 and
          len(corrects.shape) == 1 and unreliability_per_param.shape[0] == corrects.shape[0])
  num_search_params = unreliability_per_param.shape[1]
  min_unrel = torch.zeros(1, num_search_params, dtype=torch.float, device=device(config.cuda))
  max_unrel = torch.zeros(1, num_search_params, dtype=torch.float, device=device(config.cuda))
  reject_param = torch.zeros(num_search_params, dtype=torch.bool, device=device(config.cuda))

  for i in range(num_search_params):
    unreliability_i_finite = unreliability_per_param[unreliability_per_param[:, i].isfinite(), i]
    if unreliability_i_finite.shape[0] == 0:
      reject_param[i] == True  # all bounds are infinite
    else:
      min_unrel[0, i] = unreliability_i_finite.min()
      max_unrel[0, i] = unreliability_i_finite.max()

  # loop through min and max finite bounds, but also consider infinity
  tprs, fprs = np.zeros((config.threshold_divs + 1 + 2, num_search_params), dtype=np.float), \
               np.zeros((config.threshold_divs + 1 + 2, num_search_params), dtype=np.float)
  coverages = np.zeros((config.threshold_divs + 1 + 2, num_search_params), dtype=np.float)
  acc_or_precisions = np.zeros((config.threshold_divs + 1 + 2, num_search_params), dtype=np.float)

  corrects = corrects.unsqueeze(1)  # num samples, 1
  for ts_i in range(config.threshold_divs + 1 + 2):
    if ts_i == 0:
      ts = torch.ones(1, num_search_params, dtype=torch.float, device=device(config.cuda)) * -np.inf
    elif ts_i <= config.threshold_divs + 1:
      ts = min_unrel + ((ts_i - 1) / float(config.threshold_divs)) * (
      max_unrel - min_unrel)  # 1, num params
    else:
      assert ts_i == config.threshold_divs + 2
      ts = torch.ones(1, num_search_params, dtype=torch.float, device=device(config.cuda)) * np.inf

    accepts = unreliability_per_param <= ts  # num samples, num params

    # all size num params
    tp = (accepts * corrects).sum(dim=0)
    fp = (accepts * (~corrects)).sum(dim=0)
    tn = ((~accepts) * (~corrects)).sum(dim=0)
    fn = ((~accepts) * corrects).sum(dim=0)

    tpr = tp / (tp + fn).to(torch.float)
    fpr = fp / (fp + tn).to(torch.float)
    tpr[~tpr.isfinite()] = 0.
    fpr[~fpr.isfinite()] = 0.

    coverage = (tp + fp) / (tp + fp + tn + fn).to(torch.float)  # how many we accepted
    acc_or_precision = tp / (tp + fp).to(torch.float)  # how many were correct within the accepted
    acc_or_precision[~acc_or_precision.isfinite()] = 0.

    coverages[ts_i, :] = coverage.cpu().numpy()

    acc_or_precisions[ts_i, :] = acc_or_precision.cpu().numpy()

    tprs[ts_i, :] = tpr.cpu().numpy()
    fprs[ts_i, :] = fpr.cpu().numpy()

  results = np.zeros(num_search_params, dtype=np.float)
  for i in range(num_search_params):
    if not reject_param[i]:
      if not config.select_on_AUROC:
        results[i] = compute_area(xs=coverages[:, i], ys=acc_or_precisions[:, i], mode="under")
      else:
        results[i] = compute_area(xs=fprs[:, i], ys=tprs[:, i], mode="under")
    else:
      results[i] = - np.inf

  return results, reject_param


# --------------------------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------------------------


def test_subfunctions(config, method_variables, batch_sz=10000):
  # Warning: brute force. Use for small N / tiny networks only.

  print("num samples all: %s, %s" % (
  method_variables["sample_counts"].sum().item(), method_variables["m"]))
  print("CHECK ORIG: prob norm: %s" % method_variables["global_prob_norm"])
  # multiply by number of populated polytopes
  assert (method_variables["sample_counts"] > 0).all()

  print("CHECK ORIG: closeness mass: %s" % (
  method_variables["train_ind_to_patt"].shape[0] * method_variables["global_prob_norm"] /
  method_variables["m"]))

  train_acc = method_variables["corrects"].sum().item() / float(
    method_variables["sample_counts"].sum().item())

  print(
    "sample counts all %s, corrects %s, acc %s" % (method_variables["sample_counts"].sum().item(),
                                                   method_variables["corrects"].sum().item(),
                                                   train_acc))
  print("CHECK ORIG: training error: %s" % (1. - train_acc))

  print("CHECK ORIG: example normalisation term: %s" % method_variables["train_patt_factors"][0])

  print("CHECK ORIG: total prob: 1.0 (obviously)")

  # check all patterns unique
  for i in range(method_variables["train_ind_to_patt"].shape[0]):
    matches = (
    method_variables["train_ind_to_patt"] == method_variables["train_ind_to_patt"][i].unsqueeze(
      0)).all(dim=1).sum().item()
    assert matches == 1

  # compute factor for first polytope pattern
  fst_pattern = method_variables["train_ind_to_patt"][0]
  fst_factor = 0.

  num_queries = 2 ** method_variables["N"]
  total_prob = 0.
  total_risk = 0.
  true_prob_norm = 0.
  true_closeness_mass = 0.
  for batch_i in range((num_queries // batch_sz) + 1):
    start = batch_i * batch_sz
    end = min(num_queries, (batch_i + 1) * batch_sz)
    if end == start: continue

    query_pattern = torch.zeros(end - start, method_variables["N"], dtype=torch.bool,
                                device=device(config.cuda))
    for n in range(start, end):
      for mask in range(1, method_variables["N"] + 1):
        query_pattern[n - start, mask - 1] = (n & (
        1 << (method_variables["N"] - mask))) > 0  # fill res left to right
      if n < 10:
        print((n, bin(n), query_pattern[n - start]))

    closeness = compute_closeness(config, query_pattern,
                                  method_variables)  # num queries x num training samples
    true_closeness_mass += closeness.sum().item()

    prob_norm_part = (closeness * method_variables["sample_counts"].unsqueeze(0)).sum(dim=1)
    true_prob_norm += prob_norm_part.sum().item()

    smooth_prob = prob_norm_part / method_variables["global_prob_norm"]  # num queries
    total_prob += smooth_prob.sum().item()

    emprisk = compute_emprisk(closeness, method_variables)
    total_risk += (smooth_prob * emprisk).sum().item()

    shared_with_fst = (query_pattern == fst_pattern.unsqueeze(0)).sum(dim=1)  # num queries
    closeness_with_fst = close_par(config, shared_with_fst, method_variables["N"],
                                   method_variables["p"])  # checked, correct
    fst_factor += (smooth_prob * closeness_with_fst).sum().item()

    print(
      "batch %d / %d (%d - %d), so far: prob %s, risk %s, true prob norm %s, true closeness mass %s, %s" %
      (batch_i, num_queries // batch_sz, start, end, total_prob, total_risk, true_prob_norm,
       true_closeness_mass, datetime.now()))
    stdout.flush()

  print(
    "CHECK COMPUTED: total prob %s, training error %s, prob norm %s, closeness mass %s, example normalisation term %s, %s" %
    (total_prob, total_risk, true_prob_norm, true_closeness_mass, fst_factor, datetime.now()))

  print(
    "Compare the CHECK COMPUTED values against CHECK ORIG values, they should be the same within some floating point tolerance.")
  print("---------------------\n")
  stdout.flush()
