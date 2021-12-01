import torch
import torch.nn.functional as F
from util.methods.subfunctions import start_name, get_keywords
from util.methods.class_distance import add_class_distance_hooks, pop_class_distance_feats
from sys import stdout
from datetime import datetime
from util.general import device
from sklearn.cluster import MiniBatchKMeans #KMeans
import numpy as np


tack_et_al_pattern_keywords = {
  "VGG": [("fieldname_exact", "root_classifier_5", "out")], # just before last linear layer
  "ResNetModel": [("fieldname_exact", "root_linear_feats_2", "out")]
}

def tack_et_al_pre(config, model, train_loader, val_loader):
  # construct coreset with size 10% of samples
  method_variables = {}

  # Add hooks to model
  assert config.model in ["vgg16_bn", "resnet50model"]
  names, hook_handles = [], []
  add_class_distance_hooks(model, start_name, get_keywords(model, tack_et_al_pattern_keywords),
                           names, hook_handles)
  print("(tack_et_al_pre) layer names: %s" % names)

  method_variables["pattern_layer_names"] = names
  method_variables["num_layers"] = len(names)

  """
  # Run k-means on features
  # https://stackoverflow.com/questions/46409846/using-k-means-with-cosine-similarity-python
  batch_size = int(config.tack_et_al_batch_size_pc * num_samples)
  large_train_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=config.workers,
                                                   pin_memory=True)
  """
  feats = []
  for batch_i, (imgs, targets) in enumerate(train_loader):
    if batch_i % max(1, len(train_loader) // 100) == 0:
      print("(tack_et_al_pre) training data: %d / %d, %s" % (
      batch_i, len(train_loader), datetime.now()))
      stdout.flush()

    imgs = imgs.to(device(config.cuda))
    with torch.no_grad():
      _ = model(imgs)

    curr_feats = pop_class_distance_feats(config, model, method_variables,
                                          tack_et_al_pattern_keywords) # num layers: num samples, feat len
    curr_feats = torch.cat(curr_feats, dim=1)

    # (min) (squared) euclidian dist in normalized vector space = (min) (2*) cosine distance in original vector space
    curr_feats = F.normalize(curr_feats, p=2.0, dim=1)
    feats.append(curr_feats.cpu())

  feats = torch.cat(feats, dim=0) # num samples, feat len
  #feats = feats.numpy()
  method_variables["feats"] = feats
  #method_variables["num_batches"] =  int(np.ceil(feats.shape[0] / float(config.tack_et_al_batch_size)))

  print("(tack_et_al_pre) feats size %s" % str(feats.shape))

  """
  num_samples = len(train_loader.dataset)
  num_centroids = int(config.tack_et_al_coreset_pc * num_samples)
  method_variables["num_centroids"] = num_centroids
  print("(tack_et_al_pre) num centroids: %s" % num_centroids)
  
  # labels each sample with a label such that intra-label dists are minimized
  # is the centroid the same in both cases? centroid is mean of normalized samples, fine
  batch_size = int(feats.shape[0] * config.tack_et_al_batch_size_pc)
  assert feats.shape[0] % batch_size == 0
  num_batches = int(feats.shape[0] / batch_size)
  kmeans = MiniBatchKMeans(n_clusters=num_centroids, random_state=0, batch_size=batch_size)
  for b_i in range(num_batches):
    print("(tack_et_al_pre) fitting %d / %d (%d, %d)" % (b_i, num_batches, feats.shape[0], batch_size))
    stdout.flush()
    kmeans = kmeans.partial_fit(feats[(batch_size * b_i):(batch_size * (b_i + 1))])

  centroid_len = np.sqrt(np.square(kmeans.cluster_centers_).sum(axis=1)[:, None])
  method_variables["normed_centroids"] = kmeans.cluster_centers_ / centroid_len
  """
  return model, method_variables


def tack_et_al_metric(config, method_variables, model, imgs, targets):
  # normalize centroids before dot product with these test samples to get cosine similarity

  with torch.no_grad():
    preds = model(imgs)
  preds_flat = preds.argmax(dim=1)
  correct = preds_flat.eq(targets)

  curr_feats = pop_class_distance_feats(config, model, method_variables,
                                        tack_et_al_pattern_keywords)  # num layers: num samples, feat len
  curr_feats = torch.cat(curr_feats, dim=1)

  norms = torch.linalg.norm(curr_feats, ord=2, dim=1)

  curr_feats = F.normalize(curr_feats, p=2, dim=1)
  curr_feats = curr_feats.cpu() #.numpy()

  num_batches = int(np.ceil(curr_feats.shape[0] / float(config.tack_et_al_split_batch)))
  highest_sims = []
  for b_i in range(num_batches):
    curr_feats_batch = curr_feats[(b_i * config.tack_et_al_split_batch):
      min((b_i + 1) * config.tack_et_al_split_batch, curr_feats.shape[0])]
    # num train samples, num feats, split batch
    sims = method_variables["feats"].unsqueeze(2) * curr_feats_batch.transpose(0, 1).unsqueeze(0)
    curr_feats_highest_sims, _ = sims.sum(dim=1).max(dim=0) # maximum of dot products
    assert curr_feats_highest_sims.shape == (curr_feats_batch.shape[0],)
    highest_sims.append(curr_feats_highest_sims)

  highest_sims = torch.cat(highest_sims, dim=0)

  """
  sims = np.dot(method_variables["normed_centroids"], curr_feats.T) # num_centroids, num samples
  assert sims.shape == (method_variables["num_centroids"], imgs.shape[0])
  highest_sims = sims.max(axis=0)
  assert highest_sims.shape == (imgs.shape[0],) and norms.shape == (imgs.shape[0],)
  """

  unreliability = - torch.tensor(highest_sims, device=device(config.cuda)) * norms # , device=device(config.cuda)
  return unreliability, correct

"""
def pad_samples(curr_feats, size):
  if curr_feats.shape[0] == size: return curr_feats

  pad = np.random.choice(curr_feats.shape[0], size=(size - curr_feats.shape[0]), replace=False)
  padded = curr_feats[pad]
  if len(pad) == 1: padded = padded.unsqueeze(0)
  res = torch.cat([curr_feats, padded], dim=0)
  assert res.shape[0] == size
  return res
"""
