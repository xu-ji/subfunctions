import torch
import torch.nn.functional as F
from copy import deepcopy
from util.general import device
from util.methods.tack_et_al import add_class_distance_hooks, start_name, get_keywords, \
  tack_et_al_pattern_keywords, pop_class_distance_feats
from torchvision import transforms
import os
from datetime import datetime
from sys import stdout
import numpy as np

# Bergman doesn't work on supervised features at all!

def bergman_et_al_pre(config, model, train_loader, val_loader):
  method_variables = {}

  # Add hooks to model
  assert config.model in ["vgg16_bn", "resnet50model"]
  names, hook_handles = [], []
  add_class_distance_hooks(model, start_name, get_keywords(model, tack_et_al_pattern_keywords),
                           names, hook_handles)
  print("(bergman) layer names: %s" % names)

  method_variables["pattern_layer_names"] = names
  method_variables["num_layers"] = len(names)

  # compute mean features for each of M transforms
  orig_dataset = train_loader.dataset

  mean_features = []
  for m in range(config.bergman_et_al_M):
    new_dataset = deepcopy(orig_dataset)
    new_dataset.transform = random_transform(config) # tested this works
    new_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=config.batch_size,
                                             shuffle=False, num_workers=config.workers,
                                             pin_memory=True)

    feats = None
    num_feats = 0
    for batch_i, (imgs, targets) in enumerate(new_dataloader):
      if batch_i % max(1, len(new_dataloader) // 100) == 0:
        print("(berman) training data for m %d: %d / %d, %s" % (
          m, batch_i, len(new_dataloader), datetime.now()))
        stdout.flush()

      imgs = imgs.to(device(config.cuda))
      with torch.no_grad():
        _ = model(imgs)

      curr_feats = pop_class_distance_feats(config, model, method_variables,
                                            tack_et_al_pattern_keywords)
      # num layers: num
      # samples, feat len
      curr_feats = torch.cat(curr_feats, dim=1) # should just be 1 tensor in list though
      curr_feats = curr_feats.sum(dim=0)
      if feats is None:
        feats = curr_feats
      else:
        feats += curr_feats
      num_feats += imgs.shape[0]

    avg_feats = feats / float(num_feats)
    assert (len(avg_feats.shape) == 1)
    mean_features.append(avg_feats)

  mean_features = torch.stack(mean_features, dim=0) # M, feat len
  assert (mean_features.shape == (config.bergman_et_al_M, avg_feats.shape[0]))

  print("mean feature info, for each of M:")
  np.set_printoptions(precision=32)
  print((mean_features.mean(dim=1).cpu().numpy().min(), mean_features.mean(dim=1).cpu().numpy().max()))
  print((mean_features.max(dim=1)[0].cpu().numpy().min(), mean_features.max(dim=1)[0].cpu().numpy().max()))
  print((mean_features.min(dim=1)[0].cpu().numpy().min(), mean_features.min(dim=1)[0].cpu().numpy().max()))
  print((mean_features.std(dim=1).cpu().numpy().min(), mean_features.std(dim=1).cpu().numpy().max()))

  mean_features = mean_features.transpose(0, 1).unsqueeze(0) # 1, feat len, M

  method_variables["mean_features"] = mean_features

  return model, method_variables


def bergman_et_al_metric(config, method_variables, model, imgs, targets):
  with torch.no_grad():
    preds = model(imgs)
  preds_flat = preds.argmax(dim=1)
  correct = preds_flat.eq(targets)

  curr_feats = pop_class_distance_feats(config, model, method_variables,
                                        tack_et_al_pattern_keywords)  # num layers: num samples, feat len
  curr_feats = torch.cat(curr_feats, dim=1) # num samples, feat len

  # get each sample x each mean feature

  curr_feats = curr_feats.unsqueeze(dim=2)

  # sample, feat len, M
  closeness =  torch.linalg.norm(curr_feats - method_variables["mean_features"], ord=2, dim=1) # middle
  assert (closeness.shape == (imgs.shape[0], config.bergman_et_al_M))
  closeness = torch.exp(- torch.pow(closeness, 2)) # sample, M

  # divide result for each by sum of that row for the sample
  closeness = closeness / closeness.sum(dim=1, keepdim=True)
  assert (closeness.shape == (imgs.shape[0], config.bergman_et_al_M))

  unreliability = - torch.log(closeness).sum(dim=1)
  assert (unreliability.shape == (imgs.shape[0],))

  print("unreliability range: %s " % str((unreliability.max(), unreliability.min())))

  return unreliability, correct


def random_transform(config):
  assert config.data in ["cifar10", "cifar100"]

  tfs = []

  crop_start = np.random.choice(32, size=2, replace=True)
  crop_sz = np.random.choice(26, size=2, replace=True)
  tfs.append(lambda x: transforms.functional.crop(x, crop_start[0], crop_start[1], crop_sz[0], crop_sz[1]))

  if np.random.rand() > 0.5:
    tfs.append(lambda x : transforms.functional.hflip(x))

  colour_hue = np.random.rand() - 0.5 # [0.5, 0.5]
  colour_saturation = np.random.rand() * 2 # [0, 2]
  tfs.append(lambda x : transforms.functional.adjust_hue(x, colour_hue))
  tfs.append(lambda x : transforms.functional.adjust_saturation(x, colour_saturation))

  if np.random.rand() > 0.5:
    tfs.append(lambda x: transforms.functional.rgb_to_grayscale(x, num_output_channels=3))

  tfs.append(transforms.ToTensor())

  stats_fname = os.path.join(config.data_root, "%s_stats.pytorch" % config.data)
  datasets_stats = torch.load(stats_fname)
  normalize = transforms.Normalize(mean=datasets_stats["mean"],
                                   std=datasets_stats["std"])
  tfs.append(normalize)

  transform = transforms.Compose(tfs)
  return transform