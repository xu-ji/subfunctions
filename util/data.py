import os.path as osp
import torch
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, Dataset
from util.general import device
from datetime import datetime
from sys import stdout
from sklearn import datasets as sk_datasets
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

classes_per_dataset = {"mnist": 10, "cifar10": 10, "cifar100": 100, "two_moons": 2}


def compute_dataset_stats(config):
  # For certain datasets, compute normalization stats and save to file for use later

  if config.data in ["mnist", "two_moons"]: return

  if config.data == "imagenet":
    traindir = osp.join(config.data_root, "images", "train")
    test_tf = transforms.Compose(
      [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    train_data = datasets.ImageFolder(traindir, transform=test_tf)

  elif config.data == "cifar10":
    train_data = datasets.CIFAR10(root=config.data_root, train=True, download=False,
                                  transform=transforms.ToTensor())

  elif config.data == "cifar100":
    train_data = datasets.CIFAR100(root=config.data_root, train=True, download=False,
                                   transform=transforms.ToTensor())
  elif config.data == "svhn":
    train_data = datasets.SVHN(root=config.data_root, split="train", download=False,
                               transform=transforms.ToTensor())

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                             shuffle=False, num_workers=config.workers,
                                             pin_memory=True)

  avg = None
  num_samples = 0
  for i, (imgs, _) in enumerate(train_loader):
    if i < 10 or i % 100 == 0: print("avg: %d / %d, %s" % (i, len(train_loader), datetime.now()))
    stdout.flush()

    imgs = imgs.to(device(config.cuda))
    summed_pixels = imgs.sum(dim=[0, 2, 3], keepdim=False)
    if avg is None:
      avg = summed_pixels
    else:
      avg += summed_pixels

    num_samples += imgs.shape[0] * imgs.shape[2] * imgs.shape[3]

  avg = avg / float(num_samples)

  std = None
  for i, (imgs, _) in enumerate(train_loader):
    if i < 10 or i % 100 == 0: print("std: %d / %d, %s" % (i, len(train_loader), datetime.now()))
    stdout.flush()

    imgs = imgs.to(device(config.cuda))
    sq_diff = (imgs - avg.unsqueeze(0).unsqueeze(2).unsqueeze(3)).pow(2)

    summed_sq_diff = sq_diff.sum(dim=[0, 2, 3], keepdim=False)
    if std is None:
      std = summed_sq_diff
    else:
      std += summed_sq_diff

  std = (std / float(num_samples - 1)).sqrt()
  stats = {"mean": avg.cpu(), "std": std.cpu()}
  print("Computed stats: %s" % stats)

  fname = osp.join(config.data_root, "%s_stats.pytorch" % config.data)
  torch.save(stats, fname)


def get_data(config, val_pc, training=False):
  if isinstance(config.seed, list):
    seed = config.seed[0]
  else:
    seed = config.seed

  if not (config.data in ["mnist", "two_moons"]):
    stats_fname = osp.join(config.data_root, "%s_stats.pytorch" % config.data)
    if not osp.exists(stats_fname):
      compute_dataset_stats(config)

    datasets_stats = torch.load(stats_fname)
    print(datasets_stats)

  if config.data == "imagenet":
    traindir = osp.join(config.data_root, "images", "train")
    valdir = osp.join(config.data_root, "images", "val")
    normalize = transforms.Normalize(mean=datasets_stats["mean"],
                                     std=datasets_stats["std"])
    # ours {'mean': tensor([0.4845, 0.4541, 0.4026]), 'std': tensor([0.2724, 0.2637, 0.2761])}
    test_tf = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])

    if training:
      train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
      ])
    else:
      train_tf = test_tf

    train_data = datasets.ImageFolder(traindir, train_tf)
    test_data = datasets.ImageFolder(valdir, test_tf)

  elif config.data == "mnist":
    train_data = datasets.MNIST(config.data_root, transform=transforms.ToTensor(), train=True,
                                download=False)
    test_data = datasets.MNIST(config.data_root, transform=transforms.ToTensor(), train=False,
                               download=False)

  elif config.data == "cifar10":
    if training or (not config.method == "explicit_density"):
      print("Normal cifar10 data loading")
      normalize = transforms.Normalize(mean=datasets_stats["mean"],
                                       std=datasets_stats["std"])

      test_tf = transforms.Compose([transforms.ToTensor(), normalize])
      # ours: {'mean': tensor([0.4914, 0.4822, 0.4465]), 'std': tensor([0.2470, 0.2435, 0.2616])}

      if training:
        train_tf = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
        ])
      else:
        train_tf = test_tf
    else:
      print("Different Cifar10 preprocessing for residual flows")
      # as defined in residual flows github
      test_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        add_noise,
      ])
      train_tf = test_tf  # using pretrained density model

    train_data = datasets.CIFAR10(root=config.data_root, train=True, download=False,
                                  transform=train_tf)
    test_data = datasets.CIFAR10(root=config.data_root, train=False, download=False,
                                 transform=test_tf)

  elif config.data == "cifar100":
    if training or (not config.method == "explicit_density"):
      print("Normal cifar100 data loading")
      normalize = transforms.Normalize(mean=datasets_stats["mean"],
                                       std=datasets_stats["std"])

      test_tf = transforms.Compose([transforms.ToTensor(), normalize])
      # ours: {'mean': tensor([0.5071, 0.4865, 0.4409]), 'std': tensor([0.2673, 0.2564, 0.2762])}

      if training:
        train_tf = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(15),
          transforms.ToTensor(),
          normalize,
        ])
      else:
        train_tf = test_tf
    else:
      print("Different Cifar100 preprocessing for residual flows")
      test_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        add_noise,
      ])
      train_tf = test_tf  # using pretrained density model

    train_data = datasets.CIFAR100(root=config.data_root, train=True, download=False,
                                   transform=train_tf)
    test_data = datasets.CIFAR100(root=config.data_root, train=False, download=False,
                                  transform=test_tf)

  elif config.data == "two_moons":
    num_train_samples = 5000
    train_noise = 0.05
    num_test_samples = 1000
    test_noise = 0.05

    x_train, y_train = sk_datasets.make_moons(n_samples=num_train_samples, shuffle=True,
                                              noise=train_noise,
                                              random_state=seed)  # float64 in [-2.5,
    # 2.5] - depends on noise, int64
    x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train)
    if config.two_moons_norm_data: x_train = normalize_2D(x_train)
    train_data = TensorDataset(x_train, y_train)

    x_test, y_test = sk_datasets.make_moons(n_samples=num_test_samples, shuffle=True,
                                            noise=test_noise, random_state=seed)
    x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test)
    if config.two_moons_norm_data: x_test = normalize_2D(x_test)
    test_data = TensorDataset(x_test, y_test)

  elif config.data == "svhn":
    if (not config.method == "explicit_density"):
      normalize = transforms.Normalize(mean=datasets_stats["mean"],
                                       std=datasets_stats["std"])

      test_tf = transforms.Compose([transforms.ToTensor(), normalize])
      assert not training
      train_tf = test_tf

    else:
      print("Different SVHN preprocessing for residual flows")
      # as defined in residual flows github
      test_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        add_noise,
      ])
      train_tf = test_tf  # using pretrained density model

    train_data = datasets.SVHN(root=config.data_root, split="train", download=False,
                               transform=train_tf)
    test_data = datasets.SVHN(root=config.data_root, split="test", download=False,
                              transform=test_tf)

  num_val = int(len(train_data) * val_pc)
  new_train_data, val_data = torch.utils.data.random_split(train_data,
                                                           [len(train_data) - num_val, num_val],
                                                           generator=torch.Generator().manual_seed(
                                                             seed))
  for dataset in [new_train_data, val_data, test_data]: assert isinstance(dataset, Dataset)

  train_loader = torch.utils.data.DataLoader(new_train_data, batch_size=config.batch_size,
                                             shuffle=training, num_workers=config.workers,
                                             pin_memory=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size,
                                           shuffle=False, num_workers=config.workers,
                                           pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                            shuffle=False, num_workers=config.workers,
                                            pin_memory=True)

  print("Dataset stats:")
  for loader in [train_loader, val_loader, test_loader]: print_data_stats(loader)

  return train_loader, val_loader, test_loader


def print_data_stats(loader):
  counts = defaultdict(int)
  for xs, ys in loader:
    for y in ys.unique():
      counts[y.item()] += (ys == y).sum().item()

  print(counts)
  # for k, v in counts.items():
  #  print("class %s: %s" % (k, v))


def normalize_2D(xs):
  print("normalizing 2D data")
  assert (len(xs.shape) == 2 and xs.shape[1] == 2)
  mean = xs.mean(dim=0, keepdim=True)
  std = xs.std(dim=0, keepdim=True)

  return (xs - mean) / std


def add_noise(x, nvals=256):
  """
  For residual flows method only
  [0, 1] -> [0, nvals] -> add noise -> [0, 1]
  """
  noise = x.new().resize_as_(x).uniform_()
  x = x * (nvals - 1) + noise
  x = x / nvals
  return x


def merge_data(config, new_distr_loader_ood, old_test_loader):
  new_x, new_y = [], []
  for i in range(len(old_test_loader.dataset)):
    new_x.append(old_test_loader.dataset[i][0])
    new_y.append(True)  # layer used as "correct" or positive

  for i in range(len(new_distr_loader_ood.dataset)):
    new_x.append(new_distr_loader_ood.dataset[i][0])  # already transformed
    new_y.append(False)  # later used as "incorrect" or negative

  new_x = torch.stack(new_x, dim=0)
  assert new_x.dtype == torch.float
  assert len(new_x.shape) == 4
  new_y = torch.tensor(new_y, dtype=torch.bool)

  new_dataset = TensorDataset(new_x, new_y)

  return torch.utils.data.DataLoader(new_dataset, batch_size=config.batch_size,
                                     shuffle=True, num_workers=config.workers, pin_memory=True)
