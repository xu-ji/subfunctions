import argparse
import os.path as osp
import torch
from util.data import *
from util.general import *
from torch.nn import CrossEntropyLoss
from torch import optim
from datetime import datetime
from sys import stdout
from functools import partial
from scripts.global_constants import *

import arch

config = argparse.ArgumentParser(allow_abbrev=False)
config.add_argument("--data", type=str, choices=["mnist", "cifar10", "cifar100", "two_moons"],
                    required=True)
config.add_argument("--data_root", type=str, default=CIFAR_DATA_ROOT)
config.add_argument("--batch_size", type=int, default=200)
config.add_argument("--workers", type=int, default=1)

config.add_argument("--models_root", type=str, default=DEFAULT_MODELS_ROOT)
config.add_argument("--model_args", type=int, nargs="+",
                    default=[])  # for mnist, hidden layer sizes
config.add_argument("--seed", type=int, default=0)
config.add_argument("--lr", type=float, default=0.1)
config.add_argument("--weight_decay", type=float, default=5e-4)
config.add_argument("--epochs", type=int, default=100)

config.add_argument("--lr_sched_epoch_gap", type=int, default=30)
config.add_argument("--lr_sched_mult", type=float, default=0.1)

config.add_argument("--cuda", default=False, action="store_true")
config.add_argument("--restart", default=False, action="store_true")

config.add_argument("--model", type=str, required=True)
config.add_argument("--val_pc", type=float, default=0.15)

config.add_argument("--two_moons_norm_data", default=False, action="store_true")

config = config.parse_args()

if not osp.exists(osp.join(config.data_root, "%s_stats.pytorch" % config.data)):
  compute_dataset_stats(config)

save_fname = osp.join(config.models_root,
                      "%s_%d_%s.pytorch" % (config.data, config.seed, config.model))

set_seed(config.seed)

train_loader, val_loader, test_loader = get_data(config, val_pc=config.val_pc,
                                                 training=True)  # same as prediction filtering
# data loading incl val pc

print("Config: %s" % config)

num_classes = classes_per_dataset[config.data]

if config.data == "cifar10" or config.data == "cifar100":
  if config.model == "vgg16_bn":
    model = arch.vgg16_bn(num_classes)
  elif config.model == "resnet50model":
    model = arch.resnet50model(num_classes)

elif config.data == "mnist" and config.model == "mlp":
  model = arch.MLP(layer_szs=config.model_args, num_classes=num_classes, in_feats=(28 * 28))

elif config.data == "two_moons" and config.model == "mlp":
  model = arch.MLP(layer_szs=config.model_args, num_classes=num_classes, in_feats=2)
  from util.two_moons import render_two_moons

  render_two_moons(config, [train_loader, test_loader])

else:
  raise NotImplementedError

print(model)

accs = []
next_ep = 0

# Train
model.to(device(config.cuda)).train()
opt = optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=config.weight_decay)

lr_sched = partial(lr_sched_maker, config.lr_sched_epoch_gap, config.lr_sched_mult)
sched = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lr_sched)

criterion = CrossEntropyLoss().to(device(config.cuda))

if config.restart:
  saved = torch.load(save_fname)
  model = saved["model"]
  opt = saved["opt"]
  sched = saved["sched"]
  accs = saved["accs"]
  next_ep = saved["next_ep"]
  if accs[-1][0] == next_ep:
    print("trimming stored accs")
    accs = accs[:-1]
  print("restarting from saved: ep %d" % next_ep)

torch.save({"model": model, "next_ep": next_ep, "accs": accs, "opt": opt, "sched": sched},
           save_fname)

for ep in range(next_ep, config.epochs):
  print("epoch %d %s, lr %f" % (ep, datetime.now(), opt.param_groups[0]["lr"]))
  if ep % 10 == 0:
    torch.save({"model": model, "next_ep": ep, "accs": accs, "opt": opt, "sched": sched},
               save_fname)

    acc = evaluate(config, model, test_loader)
    print(acc)
    accs.append((ep, acc))
  stdout.flush()

  for batch_i, (imgs, targets) in enumerate(train_loader):
    opt.zero_grad()
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    preds = model(imgs)  # no softmax
    loss = criterion(preds, targets)

    loss.backward()
    opt.step()

  sched.step()

# Save
final_acc = evaluate(config, model, test_loader)
accs.append((config.epochs, final_acc))
print("all accs %s" % accs)
torch.save({"model": model, "acc": final_acc, "accs": accs,
            "next_ep": ep, "opt": opt, "sched": sched}, save_fname)
