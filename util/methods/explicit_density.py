import numpy as np
import torch
from util.general import device
import sys
import math
from scripts.global_constants import RESIDUAL_FLOWS_PATH

sys.path.insert(0, RESIDUAL_FLOWS_PATH)
import lib.layers as layers  # from residual flows library just linked
import lib.layers.base as base_layers
from lib.resflow import ResidualFlow

for mod in [layers, base_layers, ResidualFlow]:
  assert mod is not None


def explicit_density_pre(config, model, train_loader, val_loader):
  method_variables = {}

  if config.data in ["cifar10", "cifar100"]:
    # https://github.com/rtqichen/residual-flows/
    # default param defs - load state dict later

    saved_files = torch.load(config.density_model_path_pattern % config.data)
    density_model = saved_files["density_model"]

    update_lipschitz(density_model)

    density_model.eval()

    input_size = saved_files["input_size"]
    input_size = (config.batch_size,) + input_size[1:]

    method_variables["density_model"] = density_model
    method_variables["args"] = saved_files["args"]
    method_variables["input_size"] = input_size
    method_variables["n_classes"] = saved_files["n_classes"]
    method_variables["im_dim"] = saved_files["im_dim"]

    print("input size %s, n_classes %s, im_dim %s" % (method_variables["input_size"],
                                                      method_variables["n_classes"],
                                                      method_variables["im_dim"]))
    sys.stdout.flush()

  else:
    raise NotImplementedError

  return model, method_variables


def explicit_density_metric(config, method_variables, model, imgs, targets):
  with torch.no_grad():
    preds = model(imgs)
  preds_flat = preds.argmax(dim=1)
  correct = preds_flat.eq(targets)

  with torch.no_grad():
    bpd, _, _, _ = compute_loss(method_variables, imgs)

  assert bpd.shape == (imgs.shape[0],)

  # print(("bpd", bpd.max(), bpd.min(), bpd.abs().max(), bpd.abs().min(), bpd.mean()))
  # sys.stdout.flush()

  return bpd, correct


# Helpers

def update_lipschitz(model):
  with torch.no_grad():
    for m in model.modules():
      if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m,
                                                                     base_layers.SpectralNormLinear):
        m.compute_weight(update=True)
      if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m,
                                                                    base_layers.InducedNormLinear):
        m.compute_weight(update=True)


def compute_loss(method_variables, x, beta=1.0):
  model = method_variables["density_model"]
  args = method_variables["args"]
  input_size = method_variables["input_size"]
  n_classes = method_variables["n_classes"]
  im_dim = method_variables["im_dim"]

  # print(("input_size", input_size))

  bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
  logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

  if args.data == 'celeba_5bit':
    nvals = 32
  elif args.data == 'celebahq':
    nvals = 2 ** args.nbits
  else:
    nvals = 256

  x, logpu = add_padding(method_variables, x, nvals)

  if args.squeeze_first:
    squeeze_layer = layers.SqueezeLayer(2)
    x = squeeze_layer(x)

  if args.task == 'hybrid':
    z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
    z, delta_logp = z_logp
  elif args.task == 'density':
    z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
  elif args.task == 'classification':
    z, logits_tensor = model(x.view(-1, *input_size[1:]), classify=True)

  if args.task in ['density', 'hybrid']:
    # log p(z)
    # print(("z", z.shape, z.min(), z.max(), z.abs().min(), z.abs().max()))
    # print(("delta_logp", delta_logp.shape, delta_logp.min(), delta_logp.max(), delta_logp.abs(
    # ).min(), delta_logp.abs().max()))

    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
    # print(("logpz", logpz.shape, logpz.min(), logpz.max(), logpz.abs().min(), logpz.abs().max()))

    # log p(x)
    logpx = logpz - beta * delta_logp - np.log(nvals) * (
      args.imagesize * args.imagesize * (im_dim + args.padding)
    ) - logpu
    # print(("logpx", logpx.shape, logpx.min(), logpx.max(), logpx.abs().min(), logpx.abs().max()))

    # we want to report per image...
    # bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)
    bits_per_dim = -logpx.squeeze(1) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

    # print(("bits_per_dim sizes", (-torch.mean(logpx)).shape, args.imagesize, im_dim, np.log(2)))
    # print(("bits per dim", bits_per_dim.shape, bits_per_dim.min(), bits_per_dim.max(),
    # bits_per_dim.abs().min(), bits_per_dim.abs().max()))

    logpz = torch.mean(logpz).detach()
    delta_logp = torch.mean(-delta_logp).detach()

  return bits_per_dim, logits_tensor, logpz, delta_logp


def add_padding(method_variables, x, nvals=256):
  args = method_variables["args"]

  # Theoretically, padding should've been added before the add_noise preprocessing.
  # nvals takes into account the preprocessing before padding is added.
  if args.padding > 0:
    if args.padding_dist == 'uniform':
      u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
      logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
      return torch.cat([x, u / nvals], dim=1), logpu
    elif args.padding_dist == 'gaussian':
      u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2,
                                                                                nvals / 8)
      logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
      return torch.cat([x, u / nvals], dim=1), logpu
    else:
      raise ValueError()
  else:
    return x, torch.zeros(x.shape[0], 1).to(x)


def standard_normal_logprob(z):
  logZ = -0.5 * math.log(2 * math.pi)
  return logZ - z.pow(2) / 2


def normal_logprob(z, mean, log_std):
  mean = mean + torch.tensor(0.)
  log_std = log_std + torch.tensor(0.)
  c = torch.tensor([math.log(2 * math.pi)]).to(z)
  inv_sigma = torch.exp(-log_std)
  tmp = (z - mean) * inv_sigma
  return -0.5 * (tmp * tmp + 2 * log_std + c)
