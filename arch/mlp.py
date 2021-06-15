import torch
import torch.nn as nn


class MLP(nn.Module):
  def __init__(self, layer_szs, num_classes=None, in_feats=None):
    super(MLP, self).__init__()

    self.layer_szs = layer_szs

    for i, nf in enumerate(self.layer_szs):
      setattr(self, "layer%d" % i, nn.Linear(in_feats, nf))
      setattr(self, "relu%d" % i, nn.ReLU())

      in_feats = nf

    self.last_layer = nn.Linear(in_feats, num_classes)

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)  # 28 * 28

    for i, nf in enumerate(self.layer_szs):
      lin = getattr(self, "layer%d" % i)
      relu = getattr(self, "relu%d" % i)
      x = relu(lin(x))

    return self.last_layer(x)
