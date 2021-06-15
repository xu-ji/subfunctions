import torch
import torch.nn.functional as F


def max_response_pre(config, model, train_loader, val_loader):
  return model, {}  # nothing


def max_response_metric(config, method_variables, model, imgs, targets):
  with torch.no_grad():
    preds = model(imgs)
  softmax_preds = F.softmax(preds, dim=1)
  preds_flat = preds.argmax(dim=1)
  correct = preds_flat.eq(targets)

  unreliability = 1 - softmax_preds.max(dim=1)[0]  # 1 - likelihood of most certain class

  return unreliability, correct
