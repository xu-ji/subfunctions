import torch
import torch.nn.functional as F


def entropy_pre(config, model, train_loader, val_loader):
  return model, {}  # nothing


def entropy_metric(config, method_variables, model, imgs, targets):
  with torch.no_grad():
    preds = model(imgs)
  softmax_preds = F.softmax(preds, dim=1)
  preds_flat = preds.argmax(dim=1)
  correct = preds_flat.eq(targets)

  entropy = (- softmax_preds * torch.log(softmax_preds)).sum(dim=1)

  return entropy, correct
