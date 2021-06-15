import torch
import torch.nn.functional as F


def margin_pre(config, model, train_loader, val_loader):
  return model, {}  # nothing


def margin_metric(config, method_variables, model, imgs, targets):
  with torch.no_grad():
    preds = model(imgs)
  softmax_preds = F.softmax(preds, dim=1)
  preds_flat = softmax_preds.argmax(dim=1)
  correct = preds_flat.eq(targets)

  top_scores = softmax_preds.max(dim=1)[0]

  softmax_preds[torch.arange(softmax_preds.shape[0]), preds_flat] = -1.

  second_top_scores = softmax_preds.max(dim=1)[0]

  unreliability = - (top_scores - second_top_scores) # bigger the difference, lower the unreliability

  return unreliability, correct
