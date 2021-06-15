import torch
import torch.nn.functional as F


def ensemble_var_pre(config, model, train_loader, val_loader):
  return model, {}  # nothing


def ensemble_var_metric(config, method_variables, model, imgs, targets):
  assert isinstance(model, list) and len(model) > 1

  res = []
  for m_i in range(len(model)):
    with torch.no_grad():
      preds = model[m_i](imgs)
    softmax_preds = F.softmax(preds, dim=1)
    res.append(softmax_preds)
  res = torch.stack(res, dim=0)  # num models, num samples, num classes
  avg_preds = res.mean(dim=0)  # num_samples, classes
  assert len(avg_preds.shape) == 2
  top_classes = avg_preds.argmax(dim=1)  # num_samples
  assert len(top_classes.shape) == 1

  correct = top_classes.eq(targets)

  top_res = res[:, F.one_hot(top_classes, num_classes=res.shape[2]).to(torch.bool)]  # checked
  assert top_res.shape == (len(model), imgs.shape[0])
  unreliability = top_res.var(dim=0)  # var across models for most likely class
  assert unreliability.shape == (imgs.shape[0],)

  return unreliability, correct
