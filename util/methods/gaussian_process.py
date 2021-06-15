import tensorflow as tf
import functools
import edward2 as ed
from util.data import classes_per_dataset
from util.methods.subfunctions import start_name, get_keywords, fits_keyword
from util.methods.class_distance import add_class_distance_hooks, pop_class_distance_feats
import torch
from datetime import datetime
from sys import stdout
from util.general import device
import numpy as np

# Using same procedure (minus the spectral norm) as:
# https://github.com/google/uncertainty-baselines/blob/9e12ff7c1573b83e6037fd674220215ff368d5c1/uncertainty_baselines/models/resnet50_sngp.py


gaussian_process_pattern_keywords = {
  "VGG": [("fieldname_exact", "root_features_43", "out")], # for vgg16
  "ResNetModel": [("fieldname_exact", "root_linear_feats", "out")] # for resnet50model
}


# Settings taken from their code
gp_bias = 0.
gp_input_normalization = False
gp_random_feature_type = 'rff'
gp_cov_discount_factor = -1.
gp_cov_ridge_penalty = 1.
gp_output_imagenet_initializer = True  # means using small standard deviation
gp_l2_weight = 1e-4
gp_base_learning_rate = 0.1
gp_train_epochs = 90
gp_one_minus_momentum = 0.1


def gaussian_process_pre(config, model, train_loader, val_loader):
  if isinstance(config.seed, list) or isinstance(config.seed, tuple):
    seed = config.seed[0]
  else:
    seed = config.seed
  print("(gaussian_processes_pre) setting seed %s" % seed)
  tf.random.set_seed(seed)

  print("(gaussian_processes_pre) devices %s" % tf.config.list_physical_devices('GPU'))
  tf.keras.backend.set_image_data_format('channels_first')

  method_variables = {}
  method_variables["num_classes"] = classes_per_dataset[config.data]

  # Add hooks
  names, hook_handles = [], []
  add_class_distance_hooks(model, start_name, get_keywords(model, gaussian_process_pattern_keywords),
                           names, hook_handles)
  assert len(names) == 1
  print("(gaussian_processes_pre) name %s" % names)
  method_variables["pattern_layer_names"] = names

  # Figure out feature shape
  empty = train_loader.dataset[0][0].unsqueeze(0).to(device(config.cuda))
  with torch.no_grad():
    _ = model(empty)
  curr_feats = pop_class_distance_feats(config, model, method_variables,
                                        gaussian_process_pattern_keywords)
  assert len(curr_feats) == 1 and curr_feats[0].shape[0] == 1
  feat_shape = curr_feats[0].shape[1:]
  assert len(feat_shape) == 1
  print("(gaussian_processes_pre) feat_shape %s" % feat_shape)
  method_variables["feat_len"] = feat_shape[0]

  best_gp = None
  best_acc = -np.inf
  best_train_metrics = None
  best_val_metrics = None
  best_gp_scale = None
  for gp_scale in config.gp_scales:
    gp, train_metrics = train_gp(config, gp_scale, method_variables, model, train_loader)
    val_metrics = eval_gp(config, model, gp, val_loader, method_variables)

    acc = val_metrics["val/accuracy"].result().numpy()
    print("(gaussian_processes_pre) gp scale %s, val acc %s" % (gp_scale, acc))
    stdout.flush()

    if acc > best_acc:
      best_acc = acc
      best_gp = gp
      best_train_metrics = train_metrics
      best_val_metrics = val_metrics
      best_gp_scale = gp_scale

  print("(gaussian_processes_pre) best gp scale %s" % best_gp_scale)
  method_variables["gp"] = best_gp
  method_variables["train_metrics"] = best_train_metrics
  method_variables["val_metrics"] = best_val_metrics
  method_variables["gp_scale"] = best_gp_scale

  return model, method_variables


def gaussian_process_metric(config, method_variables, model, pt_imgs, pt_targets):
  num_samples = pt_imgs.shape[0]

  with torch.no_grad():
    preds = model(pt_imgs)

  curr_feats = pop_class_distance_feats(config, model, method_variables,
                                        gaussian_process_pattern_keywords)
  assert len(curr_feats) == 1
  assert (curr_feats[0].shape == (pt_imgs.shape[0], method_variables["feat_len"]))

  curr_feats = tf.convert_to_tensor(curr_feats[0].cpu().numpy())  # float vectors

  gp = method_variables["gp"]
  logits = gp(curr_feats, training=False)
  assert isinstance(logits, (list, tuple))
  logits, covmat = logits

  assert covmat.shape == (num_samples, num_samples)
  stddev = tf.sqrt(tf.linalg.diag_part(covmat)) # covmat is batch sz, batch sz
  assert stddev.shape == (num_samples,)
  unreliability = torch.tensor(stddev.numpy(), device=device(config.cuda))

  preds_flat = preds.argmax(dim=1)
  correct = preds_flat.eq(pt_targets)

  return unreliability, correct


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def train_gp(config, gp_scale, method_variables, model, train_loader):
  inputs = tf.keras.layers.Input(shape=(method_variables["feat_len"],))  # flat vector
  gp = make_gp_layer(inputs, classes_per_dataset[config.data],
                     config.gp_hidden_dim, gp_scale, gp_bias,
                     gp_input_normalization, gp_random_feature_type,
                     gp_cov_discount_factor, gp_cov_ridge_penalty,
                     gp_output_imagenet_initializer)

  # train GP layer using model feats
  base_lr = gp_base_learning_rate * config.batch_size / 256
  decay_epochs = [
    (gp_train_epochs * 30) // 90,
    (gp_train_epochs * 60) // 90,
    (gp_train_epochs * 80) // 90,
  ]
  learning_rate = WarmUpPiecewiseConstantSchedule(
    steps_per_epoch=len(train_loader),
    base_learning_rate=base_lr,
    decay_ratio=0.1,
    decay_epochs=decay_epochs,
    warmup_epochs=5)
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                      momentum=1.0 - gp_one_minus_momentum,
                                      nesterov=True)

  metrics = {
    "train/negative_log_likelihood": tf.keras.metrics.Mean(),
    "train/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
    "train/loss": tf.keras.metrics.Mean(),
  }

  for e in range(gp_train_epochs):
    gp.layers[-1].reset_covariance_matrix()

    for batch_i, (imgs, targets) in enumerate(train_loader):
      if batch_i % max(1, len(train_loader) // 100) == 0:
        print(
          "(class_distance_pre) training GP: e %d, batch %d / %d: nll %s, acc %s, loss %s, %s" % (
            e, batch_i, len(train_loader),
            metrics["train/negative_log_likelihood"].result(), metrics["train/accuracy"].result(),
            metrics["train/loss"].result(),
            datetime.now()))
        stdout.flush()

      pt_imgs, pt_targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
      with torch.no_grad():
        _ = model(pt_imgs)
      curr_feats = pop_class_distance_feats(config, model, method_variables,
                                            gaussian_process_pattern_keywords)
      assert len(curr_feats) == 1
      assert (curr_feats[0].shape == (pt_imgs.shape[0], method_variables["feat_len"]))

      curr_feats = tf.convert_to_tensor(curr_feats[0].cpu().numpy())  # float vectors
      targets = tf.convert_to_tensor(targets.cpu().numpy())  # longs, should/could be uint8..

      with tf.GradientTape() as tape:
        logits = gp(curr_feats, training=True)
        if isinstance(logits, (list, tuple)):
          logits, _ = logits

        negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True))

        filtered_variables = []
        for var in gp.trainable_variables:
          # Apply l2 on the weights. This excludes BN parameters and biases, but
          # pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = gp_l2_weight * 2 * tf.nn.l2_loss(
          tf.concat(filtered_variables, axis=0))

        loss = negative_log_likelihood + l2_loss

      grads = tape.gradient(loss, gp.trainable_variables)
      optimizer.apply_gradients(zip(grads, gp.trainable_variables))

      # probs = tf.nn.softmax(logits)
      metrics["train/loss"].update_state(loss)
      metrics["train/negative_log_likelihood"].update_state(
        negative_log_likelihood)
      metrics["train/accuracy"].update_state(targets, logits)

  return gp, metrics


def eval_gp(config, model, gp, val_loader, method_variables):
  metrics = {
    "val/negative_log_likelihood": tf.keras.metrics.Mean(),
    "val/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
  }

  for batch_i, (imgs, targets) in enumerate(val_loader):
    if batch_i % max(1, len(val_loader) // 100) == 0:
      print(
        "(class_distance_pre) eval GP: batch %d / %d: nll %s, acc %s, %s" % (
          batch_i, len(val_loader),
          metrics["val/negative_log_likelihood"].result(), metrics["val/accuracy"].result(),
          datetime.now()))
      stdout.flush()

    pt_imgs, pt_targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))
    with torch.no_grad():
      _ = model(pt_imgs)
    curr_feats = pop_class_distance_feats(config, model, method_variables,
                                          gaussian_process_pattern_keywords)
    assert len(curr_feats) == 1
    assert (curr_feats[0].shape == (pt_imgs.shape[0], method_variables["feat_len"]))

    curr_feats = tf.convert_to_tensor(curr_feats[0].cpu().numpy())
    targets = tf.convert_to_tensor(targets.cpu().numpy())

    logits = gp(curr_feats, training=False)
    if isinstance(logits, (list, tuple)):
      logits, _ = logits

    negative_log_likelihood = tf.reduce_mean(
      tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True))

    # probs = tf.nn.softmax(logits)
    metrics["val/negative_log_likelihood"].update_state(
      negative_log_likelihood)
    metrics["val/accuracy"].update_state(targets, logits)

  return metrics


def make_gp_layer(input_x, num_classes,
                  gp_hidden_dim, gp_scale, gp_bias,
                  gp_input_normalization, gp_random_feature_type,
                  gp_cov_discount_factor, gp_cov_ridge_penalty,
                  gp_output_imagenet_initializer):
  """
  Args:
    x: x
    num_classes: Number of output classes.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.
    gp_hidden_dim: The hidden dimension of the GP layer, which corresponds to
      the number of random features used for the approximation.
    gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
    gp_bias: The bias term for GP layer.
    gp_input_normalization: Whether to normalize the input using LayerNorm for
      GP layer. This is similar to automatic relevance determination (ARD) in
      the classic GP learning.
    gp_random_feature_type: The type of random feature to use for
      `RandomFeatureGaussianProcess`.
    gp_cov_discount_factor: The discount factor to compute the moving average of
      precision matrix.
    gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.
    gp_output_imagenet_initializer: Whether to initialize GP output layer using
      Gaussian with small standard deviation (sd=0.01).
  Returns:
    tf.keras.Model. Will return prediction and cov (uncertainty) in inference pass.
  """

  gp_output_initializer = None
  if gp_output_imagenet_initializer:
    # Use the same initializer as dense
    gp_output_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)

  output_layer = functools.partial(
    ed.layers.RandomFeatureGaussianProcess,
    num_inducing=gp_hidden_dim,
    gp_kernel_scale=gp_scale,
    gp_output_bias=gp_bias,
    normalize_input=gp_input_normalization,
    gp_cov_momentum=gp_cov_discount_factor,
    gp_cov_ridge_penalty=gp_cov_ridge_penalty,
    scale_random_features=False,
    use_custom_random_features=True,
    custom_random_features_initializer=make_random_feature_initializer(
      gp_random_feature_type),
    kernel_initializer=gp_output_initializer,
    return_gp_cov=True)

  outputs = output_layer(num_classes)(input_x)
  return tf.keras.Model(inputs=input_x, outputs=outputs, name='gp_model')


def make_random_feature_initializer(random_feature_type):
  # Use stddev=0.05 to replicate the default behavior of
  # tf.keras.initializer.RandomNormal.
  if random_feature_type == 'orf':
    return ed.initializers.OrthogonalRandomFeatures(stddev=0.05)
  elif random_feature_type == 'rff':
    return tf.keras.initializers.RandomNormal(stddev=0.05)
  else:
    return random_feature_type



class WarmUpPiecewiseConstantSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule.
  It starts with a linear warmup to the initial learning rate over
  `warmup_epochs`. This is found to be helpful for large batch size training
  (Goyal et al., 2018). The learning rate's value then uses the initial
  learning rate, and decays by a multiplier at the start of each epoch in
  `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
  """

  def __init__(self,
               steps_per_epoch,
               base_learning_rate,
               decay_ratio,
               decay_epochs,
               warmup_epochs):
    super(WarmUpPiecewiseConstantSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.base_learning_rate = base_learning_rate
    self.decay_ratio = decay_ratio
    self.decay_epochs = decay_epochs
    self.warmup_epochs = warmup_epochs

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    learning_rate = self.base_learning_rate
    if self.warmup_epochs >= 1:
      learning_rate *= lr_epoch / self.warmup_epochs
    decay_epochs = [self.warmup_epochs] + self.decay_epochs
    for index, start_epoch in enumerate(decay_epochs):
      learning_rate = tf.where(
          lr_epoch >= start_epoch,
          self.base_learning_rate * self.decay_ratio**index,
          learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'base_learning_rate': self.base_learning_rate,
    }