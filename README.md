# Predicting Generalization and Uncertainty by Shattering a Neural Network [[arXiv]](http://arxiv.org/abs/2106.08365)

This repository contains code for training classification models, running unreliability quantification experiments using subfunctions and other methods, and printing tables and figures.

System settings are defined in `scripts/global_constants.py`. You should change these according to your needs. Turn `PRINT_COMMANDS_ONLY` on to do a dry run first.

Experimental settings for main results in are defined in `scripts/slurm/cifar/constants.py`. You can change these to reduce the architectures, seeds, datasets and hyperparameters considered (default is to run everything).

Also you can edit `augment_command` in `util/general.py` to suit your infrastructure. Currently the commands use slurm or `export CUDA_VISIBLE_DEVICES`.

All commands are given below.

# Packages used
- Python 3.6.8
- PyTorch 1.6.0
- torchvision 0.7.0
- scikit-learn 0.21.3
- scipy 1.3.1

# Train classification models
`python -m scripts.slurm.train_models`

Or, download pre-trained models [here](https://www.robots.ox.ac.uk/~xuji/subfunctions/public_models.zip).

# Run unreliability experiments

Assumes you have trained the classification models.

**Subfunction error**

* First run pre-computations:
`python -m scripts.slurm.cifar.in_distribution.subfunctions_pre`

* In-distribution (uses pre-computations):
`python -m scripts.slurm.cifar.in_distribution.subfunctions`

* OOD (uses pre-computations):
`python -m scripts.slurm.cifar.out_of_distribution.subfunctions`

**Max response**

* In-distribution:
`python -m scripts.slurm.cifar.in_distribution.max_response`

* OOD:
`python -m scripts.slurm.cifar.out_of_distribution.max_response`

**Entropy**

* In-distribution:
`python -m scripts.slurm.cifar.in_distribution.entropy`

* OOD:
`python -m scripts.slurm.cifar.out_of_distribution.entropy`

**Margin**

* In-distribution:
`python -m scripts.slurm.cifar.in_distribution.margin`

* OOD:
`python -m scripts.slurm.cifar.out_of_distribution.margin`

**Class distance**

* In-distribution:
`python -m scripts.slurm.cifar.in_distribution.class_distance`

* OOD:
`python -m scripts.slurm.cifar.out_of_distribution.class_distance`

**Explicit density**

* In-distribution:
`python -m scripts.slurm.cifar.in_distribution.explicit_density`

* OOD:
`python -m scripts.slurm.cifar.out_of_distribution.explicit_density`

**GP**

* In-distribution:
`python -m scripts.slurm.cifar.in_distribution.gaussian_process`

* OOD:
`python -m scripts.slurm.cifar.out_of_distribution.gaussian_process`

# Print tables

Assumes you have run the experiments.

`python -m scripts.analysis.print_results`

# Render AUROC graphs

Assumes you have run the experiments.

`python -m scripts.analysis.plot_graphs`

# Qualitative figures

* Half-moons figures (fig. 1)
`python -m scripts.slurm.two_moons.two_moons_script`

* Sample confusion matrix (fig. 4)
`python -m scripts.slurm.cifar.run_qualitative`

* CIFAR100 boxplots (fig. 5)
`python -m scripts.slurm.cifar.run_label_distribution`

# Test the subfunctions code

`python -m scripts.slurm.test_code`
