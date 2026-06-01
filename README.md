# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Lambda Labs Training Launcher

`launch.py` automates the full training workflow on Lambda Labs — from provisioning a GPU instance to downloading weights — without touching the Lambda Labs website.

### Prerequisites

```sh
pip install paramiko requests
export LAMBDA_API_KEY=your_key_here      # from cloud.lambdalabs.com/api-keys
export WANDB_API_KEY=your_key_here       # from wandb.ai account settings
```

Make sure your GitHub SSH key is configured (`ssh -T git@github.com` should say "successfully authenticated").

### Commands

#### 1. Check GPU availability (do this first)

```sh
python launch.py datacenters --gpu-type a100
```

Shows which regions have your target GPU available and the hourly price.

> **Important:** Your persistent filesystem and GPU instance must be in the same region.

#### 2. One-time setup — create filesystem and upload data

First, create a persistent filesystem manually at the Lambda Labs dashboard (Storage section). Name it something memorable (e.g. `LM336`), pick the region from step 1, and set size to 200 GB+.

Then run:

```sh
python launch.py setup --filesystem-name LM336
```

- Verifies the filesystem exists via API
- Spins up a temporary instance, uploads your local training data to the filesystem, terminates the instance
- Saves filesystem name and region to `~/.llm_ddp_lambda.json` — all future runs use it automatically

Only run this once. Data lives on the filesystem permanently.

#### 3. Check GPU pricing

```sh
python launch.py gpus
```

Lists all available instance types with hourly price and regions that have capacity.

#### 4. Launch a training run

```sh
python launch.py train --expt-name run1 --gpu-type a100 --gpu-count 1 --max-hours 8
```

- Shows estimated cost — prompts for confirmation before launching
- Spins up GPU instance, clones the `devel` branch via SSH agent forwarding, installs deps
- Auto-patches `world_size` in the config to match `--gpu-count`
- Injects `WANDB_API_KEY` into the training environment
- Streams training logs live to your terminal
- Downloads weights and logs when done, terminates the instance
- If training crashes, still downloads whatever artifacts exist before terminating
- Saves instance ID locally so you can reconnect if your Mac disconnects

```
Option            Default                    Description
────────────────────────────────────────────────────────────────────────────────
--expt-name       (required)                 Experiment name
--config          configs/lm_config.toml     Config file path inside the repo
--gpu-type        a100                       GPU name — partial match on description
--gpu-count       1                          Number of GPUs (1, 2, 4, 8)
--max-hours       10.0                       Used for cost estimate only, does not stop training
--output-dir      ./artifacts_remote         Local dir for downloaded weights and logs
```

#### 5. Reconnect to an existing run

If your Mac disconnects mid-training:

```sh
python launch.py attach
```

Reconnects to the running instance, optionally waits for training to finish, downloads artifacts, and terminates.

### Typical workflow

```sh
# First time only
python launch.py datacenters --gpu-type a100
# → create filesystem in Lambda dashboard, then:
python launch.py setup --filesystem-name LM336

# Every training run
python launch.py gpus                               # check availability and pricing
python launch.py train --expt-name run1 \
    --gpu-type a100 --gpu-count 1 --max-hours 8
```

---

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
