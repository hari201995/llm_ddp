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

## Training Launcher (Lambda Labs / RunPod)

`launch.py` automates the full training workflow — from provisioning a GPU instance to downloading weights — without touching either provider's website.

Every command takes `--provider {lambda,runpod}` (default `lambda`), so all the Lambda Labs instructions below work unchanged. See [RunPod](#runpod) further down for the second provider.

### Prerequisites

```sh
pip install paramiko requests
export LAMBDA_API_KEY=your_key_here      # from cloud.lambdalabs.com/api-keys
export WANDB_API_KEY=your_key_here       # from wandb.ai account settings
```

Make sure your GitHub SSH key is configured (`ssh -T git@github.com` should say "successfully authenticated").

### Commands

#### 1. Check GPU availability and pricing (do this first)

```sh
python launch.py gpus --gpu-type a100
```

For Lambda, this is the one command you need — price and which regions have
capacity, together in one table. Omit `--gpu-type` to list everything.

**RunPod's equivalent one-stop command is different** (see [RunPod](#runpod)
below for why) — use `datacenters --gpu-type a100 --provider runpod` instead of
`gpus` there.

To narrow Lambda down to a single region, add `--datacenter-id` (substring match):
```sh
python launch.py gpus --gpu-type a100 --datacenter-id us-east-1
```

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

#### 3. Launch a training run

```sh
python launch.py train --expt-name run1 --gpu-type a100 --gpu-count 1 --max-hours 8
python launch.py train --expt-name single_small --config configs/small.toml --gpu-type a100 --gpu-count 1 --max-hours 3
```

- Shows estimated cost — prompts for confirmation before launching
- Spins up GPU instance, clones the `devel` branch via SSH agent forwarding, installs deps
- Auto-patches `world_size` in the config to match `--gpu-count`
- Injects `WANDB_API_KEY` into the training environment
- Streams training logs live to your terminal
- Downloads weights and logs when done, terminates the instance
- If training crashes, still downloads whatever artifacts exist before terminating
- Saves instance ID locally so you can reconnect if your Mac disconnects
- Use `--keep-alive` to skip auto-termination (useful for debugging failures)

```
Option            Default                    Description
────────────────────────────────────────────────────────────────────────────────
--expt-name       (required)                 Experiment name
--config          configs/lm_config.toml     Config file path inside the repo
--gpu-type        a100                       GPU name — partial match on description
--gpu-count       1                          Number of GPUs (1, 2, 4, 8)
--max-hours       10.0                       Used for cost estimate only, does not stop training
--output-dir      ./artifacts_remote         Local dir for downloaded weights and logs
--keep-alive      false                      Keep instance running after training ends or fails
```

#### 4. Reconnect to an existing run

If your Mac disconnects mid-training:

```sh
python launch.py attach
```

Reconnects to the running instance, optionally waits for training to finish, downloads artifacts, and terminates.

#### 5. Terminate instance manually

If you used `--keep-alive` or need to force-terminate:

```sh
python launch.py terminate
```

Terminates the active instance saved in `~/.llm_ddp_lambda.json`.

### Typical workflow

```sh
# First time only
python launch.py gpus --gpu-type a100
# → create filesystem in Lambda dashboard, then:
python launch.py setup --filesystem-name LM336

# Every training run
python launch.py gpus --gpu-type a100               # check availability and pricing
python launch.py train --expt-name run1 \
    --gpu-type a100 --gpu-count 1 --max-hours 8

# Debugging a failed run
python launch.py train --expt-name run1 --keep-alive ...
ssh ubuntu@<ip-shown-in-output>
python launch.py terminate                          # when done
```

---

## RunPod

Same commands as above, with `--provider runpod` added. State is stored separately
from Lambda's (`runpod_*`-prefixed keys in the same `~/.llm_ddp_lambda.json`), so
you can use both providers interchangeably without them interfering with each other.

### Prerequisites

```sh
pip install paramiko requests
export RUNPOD_API_KEY=your_key_here      # from runpod.io/console/user/settings
export WANDB_API_KEY=your_key_here
```

No SSH key registration step needed — your public key (`~/.ssh/id_ed25519.pub`) is
injected into every pod automatically via RunPod's standard image startup script.

### Commands

#### 1. Check GPU availability and pricing (do this first)

```sh
python launch.py datacenters --gpu-type a100 --provider runpod
```

This is RunPod's one-stop command — it loops every datacenter (one API call
each, so it takes a few seconds) and prints price + live stock + how many you
can request, per datacenter, in one table. Pick a datacenter ID with stock from
the output; you'll need it for `setup` below.

(`gpus --gpu-type a100 --provider runpod` also exists, but only shows global
pricing — no location/stock — since RunPod's pricing API and per-datacenter
stock API are separate; `--datacenter-id` narrows `gpus` to one datacenter at a
time if you already know which one you want.)

#### 2. One-time setup — create network volume and upload data

```sh
python launch.py setup --provider runpod --datacenter-id <id-from-step-1>
```

- Creates the network volume (or reuses one with the same `--volume-name` if it already exists)
- Spins up a temporary pod attached to the volume, `rsync`s your local training data onto it, terminates the pod
- Saves the volume ID and datacenter to `~/.llm_ddp_lambda.json`

Only run this once. Data lives on the network volume permanently, mounted at
`/workspace` on every future pod.

**Alternative — manual data transfer without this launcher:** RunPod also offers
[`runpodctl send/receive`](https://docs.runpod.io/pods/storage/transfer-files), a
relay-based one-time-code transfer that needs no open ports — useful for ad hoc
single-file pushes outside the network-volume flow entirely:
```sh
local$  runpodctl send myfile.tar
pod$    runpodctl receive <code-printed-by-send>
```

#### 3. Launch a training run

```sh
python launch.py train --provider runpod --expt-name run1 --gpu-type a100 --gpu-count 2 --max-hours 8
```

Same behavior as the Lambda flow (cost confirmation, repo clone, dependency
install, world_size patching, live log streaming, artifact download, auto-terminate,
`--keep-alive` for debugging) — just provisioned on RunPod instead.

#### 4. Reconnect / terminate

```sh
python launch.py attach --provider runpod
python launch.py terminate --provider runpod
```

---

## Verda

Verda (formerly DataCrunch.io, founded 2018, NVIDIA Preferred Partner) is a useful
fallback if Lambda Labs and RunPod both lack availability. EU-based datacenters
(Finland, Iceland). A100 at $1.79/hr — cheaper than Lambda ($1.99), slightly more
than RunPod PCIe ($1.39). State stored under `verda_*` keys in `~/.llm_ddp_lambda.json`.

Architecturally closer to Lambda than RunPod: bare VM (not container), direct public
IP, port 22, root user, NVMe block volumes for persistent data.

### Prerequisites

```sh
pip install paramiko requests
export VERDA_CLIENT_ID=your_client_id         # from verda.com → API settings
export VERDA_CLIENT_SECRET=your_client_secret  # (OAuth2, not a simple API key)
export WANDB_API_KEY=your_key_here
```

### Commands

#### 1. Check GPU availability and pricing

```sh
python launch.py gpus --gpu-type a100 --provider verda
```

Shows price + which specific datacenters currently have it — all in one call
(Verda's availability API is more like Lambda's: one query covers all locations).

#### 2. One-time setup — create a persistent volume and upload data

```sh
python launch.py datacenters --provider verda          # get location codes
python launch.py setup --provider verda --location-code FIN-01
```

Creates an NVMe volume, uploads your local training data onto it via a temporary
instance, then terminates that instance. All future training runs mount the same
volume at `/data`.

#### 3. Launch a training run

```sh
python launch.py train --provider verda --expt-name run1 --gpu-type a100 --max-hours 8
```

Same behavior as Lambda/RunPod — cost confirmation, repo clone, dep install,
config patch, live log streaming, artifact download, auto-terminate.

#### 4. Reconnect / terminate

```sh
python launch.py attach --provider verda
python launch.py terminate --provider verda
```

---

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
