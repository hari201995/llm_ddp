from typing import Any, Dict

import torch.nn as nn
import torch
import torch.distributed as dist
import math

"""
 What you're building:
  A wrapper optimizer where each rank only optimizes a subset (shard) of the model's parameters.

  How it works step by step:

  1. __init__:
    - Take all params, split them across ranks (rank 0 gets first chunk, rank 1 gets second 
    chunk, etc.)
    - Each rank creates a local optimizer_cls instance only on its shard of params
    - Call super().__init__(all_params, kwargs) so the base class knows about all params
  2. step():
    - Each rank runs its local optimizer step on its shard only — so only that shard's params
      get updated
    - Then all_gather so every rank gets the fully updated params from all other ranks
    - Now all ranks have the same complete updated model
  3. zero_grad():
    - Inherited from base class, works on all params — no need to override

 param_groups and state:
  - param_groups is how PyTorch tracks which params belong to this optimizer. The base class
    populates it from the params you pass to super().__init__
  - state stores AdamW's running averages (exp_avg, exp_avg_sq) — each rank only has state for
    its own shard since it only steps on those

 The fields in a param group dict are:
    - 'params' — list of parameter tensors
    - 'lr' — learning rate
    - 'weight_decay' — weight decay
    - 'betas' — for Adam-style optimizers
    - 'eps' — for Adam-style optimizers
    - 'maximize' — whether to maximize instead of minimize
"""


class OptimizerSharding(torch.optim.Optimizer):
    """
    Performs optimizer level sharding for DDP
    """

    def __init__(self, optim_params, optimizer_cls, **kwargs) -> None:
        self.params = list(optim_params)

        # store for local optimizer initialization
        self.module = optimizer_cls
        self.default_dict = kwargs

        # call the super for main optimizer for zero gradding
        super().__init__(params=self.params, defaults=kwargs)

    def step(self, **kwargs):
        # calls the optimizer
        self.local_opt.step()
        all_trainable_params = [p for p in self.params if p.requires_grad == True]
        for i, param in enumerate(all_trainable_params):
            owner_rank = i // self.per_rank
            dist.broadcast(param.data, src=owner_rank)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Python's method resolution order ensures when super.__init__ internally calls the
        add param group, it is overwritten with this fn.
        """
        all_params = param_group["params"]
        # Based on rank divide.
        this_rank = dist.get_rank()
        self.per_rank = math.ceil(len(param_group["params"]) / dist.get_world_size())
        stop = self.per_rank * (this_rank + 1)
        if stop < len(all_params):
            this_rank_param = [
                p
                for p in all_params[this_rank * (self.per_rank) : stop]
                if p.requires_grad == True
            ]
        else:
            this_rank_param = [
                p
                for p in all_params[this_rank * (self.per_rank) : len(all_params)]
                if p.requires_grad == True
            ]
        # create local optimizer
        self.local_opt = self.module(params=this_rank_param, **self.default_dict)
        # zero grading
        super().add_param_group(param_group)
