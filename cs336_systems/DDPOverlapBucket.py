import torch.nn as nn
import torch
import torch.distributed as dist
import math
from collections import defaultdict


class DDPOverlapBucket(nn.Module):
    """
    Performs parameter bucketting and async gradient all reduce for each bucket
    """

    def __init__(self, module: nn.Module, bucket_size: float):
        super().__init__()
        self.module = module
        # broadcast all params from rank 0
        for params in self.module.parameters():
            dist.broadcast(params.data, src=0)
        # async handler
        self.handles = []

        # Bucketization of paams
        self.curr_total = 0
        curr_key = 0
        self.final_bucket = defaultdict(list)

        # add backward comp hook
        for param in reversed(list(self.module.parameters())):
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    lambda p: self.bucket_params(p)
                )

                # create buckets
                p_size = param.numel() * param.element_size() / (1024**2)  # MB
                self.curr_total += p_size
                if self.curr_total <= bucket_size:
                    self.final_bucket[curr_key].append(param)
                else:
                    # put the prev bucket in final bucket
                    curr_key += 1
                    self.final_bucket[curr_key].append(param)

        # reverse dict
        self.params_to_bucket = {}
        for bucket, param in self.final_bucket.items():
            for p in param:
                self.params_to_bucket[p] = bucket

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def bucket_params(self, p):
        """
        Called at the start of training to bucket parameters.
        """
        # set grad ready flag
        p.grad_ready = True

        # Find out the key
        bucket_id = self.params_to_bucket[p]

        # check all params in the bucket are grad ready
        grad_flag = [
            getattr(x, "grad_ready", False) for x in self.final_bucket[bucket_id]
        ]

        if all(grad_flag):
            self.gradient_all_reduce(self.final_bucket[bucket_id])

    def gradient_all_reduce(self, p):
        # Gloo doesnt do Reduceop.AVG. So, we need to store reference to the parameters
        # and divide with world size after the entire all reduce is done.
        # flatten the list of gradients into a single tensor
        flat_grads = torch._utils._flatten_dense_tensors([x.grad.data for x in p])
        handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, flat_grads, p))

    def gradient_synchronization(self):
        for handle, flat_grads, p in self.handles:
            handle.wait()
            flat_grads /= dist.get_world_size()
            unflat_grads = torch._utils._unflatten_dense_tensors(
                flat_grads, [x.grad.data for x in p]
            )
            for new_grad, param in zip(unflat_grads, p):
                param.grad.data = new_grad
        self.handles.clear()

    def grad_sync_reset(self):
        """
        Reset all the grad ready flags
        """
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad_ready = False  # type: ignore
