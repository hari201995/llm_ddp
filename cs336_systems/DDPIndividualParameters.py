import torch.nn as nn
import torch.distributed as dist
import torch


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        """
        Returns a torch.nn.Module container that handles
        parameter broadcasting and gradient synchronization for
        distributed data parallel training.
        """
        self.module = module
        self.rank = dist.get_rank()
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def param_grad_sync(self):
        """
        Synchronizes the gradients of the parameters across all ranks by
        performing an all-reduce operation on the gradients of each parameter.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()

    def batched_grad_sync(self):
        """
        Synchronizes the gradients of the parameters across all ranks by
        performing an all-reduce operation on the gradients of each parameter in batches.
        """
        # synchronize gradients in batches (e.g., to overlap communication with computation).

        grads = [
            param.grad.data for param in self.module.parameters() if param.requires_grad
        ]
        # flatten the list of gradients into a single tensor
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= dist.get_world_size()
        # unflatten the gradients back into the original shapes
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        # copy the synchronized gradients back to the original parameters
        idx = 0

        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.data = unflat_grads[idx]
                idx += 1

    def forward(self, x):
        """
        Forwards the input through the underlying module.
        """
        return self.module(x)
