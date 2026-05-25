import torch
import torch.nn as nn
import torch.distributed as dist


class DDPAsyncParameter(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # broadcast from rank 0
        for param in self.module.parameters():
            dist.broadcast(param, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    lambda p: self.gradient_all_reduce(p)
                )
        self.handles = []

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def gradient_all_reduce(self, p):
        if p.requires_grad:
            # Gloo doesnt do Reduceop.AVG. So, we need to store reference to the parameters
            # and divide with world size after the entire all reduce is done.
            handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, p))

    def gradient_synchronization(self):
        for handle, p in self.handles:
            handle.wait()
            p.grad.data /= dist.get_world_size()
        self.handles.clear()
