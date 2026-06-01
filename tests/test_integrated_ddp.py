from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import numpy

from .common import ToyModel, _cleanup_process_group, _setup_process_group
from cs336_systems import DDPAsyncParameter, OptimizerSharding


def _run_integrated(rank, world_size, results_queue):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)

    # Both ranks start with same weights (same seed)
    model = ToyModel().to(device)
    ddp_model = DDPAsyncParameter.DDPAsyncParameter(model)
    dist.barrier()

    optimizer = OptimizerSharding.OptimizerSharding(
        optim_params=model.parameters(),
        optimizer_cls=torch.optim.AdamW,
        lr=0.01,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    loss_fn = nn.MSELoss()
    losses = []

    for step in range(10):
        optimizer.zero_grad()

        # Each rank uses different data to simulate DDP
        torch.manual_seed(step * world_size + rank)
        x = torch.rand(8, 10, device=device)
        y = torch.rand(8, 5, device=device)

        out = ddp_model(x)
        loss = loss_fn(out, y)
        loss.backward()

        ddp_model.gradient_synchronization()
        optimizer.step()

        if rank == 0:
            losses.append(loss.item())

        # Verify all ranks have identical weights after each step
        for param in model.parameters():
            gathered = [torch.zeros_like(param.data) for _ in range(world_size)]
            dist.all_gather(gathered, param.data)
            for g in gathered:
                assert torch.allclose(g, gathered[0], atol=1e-6), \
                    f"Step {step}: param mismatch across ranks"

    if rank == 0:
        results_queue.put(losses)

    _cleanup_process_group()


def test_integrated_ddp_optimizer_sharding():
    world_size = 2
    results_queue = mp.Queue()

    mp.spawn(
        _run_integrated,
        args=(world_size, results_queue),
        nprocs=world_size,
        join=True,
    )

    losses = results_queue.get()

    # Loss should decrease over 10 steps
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"

    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (passed)")
