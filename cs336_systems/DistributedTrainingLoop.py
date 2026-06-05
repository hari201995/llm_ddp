import tomllib
import json
import importlib

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import argparse
import wandb
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp

# Own Data types
import cs336_basics
import logging_setup
import DDPAsyncParameter
import OptimizerSharding
import utilities


def dump_config(cfg_dict, log):
    log.info("CONFIG:\n" + json.dumps(cfg_dict, indent=2, sort_keys=True))


def get_args():
    """
    Docstring for get_args
    """
    p = argparse.ArgumentParser(description="Short description of your script.")
    # positional
    p.add_argument("config_name", help="Path to config file")
    p.add_argument("expt_name", help="expt name for tag")
    return p.parse_args()


def logging_fn(rank, cfg, expt_name):
    """
    Sets up the logging mechanism.

    :param rank: Rank to which this logging is happening
    :param cfg: Dumps the entire model cfg
    :param expt_name : Name of the ablation
    """
    log, logfile = logging_setup.setup_logger(run_name=expt_name)
    ##########################
    # Training loop call
    ##########################
    log.info("Starting training")
    ##########################
    # dump config file in log
    ##########################
    if rank == 0:
        dump_config(cfg, log)
    return log


def get_data_for_rank(rank, file_ids, len_weight, world_size, steps_per_epoch):
    """
    get_data_for_rank gets the BxD data for this total training step based on the pdf.
    This is done with an skip offset so that all the devices use different part of the same shard.

    :param rank: Rank of the device.
    :param file_ids: List of all data shards available for training.
    :param len_weight: pdf of the different shards
    :param world_size : Total number of devices
    :param steps_per_epoch : Total number of steps per epoch,
    """
    choice_shards = np.random.choice(
        np.asarray(file_ids), size=world_size * steps_per_epoch, p=len_weight
    )
    choice_shards = choice_shards[rank::world_size]  # this is a list of elements shards
    return choice_shards


def setup(rank, world_size, backend_type):
    """
    Docstring for setup

    :param rank: raank of the device
    :param world_size: World size for distributed computing
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    dist.init_process_group(backend_type, rank=rank, world_size=world_size)


def distributed_training_loop(rank, expt_name, world_size, cfg):
    """
    Runs DDP training with optimizer sharding

    :param rank: rank of the device
    :param expt_name: Name of the experiment/model variant
    :param world_size: Total number of spawned proc
    :param cfg: Parameters of the experiment
    """
    ##########################
    # Variable initializations
    ##########################

    # To tokenize data
    total_epochs = cfg["training"]["params"]["epochs"]
    total_steps_per_epoch = cfg["training"]["params"]["steps_per_epoch"]

    # model  & training params
    model_name = cfg["model"]["name"]
    shards_used = torch.zeros(total_steps_per_epoch, device="cpu", dtype=torch.long)
    folder_path = cfg["data"]["params"]["path"]

    # Loss array
    ce_loss = []
    ema_ce_loss = []

    # model  & training params
    B = cfg["training"]["params"]["batch_size"]
    T = cfg["model"]["params"]["context_length"]
    theta = cfg["training"]["params"]["theta"]

    # LR params
    alpha_max = cfg["training"]["params"]["alpha_max"]
    alpha_min = cfg["training"]["params"]["alpha_min"]
    warmup_iter = cfg["training"]["params"]["warmup_iter"]
    num_cosine_iter = cfg["training"]["params"]["cooldown_iter"]

    # checkpoint variables
    checkpoint_every = cfg["training"]["params"]["checkpoint_every"]
    log_every_steps = 1

    # Validation params
    validate_every_steps = cfg["validation"]["params"]["validate_every_steps"]
    val_loss = []
    val_steps = []

    ##########################
    # Create Log file object
    ##########################
    log = logging_fn(rank, cfg, expt_name)

    ##########################
    # Data preparation
    ##########################
    shard_paths, val_shard_paths, len_weight, file_ids = utilities.data_prep(cfg)

    ##########################
    # Create model object
    ##########################
    LM = utilities.create_obj(cfg, field="model")

    ##########################
    # set the device properly
    ##########################
    device, autocast_dtype, amp_flag = utilities.rationalize_device(rank, LM)
    log.info(f"running the code in {device} ")

    ##########################
    # Initialize the LM setup
    ##########################
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        setup(rank=rank, world_size=world_size, backend_type="nccl")
    else:
        # Mac doesnt support NCCL
        setup(rank=rank, world_size=world_size, backend_type="gloo")

    # Move LM to device
    LM = LM.to(device)

    # call the DDP Class
    LM_DDP = DDPAsyncParameter.DDPAsyncParameter(LM)
    dist.barrier()

    ##########################
    # Create the optimizer
    ##########################
    O_Shard = OptimizerSharding.OptimizerSharding(
        optim_params=LM.parameters(),
        optimizer_cls=cs336_basics.adamW.AdamW,
        lr=cfg["optimizer"]["params"]["lr"],
        weight_decay=cfg["optimizer"]["params"]["weight_decay"],
        betas=cfg["optimizer"]["params"]["betas"],
        eps=cfg["optimizer"]["params"]["eps"],
    )

    ##########################
    # weights & biases for training
    ##########################
    if rank == 0 and False:
        log.info("Initializing wandb")
        run = wandb.init(
            project="llm_train_project",
            mode="offline",
            config={
                "learning_rate": alpha_max,
                "architecture": model_name,
                "dataset": folder_path,
                "epochs": total_epochs,
            },
        )

    ##########################
    # Run the training loop
    ##########################

    # Tracking variables.
    current_epoch = 0
    running_counter = 0  # learning rate counter
    stop_train = False  # Training flag
    # early exit
    stop_tensor = torch.tensor([0], device=device, dtype=torch.int)

    # training flag for pytorch so that grad is available for sure
    LM_DDP.module.train()

    # Training loop
    while current_epoch < total_epochs and not (stop_train):
        # generate the data for this epoch
        shards_used.zero_()
        # different shards to memmap every epoch
        choice_shards = get_data_for_rank(
            rank, file_ids, len_weight, world_size, total_steps_per_epoch
        )

        # token positions — fixed for all steps
        token_pos = torch.arange(T, device=device)

        # Batchwise loop
        for b_id in range(total_steps_per_epoch):
            # get the shard for this b_id
            shards_used[b_id] = int(choice_shards[b_id])
            # load this shard
            shard_file = shard_paths[choice_shards[b_id]]
            tokenized_data = np.memmap(shard_file, dtype=np.uint16, mode="r")

            # memmap notes inside data loader. loads the data.
            try:
                x, target = cs336_basics.data_loader.data_loader(
                    tokenized_data,
                    batch_size=B,
                    context_length=T,
                    device_type=device.type,
                )
            finally:
                if hasattr(tokenized_data, "_mmap"):
                    tokenized_data._mmap.close()

            # get the learning rate
            current_lr = cs336_basics.learning_rate_schedule.learning_rate_schedule(
                running_counter,
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                warmup_iter=warmup_iter,
                num_cosine_iter=num_cosine_iter,
            )

            # update the learning rate of optimizer
            for i in range(len(O_Shard.local_opt.param_groups)):
                O_Shard.local_opt.param_groups[i]["lr"] = current_lr

            # make O zero grad
            O_Shard.zero_grad()

            ##########################
            # forward computation
            ##########################
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=amp_flag
            ):
                # Run the language model
                y = LM_DDP.module.tranform_lm_model(
                    x,
                    rope_theta=theta,
                    token_positions=token_pos[: x.size(1)],
                    max_seq_len=x.size(1),
                )
                loss = cs336_basics.cross_entropy_loss.cross_entropy_loss(y, target)

            ##########################
            # Gradient computation
            ##########################
            loss.backward()

            # gradient syncronization across all machines
            LM_DDP.gradient_synchronization()

            # Run gradient clipping
            cs336_basics.gradient_clip.gradient_clipping(
                list(LM.parameters()), max_l2_norm=1
            )

            ##########################
            # Optimiser step
            ##########################
            O_Shard.step()

            # update running counter
            running_counter += 1

            ##########################
            # Validation & Checkpointing model and loss
            ##########################

            if rank == 0:
                # Validate using only rank 0
                if running_counter % validate_every_steps == 0:
                    # Validation utilities
                    cal_loss_val = utilities.validate(
                        rank,
                        LM=LM_DDP,
                        val_data=val_shard_paths,
                        cfg=cfg,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        amp_flag=amp_flag,
                    )
                    val_loss.append(cal_loss_val)
                    val_steps.append(running_counter)

            # restore train mode on all ranks after validation step
            if running_counter % validate_every_steps == 0:
                LM_DDP.module.train()

                # Model checkpointing ( Use only rank 0)
                if running_counter % checkpoint_every == 0:
                    pkg = {
                        "model_state": LM.state_dict(),
                        "optimizer_state": O_Shard.state_dict(),
                        "iter": running_counter,
                        "config": cfg,
                    }
                    latest_checkpoint = (
                        f"artifacts/checkpoint/iter_{running_counter}.pt"
                    )
                    os.makedirs("artifacts/checkpoint", exist_ok=True)
                    torch.save(pkg, latest_checkpoint)

                # Computing and Logging CE loss

                if running_counter % log_every_steps == 0:
                    # add the loss in ce_loss for plot
                    ce_loss.append(loss.item())
                    if ema_ce_loss != []:
                        curr_ema = 0.8 * ema_ce_loss[-1] + 0.2 * loss.item()
                        ema_ce_loss.append(curr_ema)
                    else:
                        ema_ce_loss.append(loss.item())

                    last_val = val_loss[-1] if val_loss else float("nan")

                    log.info(f"step={running_counter} loss={loss.item():.4f}\
                            lr={current_lr:.3e} val_loss={last_val}")

                    pass
                    if loss.item() < 1.8:
                        stop_tensor[0] = 1

            if running_counter % validate_every_steps == 0:
                # stall others until validation is done
                dist.barrier()

            # Make all other ranks wait until the rank 0 finishes validation
            dist.broadcast(stop_tensor, src=0)

            # Early finish
            if stop_tensor.item() == 1:
                stop_train = True
                break

        # update epoch counter
        current_epoch += 1
        # save the choice of shards used in training to see diversity
        if rank == 0:
            torch.save(
                shards_used,
                f"artifacts/logs/{expt_name}_choice_shards_epoch_{current_epoch}.pt",
            )

    ##########################
    # Plotting and saving the model
    ##########################

    if rank == 0:
        if "pkg" not in locals():
            pkg = {
                "model_state": LM.state_dict(),
                "optimizer_state": O_Shard.state_dict(),
                "iter": running_counter,
                "config": cfg,
            }

        # save the model
        torch.save(pkg, f"artifacts/checkpoint/{expt_name}.pt")
        # plot the loss
        steps_train = list(range(len(ce_loss)))
        plt_title = expt_name + "-Training loss Vs Validation loss"

        plt.figure()
        plt.plot(val_steps, val_loss, label="validation loss")
        plt.plot(steps_train, ema_ce_loss, label="EMA train loss")
        plt.xlabel("step")
        plt.ylabel("CE loss")
        plt.title(plt_title)

        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.savefig("artifacts/logs/loss.png", dpi=600)
        # Also print final values
        print(f"Final loss: {ce_loss[-1]:.4f}")
        log.info("Training complete")
        pass


def main():
    # Example command :
    arg = get_args()

    os.makedirs("artifacts/checkpoint", exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)

    # Open config
    with open(arg.config_name, "rb") as f:
        cfg = tomllib.load(f)

    ##########################
    # Variable initializations
    ##########################
    # DDP params
    world_size = cfg["training"]["params"][
        "world_size"
    ]  # Number of GPUs to use across nodes

    ##########################
    # Create process groups
    ##########################
    mp.spawn(  # type: ignore
        fn=distributed_training_loop,
        args=(arg.expt_name, world_size, cfg),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
