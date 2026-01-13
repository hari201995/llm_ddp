import tomllib
import importlib
import numpy as np
import os
import torch
import matplotlib
import argparse
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import json

import cs336_basics
import logging_setup
import time


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
    Docstring for logging_fn

    :param rank: Description
    :param cfg: Description
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


def rationalize_device(rank, LM):
    """
    Docstring for rationalize_device

    :param rank: Description
    :param LM: Description
    """
    if LM.device.type == "cuda":
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = (
            True  # TF32 for any remaining FP32 matmuls
        )
        torch.backends.cudnn.allow_tf32 = True
        # autocast data type for optimization
        autocast_dtype = torch.bfloat16
        amp_flag = True
    elif LM.device.type == "mps":
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        # autocast data type for optimization
        autocast_dtype = torch.float16
        amp_flag = True
    else:
        device = torch.device("cpu")
        autocast_dtype = torch.float32
        amp_flag = False

    return device, autocast_dtype, amp_flag


def create_obj(cfg, field, addtl_params=None):
    """
    Docstring for create_obj

    :param cfg: Description
    :param field: Description
    :param addtl_params: Description
    """
    class_path = cfg[field]["name"]
    cls_params = cfg[field]["params"]

    # Split path into module and class
    module_name, class_name = class_path.rsplit(".", 1)

    # Dynamically import module
    mod = importlib.import_module(module_name)

    # Add config params
    if addtl_params != None:
        # this is mainly for handling optimizer
        cls_params["params"] = addtl_params

    # Retrieve class object from module
    cls = getattr(mod, class_name)
    obj = cls(**cls_params)
    return obj


def data_prep(cfg):
    """
    Docstring for data_prep

    :param cfg: Description
    """
    # Training params
    folder_path = cfg["data"]["params"]["path"]
    file_tag = cfg["data"]["params"]["tag"]
    num_shards = cfg["data"]["params"]["num_shards"]
    offset_file = 900
    # Validation params
    validation_data_path = cfg["validation"]["data"]["path"]
    validation_data_tag = cfg["validation"]["data"]["tag"]
    validation_num_shards = cfg["validation"]["data"]["num_shards"]
    ##########################
    # Training data preparation
    ##########################
    # load the data from .bin & compute the pdf for selection in training
    file_ids = range(num_shards)
    len_x = []
    shard_paths = []
    for l in file_ids:
        file_pattern = os.path.join(folder_path, f"{file_tag}{l}.bin")
        size_per_file = os.path.getsize(file_pattern)
        unit_memory = 2  # bytes
        num_tokens_per_file = size_per_file / unit_memory
        len_x.append(num_tokens_per_file)
        shard_paths.append(file_pattern)

    # compute pdf of training data
    total_tokens = sum(len_x)
    len_weight = np.asarray([(x / total_tokens) for x in len_x])

    ##########################
    # validation data preparation
    ##########################
    # Prepare list of validation shard paths
    val_shard_paths = []
    for d in range(validation_num_shards):
        file_pattern = os.path.join(
            validation_data_path, f"{validation_data_tag}{offset_file + d}.bin"
        )
        val_shard_paths.append(file_pattern)

    return shard_paths, val_shard_paths, len_weight, file_ids


def get_data_for_rank(rank, file_ids, len_weight, world_size, steps_per_epoch):
    """
    Docstring for get_data_for_rank

    :param rank: Description
    :param file_ids: Description
    :param len_weight: Description
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


def validate(rank, LM, val_data, cfg, device, autocast_dtype, amp_flag):
    """
    Validates the model in the middle of training

    Args :
        LM : Language model
        val_Data : validation shard file paths
        cfg : config file
        device : device in which validation has to happen
    """
    # model params
    B = cfg["training"]["params"]["batch_size"]
    T = cfg["model"]["params"]["context_length"]
    theta = cfg["training"]["params"]["theta"]
    # Validation params
    validation_num_shards = cfg["validation"]["data"]["num_shards"]
    validation_num_tries = cfg["validation"]["params"]["num_tries"]
    # validation loss
    ce_loss = []

    # validation loop
    LM.eval()

    for b_id in range(validation_num_shards):
        shard_file = val_data[b_id]
        tokenized_data = np.memmap(shard_file, dtype=np.uint16, mode="r")
        try:
            for b in range(validation_num_tries):
                x, target = cs336_basics.data_loader.data_loader(
                    tokenized_data,
                    batch_size=B,
                    context_length=T,
                    device_type=device.type,
                )

                # safe to move again
                x = x.to(device=device)
                target = target.to(device=device)

                # token positions
                token_pos = torch.arange(T, device=device)

                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type, dtype=autocast_dtype, enabled=amp_flag
                    ):
                        # Run the language model
                        logits = LM.tranform_lm_model(
                            x,
                            rope_theta=theta,
                            token_positions=token_pos,
                            max_seq_len=x.size(1),
                        )

                        # compute entropy loss
                        loss = cs336_basics.cross_entropy_loss.cross_entropy_loss(
                            logits, target
                        )

                # Status update
                ce_loss.append(loss.item())
        finally:
            if hasattr(tokenized_data, "_mmap"):
                tokenized_data._mmap.close()

    final_val_loss = sum(ce_loss) / len(ce_loss)
    return final_val_loss


def distributed_training_loop(rank, expt_name, world_size, cfg):
    """
    Docstring for distributed_training_loop

    :param rank: Description
    :param expt_name: Description
    :param world_size: Description
    :param cfg: Description
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
    alpha_max = cfg["training"]["params"]["alpha_max"]
    folder_path = cfg["data"]["params"]["path"]
    total_epochs = cfg["training"]["params"]["epochs"]
    total_steps_per_epoch = cfg["training"]["params"]["steps_per_epoch"]
    shards_used = torch.zeros(total_steps_per_epoch, device="cpu", dtype=torch.int64)
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
    log_every_steps = 20
    # Validation params
    validate_every_steps = cfg["validation"]["params"]["validate_every_steps"]
    val_loss = []
    val_steps = []

    ##########################
    # Create Log file object
    ##########################
    log = logging_fn(rank, cfg, expt_name)

    ##########################
    # Create model object
    ##########################
    LM = create_obj(cfg, field="model")

    ##########################
    # set the device properly
    ##########################
    device, autocast_dtype, amp_flag = rationalize_device(rank, LM)
    log.info(f"running the code in {device} ")

    ##########################
    # Initialize the setup
    ##########################
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        setup(rank=rank, world_size=world_size, backend_type="nccl")
    else:
        setup(rank=rank, world_size=world_size, backend_type="gloo")

    # Move LM to device
    LM = LM.to(device)

    # Compile the LM
    if hasattr(torch, "compile"):
        mode = "max-autotune" if device.type == "cuda" else "reduce-overhead"
        try:
            LM = torch.compile(LM, mode=mode)
        except Exception as e:
            log.warning(f"torch.compile disabled ({e}); continuing uncompiled.")

    ##########################
    # Create the optimizer
    ##########################
    O = create_obj(cfg, field="optimizer", addtl_params=[p for p in LM.parameters()])

    ##########################
    # weights & biases for training
    ##########################
    if rank == 0:
        log.info("Initializing wandb")
        run = wandb.init(
            # Set the wandb project where this run will be logged.
            project="llm_train_project",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": alpha_max,
                "architecture": model_name,
                "dataset": folder_path,
                "epochs": total_epochs,
            },
        )

    try:
        ##########################
        # Broadcast parameter
        ##########################
        # from rank 0, broadcast all params
        with torch.no_grad():
            for p in LM.parameters():
                dist.broadcast(p.data, src=0)
            for b in LM.buffers():
                dist.broadcast(b.data, src=0)

        # Initialize barrier & sync cuda
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize()
        # run the training loop
        ##########################
        # Prepare the data
        ##########################
        shard_paths, val_shard_paths, len_weight, file_ids = data_prep(cfg)

        ##########################
        # Run the training loop
        ##########################
        current_epoch = 0
        running_counter = 0  # learning rate counter
        stop_train = False  # Training flag
        LM.train()  # training flag for pytorch so that grad is available for sure
        # early exit
        stop_tensor = torch.tensor([0], device=device, dtype=torch.int)
        # Training loop
        while current_epoch < total_epochs and not (stop_train):
            # generate the data for this epoch
            shards_used.zero_()
            # different shards to memmap every epoch
            choice_shards = get_data_for_rank(
                rank, file_ids, len_weight, world_size, total_steps_per_epoch
            )
            total_time_list = []
            comm_time_list = []
            for b_id in range(total_steps_per_epoch):
                b_time_start = time.time()
                # get the shard for this b_id
                shards_used[b_id] = int(choice_shards[b_id])
                # load this shard
                shard_file = shard_paths[choice_shards[b_id]]
                tokenized_data = np.memmap(shard_file, dtype=np.uint16, mode="r")
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

                # safe to move again
                x = x.to(device=device)
                target = target.to(device=device)
                # token positions
                token_pos = torch.arange(T, device=device)
                # get the learning rate
                current_lr = cs336_basics.learning_rate_schedule.learning_rate_schedule(
                    running_counter,
                    alpha_max=alpha_max,
                    alpha_min=alpha_min,
                    warmup_iter=warmup_iter,
                    num_cosine_iter=num_cosine_iter,
                )

                # update the learning rate of optimizer
                for i in range(len(O.param_groups)):
                    O.param_groups[i]["lr"] = current_lr
                # make O zero grad
                O.zero_grad()
                ##########################
                # forward computation
                ##########################
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=amp_flag
                ):
                    # Run the language model
                    y = LM.tranform_lm_model(
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
                ##########################
                # All reduce gradient with avg computation
                ##########################
                g_time_start = time.time()
                for p in list(LM.parameters()):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                g_time_end = time.time()
                comm_time_list.append(g_time_end)
                ##########################
                # Optimiser step
                ##########################
                # Run gradient clipping
                cs336_basics.gradient_clip.gradient_clipping(
                    list(LM.parameters()), max_l2_norm=1
                )
                # Run optimizer (ZeRO-1)
                O.step()
                # update running counter
                running_counter += 1

                ##########################
                # Validation & Checkpointing model and loss
                ##########################

                if rank == 0:
                    # Validate using only rank 0
                    if running_counter % validate_every_steps == 0:
                        cal_loss_val = validate(
                            rank,
                            LM=LM,
                            val_data=val_shard_paths,
                            cfg=cfg,
                            device=device,
                            autocast_dtype=autocast_dtype,
                            amp_flag=amp_flag,
                        )
                        val_loss.append(cal_loss_val)
                        val_steps.append(running_counter)
                    # restore LM Mode
                    LM.train()
                    # Model checkpointing ( Use only rank 0)
                    if running_counter % checkpoint_every == 0:
                        pkg = {
                            "model_state": LM.state_dict(),
                            "optimizer_state": O.state_dict(),
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

                        log.info(
                            f"step={running_counter} loss={loss.item():.4f}\
                                lr={current_lr:.3e} val_loss={last_val}"
                        )

                        # wandb counter
                        log_dict_wandb = {
                            "train_loss": loss.item(),
                            "ema_loss": ema_ce_loss[-1],
                            "lr": current_lr,
                        }

                        if val_loss != []:
                            log_dict_wandb["val_loss"] = last_val

                        # log the loss in W&B
                        run.log(
                            log_dict_wandb,
                            step=running_counter,
                        )
                        if loss.item() < 1.8:
                            stop_tensor[0] = 1

                    dist.broadcast(stop_tensor, src=0)
                    if stop_tensor.item() == 1:
                        stop_train = True
                        break
                # end time
                b_time_end = time.time()
                # total time calc
                b_total_time = b_time_end - b_time_start
                total_time_list.append(b_total_time)

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
                    "optimizer_state": O.state_dict(),
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
            # complete wandb
            run.log({"loss_curve": wandb.Image("artifacts/logs/loss.png")})
            run.finish()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    # Example command :
    # python -u cs336_basics/training_loop.py lm_config.toml dummy_run
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
