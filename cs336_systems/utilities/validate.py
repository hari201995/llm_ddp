import cs336_basics
import torch
import numpy as np


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
                        logits = LM.module.tranform_lm_model(
                            x,
                            rope_theta=theta,
                            token_positions=token_pos,
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
