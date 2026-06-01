import torch


def rationalize_device(rank, LM):
    """
    Finds the best suitable device based on the toml parameter. If run in mac, it chooses mps,
    if cuda is the specified environment, then it finds if its possible or it just falls back to
    cpu.
    It also sets the autocast flag which automatically does model quantization to speed up training.

    :param rank: Rank of the device.
    :param LM: language model object.
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
