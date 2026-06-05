import numpy as np
import os


def data_prep(cfg):
    """
    Data Prepping function that computes pdf at which shards have to be selected.

    :param cfg: Config file path of the experiment
    """

    # Training params
    folder_path = cfg["data"]["params"]["path"]
    file_tag = cfg["data"]["params"]["tag"]
    num_shards = cfg["data"]["params"]["num_shards"]
    offset_file = 0

    # Validation params
    validation_data_path = cfg["validation"]["data"]["path"]
    validation_data_tag = cfg["validation"]["data"]["tag"]
    validation_num_shards = cfg["validation"]["data"]["num_shards"]
    validation_shard_offset = cfg["validation"]["data"].get("shard_offset", 0)

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
            validation_data_path, f"{validation_data_tag}{validation_shard_offset + d}.bin"
        )
        val_shard_paths.append(file_pattern)

    return shard_paths, val_shard_paths, len_weight, file_ids
