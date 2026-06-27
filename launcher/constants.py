from pathlib import Path

REPO_URL         = "https://github.com/hari201995/llm_ddp.git"
LOCAL_TRAIN_DATA = Path("/Users/hari/Documents/backups/owt_train_token_out")
LOCAL_VALID_DATA = Path("/Users/hari/Documents/backups/owt_valid_token_out")
DATA_MOUNT_BASE  = "/home/ubuntu"
REPO_DIR         = "/home/ubuntu/llm_ddp"
SSH_USERNAME     = "ubuntu"
DOCKER_IMAGE     = "lambdalabs/worker:pytorch2.3.1-cuda12.1.0"
FILESYSTEM_SIZE  = 200
SSH_KEY          = Path.home() / ".ssh" / "id_ed25519"
SSH_PUB_KEY      = Path.home() / ".ssh" / "id_ed25519.pub"
CONFIG_FILE      = Path.home() / ".llm_ddp_lambda.json"
LAMBDA_API       = "https://cloud.lambdalabs.com/api/v1"
POLL_INTERVAL    = 10

RUNPOD_REST            = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL         = "https://api.runpod.io/graphql"
RUNPOD_SSH_USERNAME    = "root"
RUNPOD_REPO_DIR        = "/root/llm_ddp"
RUNPOD_DATA_MOUNT      = "/workspace"
RUNPOD_DOCKER_IMAGE    = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
RUNPOD_CONTAINER_DISK  = 50

VERDA_REST             = "https://api.verda.com/v1"
VERDA_SSH_USERNAME     = "root"
VERDA_REPO_DIR         = "/root/llm_ddp"
VERDA_DATA_MOUNT       = "/data"
VERDA_IMAGE            = "ubuntu-22-04-cuda-12-4-docker"
VERDA_OS_VOLUME_SIZE   = 50
