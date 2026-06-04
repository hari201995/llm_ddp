#!/usr/bin/env python3
"""
Lambda Labs training launcher for llm_ddp.

Commands:
  python launch.py setup                        -- one-time: create filesystem and upload data
  python launch.py gpus                         -- list GPUs with pricing and availability
  python launch.py datacenters                  -- show availability per region for a GPU type
  python launch.py train --expt-name <name>     -- launch a training run
  python launch.py attach                       -- reconnect to existing instance and download

Examples:
  python launch.py datacenters --gpu-type a100
  python launch.py setup --region us-west-2
  python launch.py gpus
  python launch.py train --expt-name run1 --gpu-type a100 --gpu-count 2 --max-hours 8
  python launch.py attach
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import paramiko
import paramiko.agent
import requests

# ── project constants ──────────────────────────────────────────────────────────
REPO_URL         = "https://github.com/hari201995/llm_ddp.git"
LOCAL_TRAIN_DATA = Path("/Users/hari/Documents/backups/tiny_stories_train_token_out")
LOCAL_VALID_DATA = Path("/Users/hari/Documents/backups/tiny_stories_valid_token_out")
DATA_MOUNT_BASE  = "/home/ubuntu"   # Lambda mounts filesystems at /home/ubuntu/<fs-name>
REPO_DIR         = "/home/ubuntu/llm_ddp"
SSH_USERNAME     = "ubuntu"                   # Lambda uses ubuntu, not root
DOCKER_IMAGE     = "lambdalabs/worker:pytorch2.3.1-cuda12.1.0"
FILESYSTEM_SIZE  = 200                        # GB — Lambda minimum is 200 GB
SSH_KEY          = Path.home() / ".ssh" / "id_ed25519"
SSH_PUB_KEY      = Path.home() / ".ssh" / "id_ed25519.pub"
CONFIG_FILE      = Path.home() / ".llm_ddp_lambda.json"
LAMBDA_API       = "https://cloud.lambdalabs.com/api/v1"
POLL_INTERVAL    = 10


# ── config helpers ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ── Lambda API helpers ─────────────────────────────────────────────────────────
def init_lambda() -> str:
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        print("ERROR: LAMBDA_API_KEY not set. Run: export LAMBDA_API_KEY=your_key")
        sys.exit(1)
    return api_key


def api(api_key: str, method: str, endpoint: str, **kwargs) -> dict:
    resp = requests.request(
        method,
        f"{LAMBDA_API}/{endpoint}",
        headers={"Authorization": f"Bearer {api_key}"},
        **kwargs,
    )
    if not resp.ok:
        raise RuntimeError(f"Lambda API error {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def get_instance_types(api_key: str) -> dict:
    return api(api_key, "GET", "instance-types")["data"]


def get_instances(api_key: str) -> list:
    return api(api_key, "GET", "instances")["data"]


def get_filesystems(api_key: str) -> list:
    return api(api_key, "GET", "file-systems")["data"]


def get_ssh_keys(api_key: str) -> list:
    return api(api_key, "GET", "ssh-keys")["data"]


def ensure_ssh_key(api_key: str) -> str:
    """Register local SSH public key with Lambda if not already there. Returns key name."""
    if not SSH_PUB_KEY.exists():
        raise RuntimeError(f"SSH public key not found at {SSH_PUB_KEY}")

    pub_key_text = SSH_PUB_KEY.read_text().strip()
    existing = get_ssh_keys(api_key)

    # Check if already registered by comparing public key content
    for k in existing:
        if k.get("public_key", "").strip() == pub_key_text:
            print(f"SSH key already registered: {k['name']}")
            return k["name"]

    # Register it
    key_name = f"llm-ddp-key"
    api(api_key, "POST", "ssh-keys", json={"name": key_name, "public_key": pub_key_text})
    print(f"SSH key registered: {key_name}")
    return key_name


def get_filesystem_by_name(api_key: str, name: str) -> dict | None:
    """Look up an existing filesystem by name."""
    filesystems = get_filesystems(api_key)
    for fs in filesystems:
        if fs.get("name") == name:
            return fs
    return None


def launch_instance(api_key: str, instance_type: str, region: str, ssh_key_name: str,
                    filesystem_name: str, name: str) -> dict:
    resp = api(api_key, "POST", "instance-operations/launch", json={
        "instance_type_name": instance_type,
        "region_name": region,
        "ssh_key_names": [ssh_key_name],
        "file_system_names": [filesystem_name],
        "name": name,
        "quantity": 1,
    })
    # Lambda returns {"data": {"instance_ids": [...]}}
    ids = resp["data"].get("instance_ids", [])
    if not ids:
        raise RuntimeError(f"No instance IDs returned: {resp}")
    return {"id": ids[0]}


def terminate_instance(api_key: str, instance_id: str):
    api(api_key, "POST", "instance-operations/terminate", json={"instance_ids": [instance_id]})


def wait_for_instance(api_key: str, instance_id: str) -> dict:
    print("Waiting for instance to be ready", end="", flush=True)
    while True:
        instances = get_instances(api_key)
        inst = next((i for i in instances if i["id"] == instance_id), None)
        if inst and inst.get("status") == "active":
            print(" ready.")
            return inst
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


def find_instance_type(instance_types: dict, gpu_type: str, gpu_count: int, region: str = None) -> tuple[str, dict]:
    """Find instance type matching gpu_type and gpu_count, optionally filtered by region availability.

    gpu_count is matched against the instance name, e.g. gpu_1x_a100_sxm4 → count=1.
    """
    import re
    matches = []
    for name, info in instance_types.items():
        it = info.get("instance_type", {})
        desc = it.get("description", "").lower()
        if gpu_type.lower() not in desc:
            continue
        # parse count from name: gpu_Nx_... → N
        m = re.match(r"gpu_(\d+)x_", name)
        if not m or int(m.group(1)) != gpu_count:
            continue
        if region:
            available = [r["name"] for r in info.get("regions_with_capacity_available", [])]
            if region not in available:
                continue
        matches.append((name, info))

    if not matches:
        return None, None
    return min(matches, key=lambda x: x[1]["instance_type"].get("price_cents_per_hour", 9999))


# ── SSH helpers ────────────────────────────────────────────────────────────────
def ssh_connect(host: str, retries: int = 20) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(str(SSH_KEY))
    for attempt in range(retries):
        try:
            client.connect(host, port=22, username=SSH_USERNAME, pkey=key, timeout=15)
            print(f"SSH connected to {host}")
            return client
        except Exception as e:
            print(f"  SSH not ready ({attempt+1}/{retries}): {e}")
            time.sleep(15)
    raise RuntimeError(f"Could not SSH into {host}")


def run_cmd(client: paramiko.SSHClient, cmd: str, description: str = None):
    """Run command with SSH agent forwarding and live output streaming."""
    if description:
        print(f"\n>>> {description}")

    transport = client.get_transport()
    session = transport.open_session()

    # Agent forwarding — GitHub auth uses your local key, never copied to instance
    if os.environ.get("SSH_AUTH_SOCK"):
        paramiko.agent.AgentRequestHandler(session)

    session.get_pty()
    session.exec_command(cmd)

    while True:
        if session.recv_ready():
            print(session.recv(4096).decode("utf-8", errors="replace"), end="", flush=True)
        if session.exit_status_ready():
            while session.recv_ready():
                print(session.recv(4096).decode("utf-8", errors="replace"), end="", flush=True)
            break
        time.sleep(0.05)

    exit_code = session.recv_exit_status()
    session.close()

    if exit_code != 0:
        raise RuntimeError(f"Command failed (exit {exit_code}): {cmd[:80]}")


def rsync_up(host: str, local: Path, remote: str):
    ssh_opts = f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -A"
    subprocess.run(
        ["rsync", "-avz", "--progress", "-e", ssh_opts, str(local), f"{SSH_USERNAME}@{host}:{remote}"],
        check=True,
    )


def rsync_down(host: str, remote: str, local: Path):
    local.mkdir(parents=True, exist_ok=True)
    ssh_opts = f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -A"
    subprocess.run(
        ["rsync", "-avz", "--progress", "-e", ssh_opts, f"{SSH_USERNAME}@{host}:{remote}", str(local) + "/"],
        check=True,
    )


def download_artifacts(host: str, output_dir: Path):
    print(f"\nDownloading artifacts to {output_dir}...")
    rsync_down(host, f"{REPO_DIR}/artifacts/", output_dir)
    print("Artifacts downloaded.")


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_gpus(args):
    """List all GPU instance types with pricing."""
    api_key = init_lambda()
    instance_types = get_instance_types(api_key)

    print(f"\n{'Instance Type':<35} {'Description':<35} {'$/hr':<10} {'Regions available'}")
    print("-" * 100)

    for name, info in sorted(instance_types.items(), key=lambda x: x[1]["instance_type"].get("price_cents_per_hour", 0)):
        it      = info["instance_type"]
        desc    = it.get("description", "?")
        price   = it.get("price_cents_per_hour", 0) / 100
        regions = [r["name"] for r in info.get("regions_with_capacity_available", [])]
        region_s = ", ".join(regions) if regions else "none"
        print(f"{name:<35} {desc:<35} ${price:<9.2f} {region_s}")


def cmd_datacenters(args):
    """Show which regions have capacity for a given GPU type."""
    api_key = init_lambda()
    instance_types = get_instance_types(api_key)

    print(f"\nNote: Filesystem and instance MUST be in the same region.\n")
    print(f"{'Instance Type':<35} {'$/hr':<10} {'Available Regions'}")
    print("-" * 80)

    for name, info in sorted(instance_types.items(), key=lambda x: x[1]["instance_type"].get("price_cents_per_hour", 0)):
        it   = info["instance_type"]
        desc = it.get("description", "?")
        if args.gpu_type and args.gpu_type.lower() not in desc.lower():
            continue
        price   = it.get("price_cents_per_hour", 0) / 100
        regions = [f"{r['name']} ({r.get('description','')})" for r in info.get("regions_with_capacity_available", [])]
        if not regions:
            continue
        print(f"{name:<35} ${price:<9.2f} {', '.join(regions)}")


def cmd_setup(args):
    """One-time: register SSH key, verify filesystem, upload training data."""
    api_key = init_lambda()
    cfg = load_config()

    # Register SSH key
    ssh_key_name = ensure_ssh_key(api_key)
    cfg["ssh_key_name"] = ssh_key_name

    # Look up filesystem (must be created manually in the Lambda dashboard)
    fs_name = args.filesystem_name
    print(f"\nLooking up filesystem '{fs_name}'...")
    fs = get_filesystem_by_name(api_key, fs_name)
    if not fs:
        filesystems = get_filesystems(api_key)
        existing = [f["name"] for f in filesystems] if filesystems else []
        print(f"ERROR: Filesystem '{fs_name}' not found.")
        print(f"  Go to https://cloud.lambdalabs.com/file-system and create it manually.")
        if existing:
            print(f"  Your existing filesystems: {existing}")
        print(f"  Then re-run: python launch.py setup --filesystem-name <name> --region <region>")
        sys.exit(1)

    region = fs.get("region", {})
    if isinstance(region, dict):
        region = region.get("name", args.region)
    cfg["filesystem_name"] = fs_name
    cfg["region"] = region
    save_config(cfg)
    print(f"Filesystem found: {fs_name} (region: {region})")

    # Find cheapest GPU in this region for the upload instance
    instance_types = get_instance_types(api_key)
    upload_candidate = None
    for name, info in sorted(instance_types.items(), key=lambda x: x[1]["instance_type"].get("price_cents_per_hour", 9999)):
        regions = [r["name"] for r in info.get("regions_with_capacity_available", [])]
        if region in regions:
            upload_candidate = (name, info)
            break

    if not upload_candidate:
        print(f"ERROR: No instances available in {region}. Check 'python launch.py datacenters'.")
        sys.exit(1)

    inst_type_name = upload_candidate[0]
    price = upload_candidate[1]["instance_type"].get("price_cents_per_hour", 0) / 100
    print(f"\nUsing {inst_type_name} (${price:.2f}/hr) for upload instance...")

    inst = launch_instance(api_key, inst_type_name, region, ssh_key_name, fs_name, "data-upload")
    inst_id = inst["id"]
    print(f"Instance: {inst_id}")

    data_mount = f"{DATA_MOUNT_BASE}/{fs_name}"

    try:
        inst = wait_for_instance(api_key, inst_id)
        host = inst["ip"]
        time.sleep(15)

        client = ssh_connect(host)
        run_cmd(client, f"mkdir -p {data_mount}")
        client.close()

        print(f"\nUploading training data to filesystem at {data_mount}...")
        rsync_up(host, LOCAL_TRAIN_DATA, f"{data_mount}/")
        rsync_up(host, LOCAL_VALID_DATA, f"{data_mount}/")
        print("Upload complete.")
    finally:
        print(f"\nTerminating upload instance {inst_id}...")
        terminate_instance(api_key, inst_id)
        print("Setup done. Filesystem is ready for training runs.")


def cmd_train(args):
    """Launch a training run on Lambda Labs."""
    api_key = init_lambda()
    cfg = load_config()

    # Check for existing active instance
    if cfg.get("active_instance_id"):
        print(f"WARNING: Active instance already exists: {cfg['active_instance_id']}")
        choice = input("  [t] Terminate and start fresh  [a] Attach to it  [q] Quit: ").strip().lower()
        if choice == "t":
            terminate_instance(api_key, cfg["active_instance_id"])
            cfg.pop("active_instance_id", None)
            save_config(cfg)
            print("Terminated.")
        elif choice == "a":
            cmd_attach(args)
            return
        else:
            sys.exit(0)

    fs_name      = cfg.get("filesystem_name")
    region       = cfg.get("region")
    ssh_key_name = cfg.get("ssh_key_name")
    data_mount   = f"{DATA_MOUNT_BASE}/{fs_name}"

    if not all([fs_name, region, ssh_key_name]):
        print("ERROR: Setup not complete. Run 'python launch.py setup' first.")
        sys.exit(1)

    # Find matching instance type available in our region
    instance_types = get_instance_types(api_key)
    inst_type_name, inst_info = find_instance_type(instance_types, args.gpu_type, args.gpu_count, region)

    if not inst_type_name:
        print(f"ERROR: No '{args.gpu_type}' x{args.gpu_count} instance available in region '{region}'.")
        print("Run 'python launch.py gpus' to see available types and regions.")
        sys.exit(1)

    # Cost check
    price_per_hr = inst_info["instance_type"].get("price_cents_per_hour", 0) / 100
    estimated_cost = price_per_hr * args.max_hours

    print(f"\n{'Instance type:':<22} {inst_type_name}")
    print(f"{'GPU:':<22} {inst_info['instance_type'].get('description','?')}")
    print(f"{'Rate:':<22} ${price_per_hr:.2f}/hr")
    print(f"{'Max hours:':<22} {args.max_hours}h")
    print(f"{'Estimated cost:':<22} ${estimated_cost:.2f}")

    if input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        print("\nWARNING: WANDB_API_KEY not set. wandb logging will fail.")

    # Launch instance
    print("\nLaunching training instance...")
    inst = launch_instance(api_key, inst_type_name, region, ssh_key_name, fs_name, f"train-{args.expt_name}")
    inst_id = inst["id"]
    print(f"Instance: {inst_id}")

    cfg["active_instance_id"] = inst_id
    cfg["active_expt"]        = args.expt_name
    cfg["active_out_dir"]     = args.output_dir
    save_config(cfg)

    try:
        inst = wait_for_instance(api_key, inst_id)
        host = inst["ip"]
        time.sleep(15)

        client = ssh_connect(host)

        # Clone repo using HTTPS with GitHub token
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token:
            print("ERROR: GITHUB_TOKEN not set. Run: export GITHUB_TOKEN=your_personal_access_token")
            raise RuntimeError("GITHUB_TOKEN not set")
        authed_url = REPO_URL.replace("https://", f"https://{github_token}@")
        run_cmd(client, f"git clone --branch devel {authed_url} {REPO_DIR}", "Cloning repo")

        # Install dependencies
        run_cmd(client,
                f"cd {REPO_DIR} && python3 -m pip install uv -q && ~/.local/bin/uv sync",
                "Installing dependencies")

        # Patch config: data paths + world_size
        run_cmd(
            client,
            f"sed -i 's|tiny_stories_train_token_out|{data_mount}/tiny_stories_train_token_out|g' {REPO_DIR}/{args.config} && "
            f"sed -i 's|tiny_stories_valid_token_out|{data_mount}/tiny_stories_valid_token_out|g' {REPO_DIR}/{args.config} && "
            f"grep -q 'world_size' {REPO_DIR}/{args.config} "
            f"  && sed -i 's/world_size = [0-9]*/world_size = {args.gpu_count}/' {REPO_DIR}/{args.config} "
            f"  || echo 'world_size = {args.gpu_count}' >> {REPO_DIR}/{args.config}",
            f"Patching config (world_size={args.gpu_count}, data paths)",
        )

        run_cmd(client, f"mkdir -p {REPO_DIR}/artifacts/logs {REPO_DIR}/artifacts/checkpoint")

        # Launch training in background, tail log so warnings/errors are clearly separated
        log_file = f"{REPO_DIR}/artifacts/logs/train.log"
        pid_file = f"{REPO_DIR}/artifacts/logs/train.pid"
        train_cmd = (
            f"mkdir -p {REPO_DIR}/artifacts/logs {REPO_DIR}/artifacts/checkpoint && "
            f"cd {REPO_DIR} && "
            f"WANDB_API_KEY={wandb_key} WANDB_MODE=online "
            f"nohup .venv/bin/python -u cs336_systems/DistributedTrainingLoop.py "
            f"{args.config} {args.expt_name} "
            f"> {log_file} 2>&1 & echo $! > {pid_file} && "
            f"sleep 10 && tail -f {log_file}"
        )
        run_cmd(client, train_cmd, f"Training: {args.expt_name}")
        client.close()

        download_artifacts(host, Path(args.output_dir))

    except Exception as e:
        print(f"\nERROR during training: {e}")
        if args.keep_alive:
            print(f"--keep-alive set. Instance {inst_id} is still running.")
            print(f"  SSH in: ssh ubuntu@{host}")
            print(f"  Terminate later: python launch.py terminate")
            return
        print("Attempting to download whatever artifacts exist...")
        try:
            instances = get_instances(api_key)
            inst = next((i for i in instances if i["id"] == inst_id), None)
            if inst and inst.get("status") == "active":
                download_artifacts(inst["ip"], Path(args.output_dir))
        except Exception as dl_err:
            print(f"Could not download artifacts: {dl_err}")
    else:
        if args.keep_alive:
            print(f"\n--keep-alive set. Instance {inst_id} is still running.")
            print(f"  SSH in: ssh ubuntu@{host}")
            print(f"  Terminate later: python launch.py terminate")
            return
    finally:
        if not args.keep_alive:
            print(f"\nTerminating instance {inst_id}...")
            terminate_instance(api_key, inst_id)
            cfg.pop("active_instance_id", None)
            cfg.pop("active_expt", None)
            cfg.pop("active_out_dir", None)
            save_config(cfg)
            print("Instance terminated.")


def cmd_attach(args):
    """Reconnect to existing instance, download artifacts, and terminate."""
    api_key = init_lambda()
    cfg = load_config()

    inst_id = cfg.get("active_instance_id")
    out_dir = cfg.get("active_out_dir", "./artifacts_remote")

    if not inst_id:
        print("No active instance found in config.")
        sys.exit(1)

    instances = get_instances(api_key)
    inst = next((i for i in instances if i["id"] == inst_id), None)

    if not inst or inst.get("status") != "active":
        print("Instance is not running.")
        sys.exit(1)

    host   = inst["ip"]
    client = ssh_connect(host)

    _, stdout, _ = client.exec_command("pgrep -f DistributedTrainingLoop.py")
    still_running = stdout.read().strip()

    if still_running:
        choice = input("Training still running. [w] Wait  [d] Download now and terminate: ").strip().lower()
        if choice == "w":
            run_cmd(client, f"tail -f {REPO_DIR}/artifacts/logs/train.log")

    client.close()
    download_artifacts(host, Path(out_dir))

    if input("\nTerminate instance? [y/N] ").strip().lower() == "y":
        terminate_instance(api_key, inst_id)
        cfg.pop("active_instance_id", None)
        cfg.pop("active_expt", None)
        cfg.pop("active_out_dir", None)
        save_config(cfg)
        print("Instance terminated.")


def cmd_terminate(args):
    """Terminate the active instance saved in config."""
    api_key = init_lambda()
    cfg = load_config()
    inst_id = cfg.get("active_instance_id")
    if not inst_id:
        print("No active instance found in config.")
        sys.exit(1)
    print(f"Terminating instance {inst_id}...")
    terminate_instance(api_key, inst_id)
    cfg.pop("active_instance_id", None)
    cfg.pop("active_expt", None)
    cfg.pop("active_out_dir", None)
    save_config(cfg)
    print("Instance terminated.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lambda Labs training launcher for llm_ddp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p = sub.add_parser("setup", help="One-time: verify filesystem and upload data")
    p.add_argument("--filesystem-name", default="llm-ddp-data",
                   help="Name of filesystem you created at cloud.lambdalabs.com/file-system")
    p.add_argument("--region", default="", help="Overrides region if not auto-detected from filesystem")

    # gpus
    sub.add_parser("gpus", help="List all GPU instance types with pricing and availability")

    # datacenters
    p = sub.add_parser("datacenters", help="Show available regions for a GPU type")
    p.add_argument("--gpu-type", default="", help="Filter by GPU type (e.g. 'a100', 'h100')")

    # train
    p = sub.add_parser("train", help="Launch a training run")
    p.add_argument("--config",     default="configs/lm_config.toml")
    p.add_argument("--expt-name",  required=True)
    p.add_argument("--gpu-type",   default="a100",  help="GPU type (partial match on description)")
    p.add_argument("--gpu-count",  type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--max-hours",  type=float, default=10.0)
    p.add_argument("--output-dir", default="./artifacts_remote")
    p.add_argument("--keep-alive", action="store_true",
                   help="Do not terminate the instance after training (useful for debugging)")

    # attach
    sub.add_parser("attach", help="Reconnect to existing instance, download artifacts, terminate")

    # terminate
    sub.add_parser("terminate", help="Terminate the active instance saved in config")

    args = parser.parse_args()
    {"setup": cmd_setup, "gpus": cmd_gpus, "datacenters": cmd_datacenters,
     "train": cmd_train, "attach": cmd_attach, "terminate": cmd_terminate}[args.command](args)


if __name__ == "__main__":
    main()
