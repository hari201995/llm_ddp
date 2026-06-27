#!/usr/bin/env python3
"""
Lambda Labs / RunPod / Verda training launcher for llm_ddp.

Every command takes --provider {lambda,runpod,verda}, default "lambda".

Commands:
  python launch.py setup        --provider <p>   -- one-time: create persistent storage and upload data
  python launch.py gpus         --provider <p>    -- list GPUs with pricing and availability
  python launch.py datacenters  --provider <p>    -- show region/datacenter availability for a GPU type
  python launch.py train        --provider <p> --expt-name <name>   -- launch a training run
  python launch.py attach       --provider <p>    -- reconnect to existing instance/pod and download

Examples (Lambda Labs, default):
  python launch.py datacenters --gpu-type a100
  python launch.py setup --region us-west-2
  python launch.py train --expt-name run1 --gpu-type a100 --gpu-count 2 --max-hours 8
  python launch.py attach

Examples (RunPod):
  export RUNPOD_API_KEY=your_key
  python launch.py datacenters --provider runpod
  python launch.py setup --provider runpod --datacenter-id <id>   # first time only
  python launch.py train --provider runpod --expt-name run1 --gpu-type a100 --gpu-count 2
  python launch.py attach --provider runpod

Getting data onto RunPod:
  This launcher's 'setup --provider runpod' creates a network volume (persistent
  across pods, unlike the container disk) and rsyncs your local training data onto
  it via a temporary pod, exactly like the Lambda filesystem flow. Every 'train'
  pod after that mounts the same volume at /workspace.

  If you'd rather move data manually, RunPod also offers 'runpodctl send/receive'
  -- a relay-based one-time-code transfer that needs no open ports, useful for ad
  hoc single-file transfers without going through this launcher at all:
    local$  runpodctl send myfile.tar
    pod$    runpodctl receive <code-printed-by-send>

Examples (Verda, formerly DataCrunch.io -- use as fallback if Lambda/RunPod unavailable):
  export VERDA_CLIENT_ID=your_client_id       # from verda.com/api
  export VERDA_CLIENT_SECRET=your_secret
  python launch.py gpus --provider verda --gpu-type a100
  python launch.py datacenters --provider verda
  python launch.py setup --provider verda --location-code FIN-01   # first time only
  python launch.py train --provider verda --expt-name run1 --gpu-type a100
  python launch.py attach --provider verda
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
LOCAL_TRAIN_DATA = Path("/Users/hari/Documents/backups/owt_train_token_out")
LOCAL_VALID_DATA = Path("/Users/hari/Documents/backups/owt_valid_token_out")
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

# RunPod-specific constants
RUNPOD_REST            = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL         = "https://api.runpod.io/graphql"
RUNPOD_SSH_USERNAME    = "root"                 # RunPod containers run as root
RUNPOD_REPO_DIR        = "/root/llm_ddp"        # ephemeral, on the container disk
RUNPOD_DATA_MOUNT      = "/workspace"           # network volume mount point (persistent)
RUNPOD_DOCKER_IMAGE    = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
RUNPOD_CONTAINER_DISK  = 50                     # GB, ephemeral disk for repo/deps

# Verda-specific constants (Verda = formerly DataCrunch.io)
VERDA_REST             = "https://api.verda.com/v1"
VERDA_SSH_USERNAME     = "root"                 # Verda instances default to root
VERDA_REPO_DIR         = "/root/llm_ddp"        # ephemeral, on the OS volume
VERDA_DATA_MOUNT       = "/data"                # persistent NVMe volume mount point
VERDA_IMAGE            = "ubuntu-22-04-cuda-12-4-docker"
VERDA_OS_VOLUME_SIZE   = 50                     # GB, OS volume for repo/deps


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


# ── RunPod API helpers ──────────────────────────────────────────────────────────
def init_runpod() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Run: export RUNPOD_API_KEY=your_key")
        sys.exit(1)
    return api_key


def runpod_rest(api_key: str, method: str, path: str, **kwargs) -> dict:
    resp = requests.request(
        method,
        f"{RUNPOD_REST}/{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        **kwargs,
    )
    if not resp.ok:
        raise RuntimeError(f"RunPod API error {resp.status_code}: {resp.text[:300]}")
    return resp.json() if resp.text else {}


def runpod_graphql(api_key: str, query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        RUNPOD_GRAPHQL,
        params={"api_key": api_key},
        json={"query": query, "variables": variables or {}},
    )
    if not resp.ok:
        raise RuntimeError(f"RunPod GraphQL error {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    data = body.get("data")
    if "errors" in body:
        if data is None:
            # No usable data at all -- a real failure
            raise RuntimeError(f"RunPod GraphQL error: {body['errors']}")
        # Partial success: some fields errored but others resolved. GraphQL allows
        # this -- warn and keep going rather than discarding otherwise-good data.
        print(f"WARNING: RunPod GraphQL returned partial errors (showing what resolved): "
              f"{[e.get('message') for e in body['errors'][:1]]}{'...' if len(body['errors']) > 1 else ''}")
    return data


def get_runpod_gpu_types(api_key: str, datacenter_id: str | None = None, gpu_count: int = 1) -> list:
    """If datacenter_id is given, also fetches live per-datacenter stock/price/quantity
    via lowestPrice (confirmed: this field 500s with no arguments, but resolves fine
    once dataCenterId + gpuCount are supplied).
    """
    if datacenter_id:
        query = """
        query GpuTypes($dcId: String, $gpuCount: Int!) {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            securePrice
            communityPrice
            maxGpuCountSecureCloud
            maxGpuCountCommunityCloud
            lowestPrice(input: {dataCenterId: $dcId, gpuCount: $gpuCount}) {
              stockStatus
              uninterruptablePrice
              maxGpuCount
              availableGpuCounts
            }
          }
        }
        """
        data = runpod_graphql(api_key, query, {"dcId": datacenter_id, "gpuCount": gpu_count})
    else:
        query = """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            securePrice
            communityPrice
            maxGpuCountSecureCloud
            maxGpuCountCommunityCloud
          }
        }
        """
        data = runpod_graphql(api_key, query)
    return data["gpuTypes"] if data else []


def find_runpod_gpu_type(gpu_types: list, gpu_type: str) -> dict | None:
    """Substring match against displayName, e.g. 'a100' -> 'NVIDIA A100 80GB PCIe'.
    Picks the cheapest secure-cloud match with known stock.
    """
    matches = [g for g in gpu_types if gpu_type.lower() in g["displayName"].lower() and g.get("secureCloud")]
    if not matches:
        return None
    return min(matches, key=lambda g: g.get("securePrice") or 9999)


def get_runpod_pool_availability(api_key: str) -> dict:
    """Single-call view of which GPU types currently have stock, and where.
    Tries DataCenter.gpuAvailability (one query covers every datacenter at once) --
    unverified against the live schema, so this can raise if the field/shape differs.
    Returns {gpu_type_id: [{"datacenter_id":..., "stockStatus":...}, ...]}.
    """
    query = """
    query Pool {
      dataCenters {
        id
        name
        gpuAvailability {
          gpuTypeId
          stockStatus
        }
      }
    }
    """
    data = runpod_graphql(api_key, query)
    by_gpu_type: dict = {}
    for dc in data["dataCenters"]:
        for avail in (dc.get("gpuAvailability") or []):
            gpu_type_id = avail.get("gpuTypeId")
            if not gpu_type_id:
                continue
            by_gpu_type.setdefault(gpu_type_id, []).append({
                "datacenter_id": dc.get("id"),
                "datacenter_name": dc.get("name"),
                "stockStatus": avail.get("stockStatus"),
            })
    return by_gpu_type


def get_runpod_network_volumes(api_key: str) -> list:
    return runpod_rest(api_key, "GET", "networkvolumes")


def get_runpod_network_volume_by_name(api_key: str, name: str) -> dict | None:
    for v in get_runpod_network_volumes(api_key):
        if v.get("name") == name:
            return v
    return None


def create_runpod_network_volume(api_key: str, name: str, size_gb: int, datacenter_id: str) -> dict:
    return runpod_rest(api_key, "POST", "networkvolumes", json={
        "name": name, "size": size_gb, "dataCenterId": datacenter_id,
    })


def launch_runpod_pod(api_key: str, name: str, gpu_type_id: str, gpu_count: int,
                       network_volume_id: str, ssh_pub_key: str) -> dict:
    body = {
        "name": name,
        "imageName": RUNPOD_DOCKER_IMAGE,
        "gpuTypeIds": [gpu_type_id],
        "gpuCount": gpu_count,
        "cloudType": "SECURE",
        "containerDiskInGb": RUNPOD_CONTAINER_DISK,
        "ports": ["22/tcp"],
        "networkVolumeId": network_volume_id,
        "volumeMountPath": RUNPOD_DATA_MOUNT,
        "env": {"PUBLIC_KEY": ssh_pub_key},
    }
    return runpod_rest(api_key, "POST", "pods", json=body)


def get_runpod_pod(api_key: str, pod_id: str) -> dict:
    return runpod_rest(api_key, "GET", f"pods/{pod_id}")


def terminate_runpod_pod(api_key: str, pod_id: str):
    runpod_rest(api_key, "DELETE", f"pods/{pod_id}")


def wait_for_runpod_pod(api_key: str, pod_id: str) -> tuple[str, int]:
    """Returns (public_ip, ssh_port) once the pod is running and SSH (22/tcp) is mapped."""
    print("Waiting for RunPod pod to be ready", end="", flush=True)
    while True:
        pod = get_runpod_pod(api_key, pod_id)
        ip = pod.get("publicIp")
        port_map = pod.get("portMappings") or {}
        ssh_port = port_map.get("22")
        if pod.get("desiredStatus") == "RUNNING" and ip and ssh_port:
            print(" ready.")
            return ip, ssh_port
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


# ── Verda API helpers ────────────────────────────────────────────────────────────
def init_verda() -> tuple[str, str]:
    client_id = os.environ.get("VERDA_CLIENT_ID")
    client_secret = os.environ.get("VERDA_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("ERROR: VERDA_CLIENT_ID / VERDA_CLIENT_SECRET not set. "
              "Run: export VERDA_CLIENT_ID=... VERDA_CLIENT_SECRET=...")
        sys.exit(1)
    return client_id, client_secret


def get_verda_token(client_id: str, client_secret: str) -> str:
    """OAuth2 client_credentials exchange. Tokens are typically valid for an hour,
    so we just fetch one per launch.py invocation rather than implementing refresh.
    """
    resp = requests.post(
        f"{VERDA_REST}/oauth2/token",
        json={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
    )
    if not resp.ok:
        raise RuntimeError(f"Verda OAuth2 error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["access_token"]


def verda_rest(token: str, method: str, path: str, **kwargs) -> dict:
    resp = requests.request(
        method,
        f"{VERDA_REST}/{path}",
        headers={"Authorization": f"Bearer {token}"},
        **kwargs,
    )
    if not resp.ok:
        raise RuntimeError(f"Verda API error {resp.status_code}: {resp.text[:300]}")
    return resp.json() if resp.text else {}


def get_verda_instance_types(token: str) -> list:
    return verda_rest(token, "GET", "instance-types")


def get_verda_availability(token: str) -> list:
    """Returns [{"location_code": ..., "availabilities": [instance_type, ...]}, ...]"""
    return verda_rest(token, "GET", "instance-availability")


def get_verda_locations(token: str) -> list:
    return verda_rest(token, "GET", "locations")


def find_verda_instance_type(instance_types: list, gpu_type: str) -> dict | None:
    """Substring match against the instance_type/gpu description fields.
    Picks the cheapest match.
    """
    matches = [
        it for it in instance_types
        if gpu_type.lower() in (it.get("instance_type", "") + " " + it.get("gpu", "")).lower()
    ]
    if not matches:
        return None
    return min(matches, key=lambda it: it.get("price_per_hour") or 9999)


def ensure_verda_ssh_key(token: str) -> str:
    """Register local SSH public key with Verda if not already there. Returns key id."""
    if not SSH_PUB_KEY.exists():
        raise RuntimeError(f"SSH public key not found at {SSH_PUB_KEY}")
    pub_key_text = SSH_PUB_KEY.read_text().strip()

    existing = verda_rest(token, "GET", "ssh-keys")
    for k in existing:
        if k.get("key", "").strip() == pub_key_text:
            print(f"SSH key already registered: {k.get('name')}")
            return k["id"]

    key_name = "llm-ddp-key"
    created = verda_rest(token, "POST", "ssh-keys", json={"name": key_name, "key": pub_key_text})
    print(f"SSH key registered: {key_name}")
    return created["id"]


def get_verda_volumes(token: str) -> list:
    return verda_rest(token, "GET", "volumes")


def get_verda_volume_by_name(token: str, name: str) -> dict | None:
    for v in get_verda_volumes(token):
        if v.get("name") == name:
            return v
    return None


def create_verda_volume(token: str, name: str, size_gb: int, location_code: str) -> dict:
    verda_rest(token, "POST", "volumes", json={
        "name": name, "size": size_gb, "type": "NVMe", "location_code": location_code,
    })
    # Verda's create returns 202 Accepted with no body in some versions of this API;
    # look it up by name afterward to get the real volume object reliably.
    time.sleep(3)
    vol = get_verda_volume_by_name(token, name)
    if not vol:
        raise RuntimeError(f"Volume '{name}' not found after creation -- check the Verda dashboard.")
    return vol


def launch_verda_instance(token: str, instance_type: str, location_code: str,
                           hostname: str, ssh_key_id: str, volume_ids: list | None = None) -> str:
    body = {
        "instance_type": instance_type,
        "image": VERDA_IMAGE,
        "hostname": hostname,
        "location_code": location_code,
        "ssh_key_ids": [ssh_key_id],
        "os_volume": {"name": f"{hostname}-os", "size": VERDA_OS_VOLUME_SIZE},
    }
    if volume_ids:
        body["volumes"] = volume_ids
    resp = verda_rest(token, "POST", "instances", json=body)
    # Docs show 202 Accepted with the new instance id in the body; fall back to
    # looking up by hostname if the response doesn't include it directly.
    instance_id = resp.get("id") if isinstance(resp, dict) else None
    if not instance_id:
        time.sleep(3)
        for inst in get_verda_instances(token):
            if inst.get("hostname") == hostname:
                instance_id = inst["id"]
                break
    if not instance_id:
        raise RuntimeError(f"Could not determine instance id after launch: {resp}")
    return instance_id


def get_verda_instances(token: str) -> list:
    return verda_rest(token, "GET", "instances")


def get_verda_instance(token: str, instance_id: str) -> dict:
    return verda_rest(token, "GET", f"instances/{instance_id}")


def terminate_verda_instance(token: str, instance_id: str):
    verda_rest(token, "PUT", "instances", json={
        "action": "delete", "id": instance_id, "delete_permanently": True,
    })


def wait_for_verda_instance(token: str, instance_id: str) -> str:
    print("Waiting for Verda instance to be ready", end="", flush=True)
    while True:
        inst = get_verda_instance(token, instance_id)
        if inst.get("status") == "running" and inst.get("ip"):
            print(" ready.")
            return inst["ip"]
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


# ── SSH helpers ────────────────────────────────────────────────────────────────
def ssh_connect(host: str, port: int = 22, username: str = SSH_USERNAME, retries: int = 20) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(str(SSH_KEY))
    for attempt in range(retries):
        try:
            client.connect(host, port=port, username=username, pkey=key, timeout=15)
            print(f"SSH connected to {host}:{port}")
            return client
        except Exception as e:
            print(f"  SSH not ready ({attempt+1}/{retries}): {e}")
            time.sleep(15)
    raise RuntimeError(f"Could not SSH into {host}:{port}")


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


def rsync_up(host: str, local: Path, remote: str, port: int = 22, username: str = SSH_USERNAME):
    ssh_opts = f"ssh -p {port} -i {SSH_KEY} -o StrictHostKeyChecking=no -A"
    subprocess.run(
        ["rsync", "-avz", "--progress", "-e", ssh_opts, str(local), f"{username}@{host}:{remote}"],
        check=True,
    )


def rsync_down(host: str, remote: str, local: Path, port: int = 22, username: str = SSH_USERNAME):
    local.mkdir(parents=True, exist_ok=True)
    ssh_opts = f"ssh -p {port} -i {SSH_KEY} -o StrictHostKeyChecking=no -A"
    subprocess.run(
        ["rsync", "-avz", "--progress", "-e", ssh_opts, f"{username}@{host}:{remote}", str(local) + "/"],
        check=True,
    )


def download_artifacts(host: str, output_dir: Path, repo_dir: str = REPO_DIR, port: int = 22, username: str = SSH_USERNAME):
    print(f"\nDownloading artifacts to {output_dir}...")
    rsync_down(host, f"{repo_dir}/artifacts/", output_dir, port=port, username=username)
    print("Artifacts downloaded.")


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_gpus(args):
    """List GPU types with pricing for the selected provider."""
    {"lambda": cmd_gpus_lambda, "runpod": cmd_gpus_runpod, "verda": cmd_gpus_verda}[args.provider](args)


def cmd_gpus_runpod(args):
    """List RunPod GPU types with pricing. If --datacenter-id is given, also
    shows per-datacenter stock status and price for that one datacenter. If
    --in-stock is given, shows only GPU types with stock somewhere right now,
    found via a single pool-wide query instead of looping every datacenter.
    """
    api_key = init_runpod()
    dc_id = getattr(args, "datacenter_id", "") or ""
    gpu_count = getattr(args, "gpu_count", 1) or 1
    gpu_filter = getattr(args, "gpu_type", "") or ""

    if getattr(args, "in_stock", False):
        try:
            pool = get_runpod_pool_availability(api_key)
        except Exception as e:
            print(f"Could not query pool-wide availability ({e}).")
            print("Falling back: use 'datacenters --gpu-type <type> --provider runpod' "
                  "to check one GPU type across every datacenter instead.")
            return
        gpu_types = {g["id"]: g for g in get_runpod_gpu_types(api_key)}
        print(f"\nProvider: runpod  (GPU types with stock somewhere, right now)")
        print(f"{'GPU':<35} {'VRAM':<8} {'Secure $/hr':<13} {'In stock at'}")
        print("-" * 100)
        any_found = False
        for gpu_type_id, entries in pool.items():
            g = gpu_types.get(gpu_type_id)
            if not g:
                continue
            if gpu_filter and gpu_filter.lower() not in g["displayName"].lower():
                continue
            in_stock_dcs = [e["datacenter_name"] or e["datacenter_id"] for e in entries
                             if e.get("stockStatus") and e["stockStatus"] != "UNAVAILABLE"]
            if not in_stock_dcs:
                continue
            vram = f"{g.get('memoryInGb', '?')}GB"
            secure = g.get("securePrice")
            secure_s = f"${secure:.2f}" if secure else "-"
            print(f"{g['displayName']:<35} {vram:<8} {secure_s:<13} {', '.join(in_stock_dcs)}")
            any_found = True
        if not any_found:
            print("(none matched -- either nothing is in stock right now, or the pool query "
                  "schema doesn't match what this command expects)")
        return

    gpu_types = get_runpod_gpu_types(api_key, datacenter_id=dc_id or None, gpu_count=gpu_count)

    print(f"\nProvider: runpod" + (f"  (datacenter: {dc_id}, gpu_count: {gpu_count})" if dc_id else ""))
    if dc_id:
        print(f"{'GPU':<35} {'VRAM':<8} {'Secure $/hr':<13} {'Stock':<13} {'Available counts'}")
    else:
        print(f"{'GPU':<35} {'VRAM':<8} {'Secure $/hr':<13} {'Community $/hr':<16} {'Max GPUs (acct)'}")
    print("-" * 100)
    for g in sorted(gpu_types, key=lambda g: g.get("securePrice") or 9999):
        if not (g.get("secureCloud") or g.get("communityCloud")):
            continue
        if gpu_filter and gpu_filter.lower() not in g["displayName"].lower():
            continue
        vram = f"{g.get('memoryInGb', '?')}GB"
        secure = g.get("securePrice")
        secure_s = f"${secure:.2f}" if secure else "-"
        if dc_id:
            lp = g.get("lowestPrice") or {}
            stock = lp.get("stockStatus") or "UNAVAILABLE"
            counts = lp.get("availableGpuCounts") or []
            counts_s = ", ".join(str(c) for c in counts) if counts else "-"
            print(f"{g['displayName']:<35} {vram:<8} {secure_s:<13} {stock:<13} {counts_s}")
        else:
            community = g.get("communityPrice")
            max_gpus = g.get("maxGpuCountSecureCloud") or g.get("maxGpuCountCommunityCloud") or "?"
            community_s = f"${community:.2f}" if community else "-"
            print(f"{g['displayName']:<35} {vram:<8} {secure_s:<13} {community_s:<16} {max_gpus}")
    if not dc_id:
        print(f"\n'Max GPUs (acct)' is your account-level cap, not live stock. To see where this "
              f"GPU actually has stock right now, run:\n"
              f"  python launch.py datacenters --gpu-type {gpu_filter or '<type>'} --provider runpod")


def cmd_gpus_lambda(args):
    """List GPU instance types with pricing, optionally filtered by --gpu-type
    and/or --datacenter-id (matched as a substring against region name/id)."""
    api_key = init_lambda()
    instance_types = get_instance_types(api_key)
    gpu_filter = getattr(args, "gpu_type", "") or ""
    region_filter = getattr(args, "datacenter_id", "") or ""

    print(f"\nProvider: lambda" + (f"  (region: {region_filter})" if region_filter else ""))
    print(f"{'Instance Type':<35} {'Description':<35} {'$/hr':<10} {'Regions available'}")
    print("-" * 100)

    for name, info in sorted(instance_types.items(), key=lambda x: x[1]["instance_type"].get("price_cents_per_hour", 0)):
        it      = info["instance_type"]
        desc    = it.get("description", "?")
        if gpu_filter and gpu_filter.lower() not in desc.lower():
            continue
        regions = [r["name"] for r in info.get("regions_with_capacity_available", [])]
        if region_filter and not any(region_filter.lower() in r.lower() for r in regions):
            continue
        price   = it.get("price_cents_per_hour", 0) / 100
        region_s = ", ".join(regions) if regions else "none"
        print(f"{name:<35} {desc:<35} ${price:<9.2f} {region_s}")


def cmd_datacenters(args):
    """Show datacenter/region availability for the selected provider."""
    {"lambda": cmd_datacenters_lambda, "runpod": cmd_datacenters_runpod, "verda": cmd_datacenters_verda}[args.provider](args)


def cmd_datacenters_runpod(args):
    """List RunPod datacenters. Network volumes (and therefore training pods) are pinned
    to a single datacenter, so pick one with capacity for your GPU type before 'setup'.

    If --gpu-type is given, recursively checks stock for that GPU across every
    datacenter (one API call per datacenter) and prints stock status for each.
    """
    api_key = init_runpod()
    gpu_filter = getattr(args, "gpu_type", "") or ""
    gpu_count = getattr(args, "gpu_count", 1) or 1
    try:
        data = runpod_graphql(api_key, "query { dataCenters { id name location } }")
        datacenters = data["dataCenters"]
    except Exception as e:
        print(f"Could not query datacenters via API ({e}).")
        print("Check available datacenters directly at https://console.runpod.io/deploy")
        return

    if not gpu_filter:
        print(f"\n{'Datacenter ID':<20} {'Name':<25} {'Location'}")
        print("-" * 70)
        for dc in datacenters:
            print(f"{dc.get('id','?'):<20} {dc.get('name','?'):<25} {dc.get('location','?')}")
        print("\nUse 'python launch.py setup --provider runpod --datacenter-id <id>' to pin your network volume.")
        print("Pass --gpu-type <type> to recursively check stock for a specific GPU across every datacenter.")
        return

    print(f"\nChecking '{gpu_filter}' stock (gpu_count={gpu_count}) across {len(datacenters)} datacenters "
          f"(one API call per datacenter, may take a few seconds)...\n")
    print(f"{'GPU':<22} {'$/hr':<8} {'Datacenter ID':<16} {'Location':<20} {'Stock':<13} {'Available counts'}")
    print("-" * 120)
    any_in_stock = False
    for dc in datacenters:
        dc_id = dc.get("id")
        if not dc_id:
            continue
        try:
            gpu_types = get_runpod_gpu_types(api_key, datacenter_id=dc_id, gpu_count=gpu_count)
        except Exception as e:
            print(f"{'?':<22} {'-':<8} {dc_id:<16} {dc.get('location','?'):<20} ERROR: {e}")
            continue
        match = find_runpod_gpu_type(gpu_types, gpu_filter)
        if not match:
            continue
        price = match.get("securePrice")
        price_s = f"${price:.2f}" if price else "-"
        lp = match.get("lowestPrice") or {}
        stock = lp.get("stockStatus") or "UNAVAILABLE"
        counts = lp.get("availableGpuCounts") or []
        counts_s = ", ".join(str(c) for c in counts) if counts else "-"
        print(f"{match['displayName']:<22} {price_s:<8} {dc_id:<16} {dc.get('location','?'):<20} {stock:<13} {counts_s}")
        if stock != "UNAVAILABLE":
            any_in_stock = True

    if not any_in_stock:
        print(f"\nNo datacenter currently shows '{gpu_filter}' in stock.")


def cmd_datacenters_lambda(args):
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
    """One-time: prepare persistent storage and upload training data, for the selected provider."""
    {"lambda": cmd_setup_lambda, "runpod": cmd_setup_runpod, "verda": cmd_setup_verda}[args.provider](args)


def cmd_setup_runpod(args):
    """One-time: create/locate a network volume, upload training data onto it."""
    api_key = init_runpod()
    cfg = load_config()

    if not SSH_PUB_KEY.exists():
        print(f"ERROR: SSH public key not found at {SSH_PUB_KEY}")
        sys.exit(1)
    ssh_pub_key = SSH_PUB_KEY.read_text().strip()

    vol_name = args.volume_name
    vol = get_runpod_network_volume_by_name(api_key, vol_name)
    if vol:
        print(f"Network volume found: {vol_name} (id={vol['id']}, datacenter={vol.get('dataCenterId')})")
    else:
        if not args.datacenter_id:
            print(f"ERROR: Network volume '{vol_name}' not found, and no --datacenter-id given to create it.")
            print("Run 'python launch.py datacenters --provider runpod' to pick one.")
            sys.exit(1)
        print(f"Creating network volume '{vol_name}' ({args.volume_size}GB) in {args.datacenter_id}...")
        vol = create_runpod_network_volume(api_key, vol_name, args.volume_size, args.datacenter_id)
        print(f"Created: id={vol['id']}")

    cfg["runpod_network_volume_id"] = vol["id"]
    cfg["runpod_datacenter_id"] = vol.get("dataCenterId", args.datacenter_id)
    save_config(cfg)

    # Find the cheapest GPU type with stock in this datacenter for the upload pod
    gpu_types = get_runpod_gpu_types(api_key)
    upload_gpu = find_runpod_gpu_type(gpu_types, args.gpu_type)
    if not upload_gpu:
        print(f"ERROR: No '{args.gpu_type}' GPU type found. Run 'python launch.py gpus --provider runpod'.")
        sys.exit(1)

    print(f"\nUsing {upload_gpu['displayName']} (${upload_gpu.get('securePrice', 0):.2f}/hr) for upload pod...")
    pod = launch_runpod_pod(api_key, "data-upload", upload_gpu["id"], 1, vol["id"], ssh_pub_key)
    pod_id = pod["id"]
    print(f"Pod: {pod_id}")

    try:
        host, port = wait_for_runpod_pod(api_key, pod_id)
        time.sleep(15)

        print(f"\nUploading training data to network volume at {RUNPOD_DATA_MOUNT}...")
        rsync_up(host, LOCAL_TRAIN_DATA, f"{RUNPOD_DATA_MOUNT}/", port=port, username=RUNPOD_SSH_USERNAME)
        rsync_up(host, LOCAL_VALID_DATA, f"{RUNPOD_DATA_MOUNT}/", port=port, username=RUNPOD_SSH_USERNAME)
        print("Upload complete.")
    finally:
        print(f"\nTerminating upload pod {pod_id}...")
        terminate_runpod_pod(api_key, pod_id)
        print("Setup done. Network volume is ready for training runs.")


def cmd_setup_lambda(args):
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
    """Launch a training run on the selected provider."""
    {"lambda": cmd_train_lambda, "runpod": cmd_train_runpod, "verda": cmd_train_verda}[args.provider](args)


def cmd_train_runpod(args):
    """Launch a training run on RunPod."""
    api_key = init_runpod()
    cfg = load_config()

    if cfg.get("runpod_active_pod_id"):
        print(f"WARNING: Active RunPod pod already exists: {cfg['runpod_active_pod_id']}")
        choice = input("  [t] Terminate and start fresh  [a] Attach to it  [q] Quit: ").strip().lower()
        if choice == "t":
            terminate_runpod_pod(api_key, cfg["runpod_active_pod_id"])
            cfg.pop("runpod_active_pod_id", None)
            save_config(cfg)
            print("Terminated.")
        elif choice == "a":
            cmd_attach_runpod(args)
            return
        else:
            sys.exit(0)

    volume_id = cfg.get("runpod_network_volume_id")
    if not volume_id:
        print("ERROR: Setup not complete. Run 'python launch.py setup --provider runpod' first.")
        sys.exit(1)

    if not SSH_PUB_KEY.exists():
        print(f"ERROR: SSH public key not found at {SSH_PUB_KEY}")
        sys.exit(1)
    ssh_pub_key = SSH_PUB_KEY.read_text().strip()

    gpu_types = get_runpod_gpu_types(api_key)
    gpu = find_runpod_gpu_type(gpu_types, args.gpu_type)
    if not gpu:
        print(f"ERROR: No '{args.gpu_type}' GPU type found. Run 'python launch.py gpus --provider runpod'.")
        sys.exit(1)

    price_per_hr = (gpu.get("securePrice") or 0) * args.gpu_count
    estimated_cost = price_per_hr * args.max_hours

    print(f"\n{'GPU type:':<22} {gpu['displayName']} x{args.gpu_count}")
    print(f"{'Rate:':<22} ${price_per_hr:.2f}/hr")
    print(f"{'Max hours:':<22} {args.max_hours}h")
    print(f"{'Estimated cost:':<22} ${estimated_cost:.2f}")

    if input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        print("\nWARNING: WANDB_API_KEY not set. wandb logging will fail.")

    print("\nLaunching training pod...")
    pod = launch_runpod_pod(api_key, f"train-{args.expt_name}", gpu["id"], args.gpu_count, volume_id, ssh_pub_key)
    pod_id = pod["id"]
    print(f"Pod: {pod_id}")

    cfg["runpod_active_pod_id"] = pod_id
    cfg["runpod_active_expt"] = args.expt_name
    cfg["runpod_active_out_dir"] = args.output_dir
    save_config(cfg)

    host = port = None
    try:
        host, port = wait_for_runpod_pod(api_key, pod_id)
        time.sleep(15)

        client = ssh_connect(host, port=port, username=RUNPOD_SSH_USERNAME)

        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token:
            print("ERROR: GITHUB_TOKEN not set. Run: export GITHUB_TOKEN=your_personal_access_token")
            raise RuntimeError("GITHUB_TOKEN not set")
        authed_url = REPO_URL.replace("https://", f"https://{github_token}@")
        run_cmd(client, f"git clone --branch devel {authed_url} {RUNPOD_REPO_DIR}", "Cloning repo")

        run_cmd(client,
                f"cd {RUNPOD_REPO_DIR} && python3 -m pip install uv -q && ~/.local/bin/uv sync",
                "Installing dependencies")

        run_cmd(
            client,
            f"sed -i 's|tiny_stories_train_token_out|{RUNPOD_DATA_MOUNT}/tiny_stories_train_token_out|g' {RUNPOD_REPO_DIR}/{args.config} && "
            f"sed -i 's|tiny_stories_valid_token_out|{RUNPOD_DATA_MOUNT}/tiny_stories_valid_token_out|g' {RUNPOD_REPO_DIR}/{args.config} && "
            f"grep -q 'world_size' {RUNPOD_REPO_DIR}/{args.config} "
            f"  && sed -i 's/world_size = [0-9]*/world_size = {args.gpu_count}/' {RUNPOD_REPO_DIR}/{args.config} "
            f"  || echo 'world_size = {args.gpu_count}' >> {RUNPOD_REPO_DIR}/{args.config}",
            f"Patching config (world_size={args.gpu_count}, data paths)",
        )

        run_cmd(client, f"mkdir -p {RUNPOD_REPO_DIR}/artifacts/logs {RUNPOD_REPO_DIR}/artifacts/checkpoint")

        log_file = f"{RUNPOD_REPO_DIR}/artifacts/logs/train.log"
        pid_file = f"{RUNPOD_REPO_DIR}/artifacts/logs/train.pid"
        train_cmd = (
            f"mkdir -p {RUNPOD_REPO_DIR}/artifacts/logs {RUNPOD_REPO_DIR}/artifacts/checkpoint && "
            f"cd {RUNPOD_REPO_DIR} && "
            f"WANDB_API_KEY={wandb_key} WANDB_MODE=offline "
            f"nohup .venv/bin/python -u cs336_systems/DistributedTrainingLoop.py "
            f"{args.config} {args.expt_name} "
            f"> {log_file} 2>&1 & echo $! > {pid_file} && "
            f"sleep 10 && tail -f {log_file}"
        )
        run_cmd(client, train_cmd, f"Training: {args.expt_name}")
        client.close()

        download_artifacts(host, Path(args.output_dir), repo_dir=RUNPOD_REPO_DIR, port=port, username=RUNPOD_SSH_USERNAME)

    except Exception as e:
        print(f"\nERROR during training: {e}")
        if args.keep_alive:
            print(f"--keep-alive set. Pod {pod_id} is still running.")
            print(f"  SSH in: ssh -p {port} {RUNPOD_SSH_USERNAME}@{host}")
            print(f"  Terminate later: python launch.py terminate --provider runpod")
            return
        print("Attempting to download whatever artifacts exist...")
        try:
            pod_now = get_runpod_pod(api_key, pod_id)
            if pod_now.get("desiredStatus") == "RUNNING" and pod_now.get("publicIp"):
                ip = pod_now["publicIp"]
                p = (pod_now.get("portMappings") or {}).get("22")
                if p:
                    download_artifacts(ip, Path(args.output_dir), repo_dir=RUNPOD_REPO_DIR, port=p, username=RUNPOD_SSH_USERNAME)
        except Exception as dl_err:
            print(f"Could not download artifacts: {dl_err}")
    else:
        if args.keep_alive:
            print(f"\n--keep-alive set. Pod {pod_id} is still running.")
            print(f"  SSH in: ssh -p {port} {RUNPOD_SSH_USERNAME}@{host}")
            print(f"  Terminate later: python launch.py terminate --provider runpod")
            return
    finally:
        if not args.keep_alive:
            print(f"\nTerminating pod {pod_id}...")
            terminate_runpod_pod(api_key, pod_id)
            cfg.pop("runpod_active_pod_id", None)
            cfg.pop("runpod_active_expt", None)
            cfg.pop("runpod_active_out_dir", None)
            save_config(cfg)
            print("Pod terminated.")


def cmd_train_lambda(args):
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
            f"WANDB_API_KEY={wandb_key} WANDB_MODE=offline "
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
    """Reconnect to the active instance/pod for the selected provider."""
    {"lambda": cmd_attach_lambda, "runpod": cmd_attach_runpod, "verda": cmd_attach_verda}[args.provider](args)


def cmd_attach_runpod(args):
    """Reconnect to existing RunPod pod, download artifacts, and terminate."""
    api_key = init_runpod()
    cfg = load_config()

    pod_id = cfg.get("runpod_active_pod_id")
    out_dir = cfg.get("runpod_active_out_dir", "./artifacts_remote")

    if not pod_id:
        print("No active RunPod pod found in config.")
        sys.exit(1)

    pod = get_runpod_pod(api_key, pod_id)
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")

    if pod.get("desiredStatus") != "RUNNING" or not ip or not port:
        print("Pod is not running (or SSH port not mapped yet).")
        sys.exit(1)

    client = ssh_connect(ip, port=port, username=RUNPOD_SSH_USERNAME)

    _, stdout, _ = client.exec_command("pgrep -f DistributedTrainingLoop.py")
    still_running = stdout.read().strip()

    if still_running:
        choice = input("Training still running. [w] Wait  [d] Download now and terminate: ").strip().lower()
        if choice == "w":
            run_cmd(client, f"tail -f {RUNPOD_REPO_DIR}/artifacts/logs/train.log")

    client.close()
    download_artifacts(ip, Path(out_dir), repo_dir=RUNPOD_REPO_DIR, port=port, username=RUNPOD_SSH_USERNAME)

    if input("\nTerminate pod? [y/N] ").strip().lower() == "y":
        terminate_runpod_pod(api_key, pod_id)
        cfg.pop("runpod_active_pod_id", None)
        cfg.pop("runpod_active_expt", None)
        cfg.pop("runpod_active_out_dir", None)
        save_config(cfg)
        print("Pod terminated.")


def cmd_attach_lambda(args):
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
    """Terminate the active instance/pod saved in config, for the selected provider."""
    {"lambda": cmd_terminate_lambda, "runpod": cmd_terminate_runpod, "verda": cmd_terminate_verda}[args.provider](args)


def cmd_terminate_runpod(args):
    """Terminate the active RunPod pod saved in config."""
    api_key = init_runpod()
    cfg = load_config()
    pod_id = cfg.get("runpod_active_pod_id")
    if not pod_id:
        print("No active RunPod pod found in config.")
        sys.exit(1)
    print(f"Terminating pod {pod_id}...")
    terminate_runpod_pod(api_key, pod_id)
    cfg.pop("runpod_active_pod_id", None)
    cfg.pop("runpod_active_expt", None)
    cfg.pop("runpod_active_out_dir", None)
    save_config(cfg)
    print("Pod terminated.")


def cmd_terminate_lambda(args):
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


# ── Verda commands ─────────────────────────────────────────────────────────────

def cmd_gpus_verda(args):
    """List Verda instance types with pricing and per-location availability."""
    client_id, client_secret = init_verda()
    token = get_verda_token(client_id, client_secret)
    instance_types = get_verda_instance_types(token)
    availability = get_verda_availability(token)
    gpu_filter = getattr(args, "gpu_type", "") or ""
    region_filter = getattr(args, "datacenter_id", "") or ""

    # Build {location_code: set(instance_types with capacity)}
    avail_map: dict = {}
    for loc in availability:
        code = loc.get("location_code", "")
        for it in (loc.get("availabilities") or []):
            avail_map.setdefault(it, set()).add(code)

    print(f"\nProvider: verda" + (f"  (region: {region_filter})" if region_filter else ""))
    print(f"{'Instance Type':<30} {'GPU':<20} {'VRAM':<8} {'$/hr':<10} {'Available in'}")
    print("-" * 110)
    for it in sorted(instance_types, key=lambda x: x.get("price_per_hour") or 9999):
        it_name = it.get("instance_type", "?")
        gpu = it.get("gpu", "?")
        vram = it.get("gpu_memory", "?")
        price = it.get("price_per_hour")
        if gpu_filter and gpu_filter.lower() not in (it_name + " " + gpu).lower():
            continue
        locs = sorted(avail_map.get(it_name, set()))
        if region_filter and not any(region_filter.lower() in l.lower() for l in locs):
            continue
        price_s = f"${price:.2f}" if price else "-"
        locs_s = ", ".join(locs) if locs else "none"
        print(f"{it_name:<30} {gpu:<20} {vram:<8} {price_s:<10} {locs_s}")


def cmd_datacenters_verda(args):
    """List Verda datacenter locations, optionally filtering by GPU stock."""
    client_id, client_secret = init_verda()
    token = get_verda_token(client_id, client_secret)
    locations = get_verda_locations(token)
    gpu_filter = getattr(args, "gpu_type", "") or ""

    if not gpu_filter:
        print(f"\n{'Code':<15} {'Name':<25} {'Country'}")
        print("-" * 55)
        for loc in locations:
            print(f"{loc.get('code','?'):<15} {loc.get('name','?'):<25} {loc.get('country_code','?')}")
        print("\nPass --gpu-type <type> to show which locations have that GPU available.")
        return

    availability = get_verda_availability(token)
    avail_map: dict = {}
    for loc in availability:
        for it_name in (loc.get("availabilities") or []):
            avail_map.setdefault(loc.get("location_code", ""), set()).add(it_name)

    instance_types = get_verda_instance_types(token)
    matching_its = {it.get("instance_type") for it in instance_types
                    if gpu_filter.lower() in (it.get("instance_type","") + " " + it.get("gpu","")).lower()}

    print(f"\n{'Code':<15} {'Name':<25} {'Country':<10} {'Available GPU instances'}")
    print("-" * 90)
    for loc in locations:
        code = loc.get("code", "")
        in_stock = [it for it in avail_map.get(code, set()) if it in matching_its]
        if not in_stock:
            continue
        print(f"{code:<15} {loc.get('name','?'):<25} {loc.get('country_code','?'):<10} {', '.join(in_stock)}")


def cmd_setup_verda(args):
    """One-time: register SSH key, create a persistent volume, and upload training data."""
    client_id, client_secret = init_verda()
    token = get_verda_token(client_id, client_secret)
    cfg = load_config()

    ssh_key_id = ensure_verda_ssh_key(token)
    cfg["verda_ssh_key_id"] = ssh_key_id

    vol_name = args.volume_name
    location_code = args.location_code
    vol = get_verda_volume_by_name(token, vol_name)
    if vol:
        print(f"Volume found: {vol_name} (id={vol['id']})")
    else:
        if not location_code:
            print(f"ERROR: Volume '{vol_name}' not found. Provide --location-code to create it.")
            print("Run 'python launch.py datacenters --provider verda' to see available locations.")
            sys.exit(1)
        print(f"Creating volume '{vol_name}' ({args.volume_size}GB) in {location_code}...")
        vol = create_verda_volume(token, vol_name, args.volume_size, location_code)
        print(f"Created: id={vol['id']}")

    cfg["verda_volume_id"] = vol["id"]
    cfg["verda_location_code"] = location_code or vol.get("location", {}).get("code", "")
    cfg["verda_ssh_key_id"] = ssh_key_id
    save_config(cfg)

    # Find cheapest available GPU in this location for the upload instance
    instance_types = get_verda_instance_types(token)
    availability = get_verda_availability(token)
    lc = cfg["verda_location_code"]
    avail_here = {it for loc in availability for it in (loc.get("availabilities") or [])
                  if loc.get("location_code") == lc}
    candidates = [it for it in instance_types if it.get("instance_type") in avail_here]
    if not candidates:
        print(f"ERROR: No instances available in {lc}.")
        sys.exit(1)
    cheapest = min(candidates, key=lambda it: it.get("price_per_hour") or 9999)
    price = cheapest.get("price_per_hour", 0)
    print(f"\nUsing {cheapest['instance_type']} (${price:.2f}/hr) for upload instance...")

    instance_id = launch_verda_instance(token, cheapest["instance_type"], lc,
                                        "data-upload", ssh_key_id, volume_ids=[vol["id"]])
    print(f"Instance: {instance_id}")

    try:
        host = wait_for_verda_instance(token, instance_id)
        time.sleep(15)
        client = ssh_connect(host, username=VERDA_SSH_USERNAME)
        run_cmd(client, f"mkdir -p {VERDA_DATA_MOUNT}")
        client.close()

        print(f"\nUploading training data to volume at {VERDA_DATA_MOUNT}...")
        rsync_up(host, LOCAL_TRAIN_DATA, f"{VERDA_DATA_MOUNT}/", username=VERDA_SSH_USERNAME)
        rsync_up(host, LOCAL_VALID_DATA, f"{VERDA_DATA_MOUNT}/", username=VERDA_SSH_USERNAME)
        print("Upload complete.")
    finally:
        print(f"\nTerminating upload instance {instance_id}...")
        terminate_verda_instance(token, instance_id)
        print("Setup done. Volume is ready for training runs.")


def cmd_train_verda(args):
    """Launch a training run on Verda."""
    client_id, client_secret = init_verda()
    token = get_verda_token(client_id, client_secret)
    cfg = load_config()

    if cfg.get("verda_active_instance_id"):
        print(f"WARNING: Active Verda instance already exists: {cfg['verda_active_instance_id']}")
        choice = input("  [t] Terminate and start fresh  [a] Attach to it  [q] Quit: ").strip().lower()
        if choice == "t":
            terminate_verda_instance(token, cfg["verda_active_instance_id"])
            cfg.pop("verda_active_instance_id", None)
            save_config(cfg)
            print("Terminated.")
        elif choice == "a":
            cmd_attach_verda(args)
            return
        else:
            sys.exit(0)

    volume_id = cfg.get("verda_volume_id")
    location_code = cfg.get("verda_location_code")
    ssh_key_id = cfg.get("verda_ssh_key_id")
    if not all([volume_id, location_code, ssh_key_id]):
        print("ERROR: Setup not complete. Run 'python launch.py setup --provider verda' first.")
        sys.exit(1)

    instance_types = get_verda_instance_types(token)
    it = find_verda_instance_type(instance_types, args.gpu_type)
    if not it:
        print(f"ERROR: No '{args.gpu_type}' instance type found. "
              "Run 'python launch.py gpus --provider verda'.")
        sys.exit(1)

    price_per_hr = (it.get("price_per_hour") or 0) * args.gpu_count
    estimated_cost = price_per_hr * args.max_hours

    print(f"\n{'Instance type:':<22} {it['instance_type']} x{args.gpu_count}")
    print(f"{'GPU:':<22} {it.get('gpu','?')}  {it.get('gpu_memory','?')} VRAM")
    print(f"{'Rate:':<22} ${price_per_hr:.2f}/hr")
    print(f"{'Max hours:':<22} {args.max_hours}h")
    print(f"{'Estimated cost:':<22} ${estimated_cost:.2f}")

    if input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        print("\nWARNING: WANDB_API_KEY not set. wandb logging will fail.")

    hostname = f"train-{args.expt_name}"
    print("\nLaunching training instance...")
    instance_id = launch_verda_instance(token, it["instance_type"], location_code,
                                        hostname, ssh_key_id, volume_ids=[volume_id])
    print(f"Instance: {instance_id}")

    cfg["verda_active_instance_id"] = instance_id
    cfg["verda_active_expt"] = args.expt_name
    cfg["verda_active_out_dir"] = args.output_dir
    save_config(cfg)

    host = None
    try:
        host = wait_for_verda_instance(token, instance_id)
        time.sleep(15)
        client = ssh_connect(host, username=VERDA_SSH_USERNAME)

        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token:
            raise RuntimeError("GITHUB_TOKEN not set. Run: export GITHUB_TOKEN=your_token")
        authed_url = REPO_URL.replace("https://", f"https://{github_token}@")
        run_cmd(client, f"git clone --branch devel {authed_url} {VERDA_REPO_DIR}", "Cloning repo")

        run_cmd(client, f"cd {VERDA_REPO_DIR} && python3 -m pip install uv -q && ~/.local/bin/uv sync",
                "Installing dependencies")

        run_cmd(client,
                f"sed -i 's|tiny_stories_train_token_out|{VERDA_DATA_MOUNT}/tiny_stories_train_token_out|g' {VERDA_REPO_DIR}/{args.config} && "
                f"sed -i 's|tiny_stories_valid_token_out|{VERDA_DATA_MOUNT}/tiny_stories_valid_token_out|g' {VERDA_REPO_DIR}/{args.config} && "
                f"grep -q 'world_size' {VERDA_REPO_DIR}/{args.config} "
                f"  && sed -i 's/world_size = [0-9]*/world_size = {args.gpu_count}/' {VERDA_REPO_DIR}/{args.config} "
                f"  || echo 'world_size = {args.gpu_count}' >> {VERDA_REPO_DIR}/{args.config}",
                f"Patching config (world_size={args.gpu_count})")

        run_cmd(client, f"mkdir -p {VERDA_REPO_DIR}/artifacts/logs {VERDA_REPO_DIR}/artifacts/checkpoint")

        log_file = f"{VERDA_REPO_DIR}/artifacts/logs/train.log"
        pid_file = f"{VERDA_REPO_DIR}/artifacts/logs/train.pid"
        train_cmd = (
            f"cd {VERDA_REPO_DIR} && "
            f"WANDB_API_KEY={wandb_key} WANDB_MODE=offline "
            f"nohup .venv/bin/python -u cs336_systems/DistributedTrainingLoop.py "
            f"{args.config} {args.expt_name} "
            f"> {log_file} 2>&1 & echo $! > {pid_file} && "
            f"sleep 10 && tail -f {log_file}"
        )
        run_cmd(client, train_cmd, f"Training: {args.expt_name}")
        client.close()
        download_artifacts(host, Path(args.output_dir), repo_dir=VERDA_REPO_DIR, username=VERDA_SSH_USERNAME)

    except Exception as e:
        print(f"\nERROR during training: {e}")
        if args.keep_alive:
            print(f"--keep-alive set. Instance {instance_id} is still running at {host}.")
            print(f"  SSH in: ssh {VERDA_SSH_USERNAME}@{host}")
            print(f"  Terminate later: python launch.py terminate --provider verda")
            return
        try:
            if host:
                download_artifacts(host, Path(args.output_dir), repo_dir=VERDA_REPO_DIR, username=VERDA_SSH_USERNAME)
        except Exception as dl_err:
            print(f"Could not download artifacts: {dl_err}")
    else:
        if args.keep_alive:
            print(f"\n--keep-alive set. Instance {instance_id} still running at {host}.")
            return
    finally:
        if not args.keep_alive:
            print(f"\nTerminating instance {instance_id}...")
            terminate_verda_instance(token, instance_id)
            cfg.pop("verda_active_instance_id", None)
            cfg.pop("verda_active_expt", None)
            cfg.pop("verda_active_out_dir", None)
            save_config(cfg)
            print("Instance terminated.")


def cmd_attach_verda(args):
    """Reconnect to existing Verda instance, download artifacts, and terminate."""
    client_id, client_secret = init_verda()
    token = get_verda_token(client_id, client_secret)
    cfg = load_config()

    instance_id = cfg.get("verda_active_instance_id")
    out_dir = cfg.get("verda_active_out_dir", "./artifacts_remote")
    if not instance_id:
        print("No active Verda instance found in config.")
        sys.exit(1)

    inst = get_verda_instance(token, instance_id)
    host = inst.get("ip")
    if inst.get("status") != "running" or not host:
        print("Instance is not running.")
        sys.exit(1)

    client = ssh_connect(host, username=VERDA_SSH_USERNAME)
    _, stdout, _ = client.exec_command("pgrep -f DistributedTrainingLoop.py")
    still_running = stdout.read().strip()

    if still_running:
        choice = input("Training still running. [w] Wait  [d] Download now and terminate: ").strip().lower()
        if choice == "w":
            run_cmd(client, f"tail -f {VERDA_REPO_DIR}/artifacts/logs/train.log")

    client.close()
    download_artifacts(host, Path(out_dir), repo_dir=VERDA_REPO_DIR, username=VERDA_SSH_USERNAME)

    if input("\nTerminate instance? [y/N] ").strip().lower() == "y":
        terminate_verda_instance(token, instance_id)
        cfg.pop("verda_active_instance_id", None)
        cfg.pop("verda_active_expt", None)
        cfg.pop("verda_active_out_dir", None)
        save_config(cfg)
        print("Instance terminated.")


def cmd_terminate_verda(args):
    """Terminate the active Verda instance saved in config."""
    client_id, client_secret = init_verda()
    token = get_verda_token(client_id, client_secret)
    cfg = load_config()
    instance_id = cfg.get("verda_active_instance_id")
    if not instance_id:
        print("No active Verda instance found in config.")
        sys.exit(1)
    print(f"Terminating instance {instance_id}...")
    terminate_verda_instance(token, instance_id)
    cfg.pop("verda_active_instance_id", None)
    cfg.pop("verda_active_expt", None)
    cfg.pop("verda_active_out_dir", None)
    save_config(cfg)
    print("Instance terminated.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lambda Labs / RunPod training launcher for llm_ddp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p = sub.add_parser("setup", help="One-time: verify persistent storage and upload data")
    p.add_argument("--provider", choices=["lambda", "runpod", "verda"], default="lambda")
    p.add_argument("--filesystem-name", default="llm-ddp-data",
                   help="[lambda] Name of filesystem you created at cloud.lambdalabs.com/file-system")
    p.add_argument("--region", default="", help="[lambda] Overrides region if not auto-detected from filesystem")
    p.add_argument("--volume-name", default="llm-ddp-data",
                   help="[runpod] Name of the network volume (created if it doesn't exist)")
    p.add_argument("--volume-size", type=int, default=200, help="[runpod] Network volume size in GB")
    p.add_argument("--datacenter-id", default="",
                   help="[runpod] Required only if the network volume doesn't exist yet; see 'datacenters' command")
    p.add_argument("--gpu-type", default="a100",
                   help="[runpod] GPU type for the temporary upload pod (partial match)")
    p.add_argument("--location-code", default="",
                   help="[verda] Datacenter location code (e.g. FIN-01). See 'datacenters --provider verda'.")

    # gpus -- the one command to check GPU availability + pricing, same flags on either provider
    p = sub.add_parser("gpus", help="List GPU types with pricing and availability (filterable, works the same on both providers)")
    p.add_argument("--provider", choices=["lambda", "runpod", "verda"], default="lambda")
    p.add_argument("--gpu-type", default="", help="Filter by GPU type, e.g. 'a100', 'h100' (substring match)")
    p.add_argument("--datacenter-id", default="",
                   help="Filter/show stock for one region (lambda, substring match) or "
                        "datacenter (runpod, exact ID from 'datacenters' command)")
    p.add_argument("--gpu-count", type=int, default=1,
                   help="[runpod, only with --datacenter-id] How many GPUs to check stock for")
    p.add_argument("--in-stock", action="store_true",
                   help="[runpod] Show only GPU types with stock somewhere right now "
                        "(one pool-wide query, not a per-datacenter loop)")

    # datacenters -- optional deeper dive: which specific region/datacenter has stock
    p = sub.add_parser("datacenters", help="Show GPU + price + per-datacenter stock in one table "
                       "(pass --gpu-type for runpod's one-stop view; lambda's 'gpus' already does this)")
    p.add_argument("--provider", choices=["lambda", "runpod", "verda"], default="lambda")
    p.add_argument("--gpu-type", default="",
                   help="Filter by GPU type, e.g. 'a100', 'h100'. For runpod, this triggers a "
                        "recursive stock check across every datacenter (one API call each).")
    p.add_argument("--gpu-count", type=int, default=1,
                   help="[runpod, only with --gpu-type] How many GPUs to check stock for")

    # train
    p = sub.add_parser("train", help="Launch a training run")
    p.add_argument("--provider", choices=["lambda", "runpod", "verda"], default="lambda")
    p.add_argument("--config",     default="configs/lm_config.toml")
    p.add_argument("--expt-name",  required=True)
    p.add_argument("--gpu-type",   default="a100",  help="GPU type (partial match on description/name)")
    p.add_argument("--gpu-count",  type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--max-hours",  type=float, default=10.0)
    p.add_argument("--output-dir", default="./artifacts_remote")
    p.add_argument("--keep-alive", action="store_true",
                   help="Do not terminate the instance/pod after training (useful for debugging)")

    # attach
    p = sub.add_parser("attach", help="Reconnect to existing instance/pod, download artifacts, terminate")
    p.add_argument("--provider", choices=["lambda", "runpod", "verda"], default="lambda")

    # terminate
    p = sub.add_parser("terminate", help="Terminate the active instance/pod saved in config")
    p.add_argument("--provider", choices=["lambda", "runpod", "verda"], default="lambda")

    args = parser.parse_args()
    {"setup": cmd_setup, "gpus": cmd_gpus, "datacenters": cmd_datacenters,
     "train": cmd_train, "attach": cmd_attach, "terminate": cmd_terminate}[args.command](args)


if __name__ == "__main__":
    main()
