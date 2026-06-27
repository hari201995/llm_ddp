import os
import sys
import time
from pathlib import Path

import requests

from .config import load_config, save_config
from .constants import (
    LOCAL_TRAIN_DATA, LOCAL_VALID_DATA, POLL_INTERVAL, REPO_URL,
    RUNPOD_CONTAINER_DISK, RUNPOD_DATA_MOUNT, RUNPOD_DOCKER_IMAGE,
    RUNPOD_GRAPHQL, RUNPOD_REPO_DIR, RUNPOD_REST, RUNPOD_SSH_USERNAME,
    SSH_PUB_KEY,
)
from .ssh import download_artifacts, rsync_up, run_cmd, ssh_connect


# ── API helpers ────────────────────────────────────────────────────────────────

def init_runpod() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Run: export RUNPOD_API_KEY=your_key")
        sys.exit(1)
    return api_key


def _rest(api_key: str, method: str, path: str, **kwargs) -> dict:
    resp = requests.request(
        method,
        f"{RUNPOD_REST}/{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        **kwargs,
    )
    if not resp.ok:
        raise RuntimeError(f"RunPod API error {resp.status_code}: {resp.text[:300]}")
    return resp.json() if resp.text else {}


def _graphql(api_key: str, query: str, variables: dict | None = None) -> dict:
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
            raise RuntimeError(f"RunPod GraphQL error: {body['errors']}")
        print(f"WARNING: RunPod GraphQL returned partial errors (showing what resolved): "
              f"{[e.get('message') for e in body['errors'][:1]]}{'...' if len(body['errors']) > 1 else ''}")
    return data


def get_gpu_types(api_key: str, datacenter_id: str | None = None, gpu_count: int = 1) -> list:
    """If datacenter_id is given, also fetches live per-datacenter stock/price/quantity."""
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
        data = _graphql(api_key, query, {"dcId": datacenter_id, "gpuCount": gpu_count})
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
        data = _graphql(api_key, query)
    return data["gpuTypes"] if data else []


def find_gpu_type(gpu_types: list, gpu_type: str) -> dict | None:
    """Substring match against displayName. Picks the cheapest secure-cloud match."""
    matches = [g for g in gpu_types if gpu_type.lower() in g["displayName"].lower() and g.get("secureCloud")]
    if not matches:
        return None
    return min(matches, key=lambda g: g.get("securePrice") or 9999)


def get_pool_availability(api_key: str) -> dict:
    """Returns {gpu_type_id: [{"datacenter_id":..., "stockStatus":...}, ...]}."""
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
    data = _graphql(api_key, query)
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


def get_network_volumes(api_key: str) -> list:
    return _rest(api_key, "GET", "networkvolumes")


def get_network_volume_by_name(api_key: str, name: str) -> dict | None:
    for v in get_network_volumes(api_key):
        if v.get("name") == name:
            return v
    return None


def create_network_volume(api_key: str, name: str, size_gb: int, datacenter_id: str) -> dict:
    return _rest(api_key, "POST", "networkvolumes", json={
        "name": name, "size": size_gb, "dataCenterId": datacenter_id,
    })


def launch_pod(api_key: str, name: str, gpu_type_id: str, gpu_count: int,
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
    return _rest(api_key, "POST", "pods", json=body)


def get_pod(api_key: str, pod_id: str) -> dict:
    return _rest(api_key, "GET", f"pods/{pod_id}")


def terminate_pod(api_key: str, pod_id: str):
    _rest(api_key, "DELETE", f"pods/{pod_id}")


def wait_for_pod(api_key: str, pod_id: str) -> tuple[str, int]:
    """Returns (public_ip, ssh_port) once the pod is running and SSH (22/tcp) is mapped."""
    print("Waiting for RunPod pod to be ready", end="", flush=True)
    while True:
        pod = get_pod(api_key, pod_id)
        ip = pod.get("publicIp")
        port_map = pod.get("portMappings") or {}
        ssh_port = port_map.get("22")
        if pod.get("desiredStatus") == "RUNNING" and ip and ssh_port:
            print(" ready.")
            return ip, ssh_port
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_gpus_runpod(args):
    api_key = init_runpod()
    dc_id = getattr(args, "datacenter_id", "") or ""
    gpu_count = getattr(args, "gpu_count", 1) or 1
    gpu_filter = getattr(args, "gpu_type", "") or ""

    if getattr(args, "in_stock", False):
        try:
            pool = get_pool_availability(api_key)
        except Exception as e:
            print(f"Could not query pool-wide availability ({e}).")
            print("Falling back: use 'datacenters --gpu-type <type> --provider runpod' "
                  "to check one GPU type across every datacenter instead.")
            return
        gpu_types = {g["id"]: g for g in get_gpu_types(api_key)}
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

    gpu_types = get_gpu_types(api_key, datacenter_id=dc_id or None, gpu_count=gpu_count)

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


def cmd_datacenters_runpod(args):
    api_key = init_runpod()
    gpu_filter = getattr(args, "gpu_type", "") or ""
    gpu_count = getattr(args, "gpu_count", 1) or 1
    try:
        data = _graphql(api_key, "query { dataCenters { id name location } }")
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
            gpu_types = get_gpu_types(api_key, datacenter_id=dc_id, gpu_count=gpu_count)
        except Exception as e:
            print(f"{'?':<22} {'-':<8} {dc_id:<16} {dc.get('location','?'):<20} ERROR: {e}")
            continue
        match = find_gpu_type(gpu_types, gpu_filter)
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


def cmd_setup_runpod(args):
    api_key = init_runpod()
    cfg = load_config()

    if not SSH_PUB_KEY.exists():
        print(f"ERROR: SSH public key not found at {SSH_PUB_KEY}")
        sys.exit(1)
    ssh_pub_key = SSH_PUB_KEY.read_text().strip()

    vol_name = args.volume_name
    vol = get_network_volume_by_name(api_key, vol_name)
    if vol:
        print(f"Network volume found: {vol_name} (id={vol['id']}, datacenter={vol.get('dataCenterId')})")
    else:
        if not args.datacenter_id:
            print(f"ERROR: Network volume '{vol_name}' not found, and no --datacenter-id given to create it.")
            print("Run 'python launch.py datacenters --provider runpod' to pick one.")
            sys.exit(1)
        print(f"Creating network volume '{vol_name}' ({args.volume_size}GB) in {args.datacenter_id}...")
        vol = create_network_volume(api_key, vol_name, args.volume_size, args.datacenter_id)
        print(f"Created: id={vol['id']}")

    cfg["runpod_network_volume_id"] = vol["id"]
    cfg["runpod_datacenter_id"] = vol.get("dataCenterId", args.datacenter_id)
    save_config(cfg)

    gpu_types = get_gpu_types(api_key)
    upload_gpu = find_gpu_type(gpu_types, args.gpu_type)
    if not upload_gpu:
        print(f"ERROR: No '{args.gpu_type}' GPU type found. Run 'python launch.py gpus --provider runpod'.")
        sys.exit(1)

    print(f"\nUsing {upload_gpu['displayName']} (${upload_gpu.get('securePrice', 0):.2f}/hr) for upload pod...")
    pod = launch_pod(api_key, "data-upload", upload_gpu["id"], 1, vol["id"], ssh_pub_key)
    pod_id = pod["id"]
    print(f"Pod: {pod_id}")

    try:
        host, port = wait_for_pod(api_key, pod_id)
        time.sleep(15)

        print(f"\nUploading training data to network volume at {RUNPOD_DATA_MOUNT}...")
        rsync_up(host, LOCAL_TRAIN_DATA, f"{RUNPOD_DATA_MOUNT}/", port=port, username=RUNPOD_SSH_USERNAME)
        rsync_up(host, LOCAL_VALID_DATA, f"{RUNPOD_DATA_MOUNT}/", port=port, username=RUNPOD_SSH_USERNAME)
        print("Upload complete.")
    finally:
        print(f"\nTerminating upload pod {pod_id}...")
        terminate_pod(api_key, pod_id)
        print("Setup done. Network volume is ready for training runs.")


def _build_train_cmd(repo_dir: str, data_mount: str, args) -> str:
    log_file = f"{repo_dir}/artifacts/logs/train.log"
    pid_file = f"{repo_dir}/artifacts/logs/train.pid"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    return (
        f"mkdir -p {repo_dir}/artifacts/logs {repo_dir}/artifacts/checkpoint && "
        f"cd {repo_dir} && "
        f"WANDB_API_KEY={wandb_key} WANDB_MODE=offline "
        f"nohup .venv/bin/python -u cs336_systems/DistributedTrainingLoop.py "
        f"{args.config} {args.expt_name} "
        f"> {log_file} 2>&1 & echo $! > {pid_file} && "
        f"sleep 10 && tail -f {log_file}"
    )


def cmd_train_runpod(args):
    api_key = init_runpod()
    cfg = load_config()

    if cfg.get("runpod_active_pod_id"):
        print(f"WARNING: Active RunPod pod already exists: {cfg['runpod_active_pod_id']}")
        choice = input("  [t] Terminate and start fresh  [a] Attach to it  [q] Quit: ").strip().lower()
        if choice == "t":
            terminate_pod(api_key, cfg["runpod_active_pod_id"])
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

    gpu_types = get_gpu_types(api_key)
    gpu = find_gpu_type(gpu_types, args.gpu_type)
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
    pod = launch_pod(api_key, f"train-{args.expt_name}", gpu["id"], args.gpu_count, volume_id, ssh_pub_key)
    pod_id = pod["id"]
    print(f"Pod: {pod_id}")

    cfg["runpod_active_pod_id"] = pod_id
    cfg["runpod_active_expt"] = args.expt_name
    cfg["runpod_active_out_dir"] = args.output_dir
    save_config(cfg)

    host = port = None
    try:
        host, port = wait_for_pod(api_key, pod_id)
        time.sleep(15)

        client = ssh_connect(host, port=port, username=RUNPOD_SSH_USERNAME)

        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token:
            raise RuntimeError("GITHUB_TOKEN not set. Run: export GITHUB_TOKEN=your_personal_access_token")
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
        run_cmd(client, _build_train_cmd(RUNPOD_REPO_DIR, RUNPOD_DATA_MOUNT, args), f"Training: {args.expt_name}")
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
            pod_now = get_pod(api_key, pod_id)
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
            terminate_pod(api_key, pod_id)
            cfg.pop("runpod_active_pod_id", None)
            cfg.pop("runpod_active_expt", None)
            cfg.pop("runpod_active_out_dir", None)
            save_config(cfg)
            print("Pod terminated.")


def cmd_attach_runpod(args):
    api_key = init_runpod()
    cfg = load_config()

    pod_id = cfg.get("runpod_active_pod_id")
    out_dir = cfg.get("runpod_active_out_dir", "./artifacts_remote")

    if not pod_id:
        print("No active RunPod pod found in config.")
        sys.exit(1)

    pod = get_pod(api_key, pod_id)
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
        terminate_pod(api_key, pod_id)
        cfg.pop("runpod_active_pod_id", None)
        cfg.pop("runpod_active_expt", None)
        cfg.pop("runpod_active_out_dir", None)
        save_config(cfg)
        print("Pod terminated.")


def cmd_terminate_runpod(args):
    api_key = init_runpod()
    cfg = load_config()
    pod_id = cfg.get("runpod_active_pod_id")
    if not pod_id:
        print("No active RunPod pod found in config.")
        sys.exit(1)
    print(f"Terminating pod {pod_id}...")
    terminate_pod(api_key, pod_id)
    cfg.pop("runpod_active_pod_id", None)
    cfg.pop("runpod_active_expt", None)
    cfg.pop("runpod_active_out_dir", None)
    save_config(cfg)
    print("Pod terminated.")
