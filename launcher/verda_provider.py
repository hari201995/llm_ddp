import os
import sys
import time
from pathlib import Path

import requests

from .config import load_config, save_config
from .constants import (
    LOCAL_TRAIN_DATA, LOCAL_VALID_DATA, POLL_INTERVAL, REPO_URL,
    SSH_PUB_KEY, VERDA_DATA_MOUNT, VERDA_IMAGE, VERDA_OS_VOLUME_SIZE,
    VERDA_REPO_DIR, VERDA_REST, VERDA_SSH_USERNAME,
)
from .ssh import download_artifacts, rsync_up, run_cmd, ssh_connect


# ── API helpers ────────────────────────────────────────────────────────────────

def init_verda() -> tuple[str, str]:
    client_id = os.environ.get("VERDA_CLIENT_ID")
    client_secret = os.environ.get("VERDA_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("ERROR: VERDA_CLIENT_ID / VERDA_CLIENT_SECRET not set. "
              "Run: export VERDA_CLIENT_ID=... VERDA_CLIENT_SECRET=...")
        sys.exit(1)
    return client_id, client_secret


def get_token(client_id: str, client_secret: str) -> str:
    """OAuth2 client_credentials exchange. Tokens are typically valid for an hour."""
    resp = requests.post(
        f"{VERDA_REST}/oauth2/token",
        json={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
    )
    if not resp.ok:
        raise RuntimeError(f"Verda OAuth2 error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["access_token"]


def _rest(token: str, method: str, path: str, **kwargs) -> dict:
    resp = requests.request(
        method,
        f"{VERDA_REST}/{path}",
        headers={"Authorization": f"Bearer {token}"},
        **kwargs,
    )
    if not resp.ok:
        raise RuntimeError(f"Verda API error {resp.status_code}: {resp.text[:300]}")
    return resp.json() if resp.text else {}


def get_instance_types(token: str) -> list:
    return _rest(token, "GET", "instance-types")


def get_availability(token: str) -> list:
    """Returns [{"location_code": ..., "availabilities": [instance_type, ...]}, ...]"""
    return _rest(token, "GET", "instance-availability")


def get_locations(token: str) -> list:
    return _rest(token, "GET", "locations")


def find_instance_type(instance_types: list, gpu_type: str) -> dict | None:
    matches = [
        it for it in instance_types
        if gpu_type.lower() in (it.get("instance_type", "") + " " + it.get("gpu", "")).lower()
    ]
    if not matches:
        return None
    return min(matches, key=lambda it: it.get("price_per_hour") or 9999)


def ensure_ssh_key(token: str) -> str:
    """Register local SSH public key with Verda if not already there. Returns key id."""
    if not SSH_PUB_KEY.exists():
        raise RuntimeError(f"SSH public key not found at {SSH_PUB_KEY}")
    pub_key_text = SSH_PUB_KEY.read_text().strip()

    existing = _rest(token, "GET", "ssh-keys")
    for k in existing:
        if k.get("key", "").strip() == pub_key_text:
            print(f"SSH key already registered: {k.get('name')}")
            return k["id"]

    key_name = "llm-ddp-key"
    created = _rest(token, "POST", "ssh-keys", json={"name": key_name, "key": pub_key_text})
    print(f"SSH key registered: {key_name}")
    return created["id"]


def get_volumes(token: str) -> list:
    return _rest(token, "GET", "volumes")


def get_volume_by_name(token: str, name: str) -> dict | None:
    for v in get_volumes(token):
        if v.get("name") == name:
            return v
    return None


def create_volume(token: str, name: str, size_gb: int, location_code: str) -> dict:
    _rest(token, "POST", "volumes", json={
        "name": name, "size": size_gb, "type": "NVMe", "location_code": location_code,
    })
    # Verda's create returns 202 Accepted with no body in some versions; look up by name afterward.
    time.sleep(3)
    vol = get_volume_by_name(token, name)
    if not vol:
        raise RuntimeError(f"Volume '{name}' not found after creation -- check the Verda dashboard.")
    return vol


def get_instances(token: str) -> list:
    return _rest(token, "GET", "instances")


def get_instance(token: str, instance_id: str) -> dict:
    return _rest(token, "GET", f"instances/{instance_id}")


def launch_instance(token: str, instance_type: str, location_code: str,
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
    resp = _rest(token, "POST", "instances", json=body)
    instance_id = resp.get("id") if isinstance(resp, dict) else None
    if not instance_id:
        time.sleep(3)
        for inst in get_instances(token):
            if inst.get("hostname") == hostname:
                instance_id = inst["id"]
                break
    if not instance_id:
        raise RuntimeError(f"Could not determine instance id after launch: {resp}")
    return instance_id


def terminate_instance(token: str, instance_id: str):
    _rest(token, "PUT", "instances", json={
        "action": "delete", "id": instance_id, "delete_permanently": True,
    })


def wait_for_instance(token: str, instance_id: str) -> str:
    print("Waiting for Verda instance to be ready", end="", flush=True)
    while True:
        inst = get_instance(token, instance_id)
        if inst.get("status") == "running" and inst.get("ip"):
            print(" ready.")
            return inst["ip"]
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_gpus_verda(args):
    client_id, client_secret = init_verda()
    token = get_token(client_id, client_secret)
    instance_types = get_instance_types(token)
    availability = get_availability(token)
    gpu_filter = getattr(args, "gpu_type", "") or ""
    region_filter = getattr(args, "datacenter_id", "") or ""

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
    client_id, client_secret = init_verda()
    token = get_token(client_id, client_secret)
    locations = get_locations(token)
    gpu_filter = getattr(args, "gpu_type", "") or ""

    if not gpu_filter:
        print(f"\n{'Code':<15} {'Name':<25} {'Country'}")
        print("-" * 55)
        for loc in locations:
            print(f"{loc.get('code','?'):<15} {loc.get('name','?'):<25} {loc.get('country_code','?')}")
        print("\nPass --gpu-type <type> to show which locations have that GPU available.")
        return

    availability = get_availability(token)
    avail_map: dict = {}
    for loc in availability:
        for it_name in (loc.get("availabilities") or []):
            avail_map.setdefault(loc.get("location_code", ""), set()).add(it_name)

    instance_types = get_instance_types(token)
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
    client_id, client_secret = init_verda()
    token = get_token(client_id, client_secret)
    cfg = load_config()

    ssh_key_id = ensure_ssh_key(token)
    cfg["verda_ssh_key_id"] = ssh_key_id

    vol_name = args.volume_name
    location_code = args.location_code
    vol = get_volume_by_name(token, vol_name)
    if vol:
        print(f"Volume found: {vol_name} (id={vol['id']})")
    else:
        if not location_code:
            print(f"ERROR: Volume '{vol_name}' not found. Provide --location-code to create it.")
            print("Run 'python launch.py datacenters --provider verda' to see available locations.")
            sys.exit(1)
        print(f"Creating volume '{vol_name}' ({args.volume_size}GB) in {location_code}...")
        vol = create_volume(token, vol_name, args.volume_size, location_code)
        print(f"Created: id={vol['id']}")

    cfg["verda_volume_id"] = vol["id"]
    cfg["verda_location_code"] = location_code or vol.get("location", {}).get("code", "")
    cfg["verda_ssh_key_id"] = ssh_key_id
    save_config(cfg)

    instance_types = get_instance_types(token)
    availability = get_availability(token)
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

    instance_id = launch_instance(token, cheapest["instance_type"], lc,
                                  "data-upload", ssh_key_id, volume_ids=[vol["id"]])
    print(f"Instance: {instance_id}")

    try:
        host = wait_for_instance(token, instance_id)
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
        terminate_instance(token, instance_id)
        print("Setup done. Volume is ready for training runs.")


def _build_train_cmd(repo_dir: str, data_mount: str, args) -> str:
    log_file = f"{repo_dir}/artifacts/logs/train.log"
    pid_file = f"{repo_dir}/artifacts/logs/train.pid"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    return (
        f"cd {repo_dir} && "
        f"WANDB_API_KEY={wandb_key} WANDB_MODE=offline "
        f"nohup .venv/bin/python -u cs336_systems/DistributedTrainingLoop.py "
        f"{args.config} {args.expt_name} "
        f"> {log_file} 2>&1 & echo $! > {pid_file} && "
        f"sleep 10 && tail -f {log_file}"
    )


def cmd_train_verda(args):
    client_id, client_secret = init_verda()
    token = get_token(client_id, client_secret)
    cfg = load_config()

    if cfg.get("verda_active_instance_id"):
        print(f"WARNING: Active Verda instance already exists: {cfg['verda_active_instance_id']}")
        choice = input("  [t] Terminate and start fresh  [a] Attach to it  [q] Quit: ").strip().lower()
        if choice == "t":
            terminate_instance(token, cfg["verda_active_instance_id"])
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

    instance_types = get_instance_types(token)
    it = find_instance_type(instance_types, args.gpu_type)
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
    instance_id = launch_instance(token, it["instance_type"], location_code,
                                  hostname, ssh_key_id, volume_ids=[volume_id])
    print(f"Instance: {instance_id}")

    cfg["verda_active_instance_id"] = instance_id
    cfg["verda_active_expt"] = args.expt_name
    cfg["verda_active_out_dir"] = args.output_dir
    save_config(cfg)

    host = None
    try:
        host = wait_for_instance(token, instance_id)
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
        run_cmd(client, _build_train_cmd(VERDA_REPO_DIR, VERDA_DATA_MOUNT, args), f"Training: {args.expt_name}")
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
            terminate_instance(token, instance_id)
            cfg.pop("verda_active_instance_id", None)
            cfg.pop("verda_active_expt", None)
            cfg.pop("verda_active_out_dir", None)
            save_config(cfg)
            print("Instance terminated.")


def cmd_attach_verda(args):
    client_id, client_secret = init_verda()
    token = get_token(client_id, client_secret)
    cfg = load_config()

    instance_id = cfg.get("verda_active_instance_id")
    out_dir = cfg.get("verda_active_out_dir", "./artifacts_remote")
    if not instance_id:
        print("No active Verda instance found in config.")
        sys.exit(1)

    inst = get_instance(token, instance_id)
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
        terminate_instance(token, instance_id)
        cfg.pop("verda_active_instance_id", None)
        cfg.pop("verda_active_expt", None)
        cfg.pop("verda_active_out_dir", None)
        save_config(cfg)
        print("Instance terminated.")


def cmd_terminate_verda(args):
    client_id, client_secret = init_verda()
    token = get_token(client_id, client_secret)
    cfg = load_config()
    instance_id = cfg.get("verda_active_instance_id")
    if not instance_id:
        print("No active Verda instance found in config.")
        sys.exit(1)
    print(f"Terminating instance {instance_id}...")
    terminate_instance(token, instance_id)
    cfg.pop("verda_active_instance_id", None)
    cfg.pop("verda_active_expt", None)
    cfg.pop("verda_active_out_dir", None)
    save_config(cfg)
    print("Instance terminated.")
