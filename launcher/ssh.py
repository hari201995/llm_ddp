import os
import subprocess
import time
from pathlib import Path

import paramiko
import paramiko.agent

from .constants import SSH_KEY, SSH_USERNAME


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


def download_artifacts(host: str, output_dir: Path, repo_dir: str, port: int = 22, username: str = SSH_USERNAME):
    print(f"\nDownloading artifacts to {output_dir}...")
    rsync_down(host, f"{repo_dir}/artifacts/", output_dir, port=port, username=username)
    print("Artifacts downloaded.")
