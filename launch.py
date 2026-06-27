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
import sys

from launcher.lambda_provider import (
    cmd_attach_lambda, cmd_datacenters_lambda, cmd_gpus_lambda,
    cmd_setup_lambda, cmd_terminate_lambda, cmd_train_lambda,
)
from launcher.runpod_provider import (
    cmd_attach_runpod, cmd_datacenters_runpod, cmd_gpus_runpod,
    cmd_setup_runpod, cmd_terminate_runpod, cmd_train_runpod,
)
from launcher.verda_provider import (
    cmd_attach_verda, cmd_datacenters_verda, cmd_gpus_verda,
    cmd_setup_verda, cmd_terminate_verda, cmd_train_verda,
)

_DISPATCH = {
    "lambda": dict(
        gpus=cmd_gpus_lambda,
        datacenters=cmd_datacenters_lambda,
        setup=cmd_setup_lambda,
        train=cmd_train_lambda,
        attach=cmd_attach_lambda,
        terminate=cmd_terminate_lambda,
    ),
    "runpod": dict(
        gpus=cmd_gpus_runpod,
        datacenters=cmd_datacenters_runpod,
        setup=cmd_setup_runpod,
        train=cmd_train_runpod,
        attach=cmd_attach_runpod,
        terminate=cmd_terminate_runpod,
    ),
    "verda": dict(
        gpus=cmd_gpus_verda,
        datacenters=cmd_datacenters_verda,
        setup=cmd_setup_verda,
        train=cmd_train_verda,
        attach=cmd_attach_verda,
        terminate=cmd_terminate_verda,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
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

    # gpus
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

    # datacenters
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

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _DISPATCH[args.provider][args.command](args)


if __name__ == "__main__":
    main()
