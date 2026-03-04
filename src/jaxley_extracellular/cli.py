from __future__ import annotations

import argparse

from jaxley_extracellular.smoke import smoke_devices, smoke_integrate, smoke_tpu


def main() -> None:
    parser = argparse.ArgumentParser(prog="jaxley-extracellular")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke-devices", help="Print JAX devices and run matmul")
    sub.add_parser("smoke-integrate", help="Run a tiny Jaxley integrate()")
    sub.add_parser("smoke-tpu", help="Require TPU backend and run matmul")

    args = parser.parse_args()

    if args.cmd == "smoke-devices":
        smoke_devices()
    elif args.cmd == "smoke-integrate":
        smoke_integrate()
    elif args.cmd == "smoke-tpu":
        smoke_tpu()
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")
