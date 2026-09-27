"""Sanity check for Apple Silicon + Laya."""
from __future__ import annotations

import os
import platform
import sys

os.environ.setdefault("USE_TF", "0")

import torch
import laya


def main() -> None:
    print("python", sys.version.split()[0])
    print("machine", platform.machine())
    print("mac", platform.mac_ver()[0])
    print("laya", laya.__version__)
    print("torch", torch.__version__)
    print("mps", torch.backends.mps.is_available())
    print("script", laya.detect_script("คิดเงินซ้ำ กรุณาคืนเงิน"))


if __name__ == "__main__":
    main()
