"""Run the Vitis AI 3.0 docker image inside Google Colab, which has no Docker daemon.

The image is pulled with skopeo, unpacked to a folder with umoci, and entered with chroot.
In a Colab cell:

    !apt-get -qq update && apt-get -qq install -y skopeo umoci
    !skopeo copy docker://docker.io/xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 oci:/content/vai_oci:3.0
    !umoci unpack --image /content/vai_oci:3.0 /content/vai_bundle && rm -rf /content/vai_oci

    from vitis_ai_colab import setup, vai, WORK
    setup()                                   # bind mounts; /content/vai_work outside = /workspace inside
    vai("python --version; which vai_c_xir")  # Python 3.7.12, torch 1.12.1, vai_c_xir

Vitis AI 3.0 (not 3.5) is used because the ZCU104 prebuilt SD image runs VART/XIR 3.0.0.
"""
import json
import os
import subprocess

ROOTFS = "/content/vai_bundle/rootfs"
CONFIG = "/content/vai_bundle/config.json"
WORK = "/content/vai_work"
_ENV = None


def sh(cmd):
    subprocess.run(cmd, shell=True, check=False)


def setup(rootfs=ROOTFS, work=WORK, config=CONFIG):
    """Bind /proc, /sys, /dev and the work folder into the unpacked image. Safe to re-run."""
    global _ENV
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.join(rootfs, "workspace"), exist_ok=True)
    for d in ("proc", "sys", "dev"):
        if not os.path.ismount(os.path.join(rootfs, d)):
            sh("mount --bind /%s %s/%s" % (d, rootfs, d))
    if not os.path.ismount(os.path.join(rootfs, "workspace")):
        sh("mount --bind %s %s/workspace" % (work, rootfs))
    sh("cp /etc/resolv.conf %s/etc/resolv.conf" % rootfs)
    _ENV = dict(e.split("=", 1) for e in json.load(open(config))["process"]["env"])
    _ENV.setdefault("HOME", "/root")


def vai(cmd, rootfs=ROOTFS):
    """Run a shell command inside Vitis AI (conda env vitis-ai-pytorch, cwd /workspace), streaming output."""
    if _ENV is None:
        setup(rootfs)
    full = ("source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch "
            "&& cd /workspace && " + cmd)
    p = subprocess.Popen(["chroot", rootfs, "/bin/bash", "-c", full], env=_ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    return p.wait()


def vai_background(script="run_pipeline.sh", log="pipeline.log", rootfs=ROOTFS):
    """Start a script from /workspace in the background (survives a Colab cell timeout); watch the log."""
    if _ENV is None:
        setup(rootfs)
    subprocess.Popen(["chroot", rootfs, "/bin/bash", "-c",
                      "nohup bash /workspace/%s > /workspace/%s 2>&1 &" % (script, log)], env=_ENV)
    print("started; follow it with: !tail -f %s/%s" % (WORK, log))
