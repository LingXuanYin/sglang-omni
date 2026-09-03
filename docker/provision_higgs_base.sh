#!/usr/bin/env bash
# Provision a fresh CUDA 13.0 host into the Higgs TTS inference base image.
#
# Run this ON the machine you intend to snapshot. It reproduces the layout the
# background service expects (see background/Dockerfile's
# INSTALL_HIGGS_DEPENDENCIES / HIGGS_SOURCE_DIR contract):
#
#   /root/miniconda3              the one interpreter every worker uses
#   /root/sglang-omni-fork        this fork, installed editable
#   /root/models/higgs-tts-3-4b   weights, fetched here rather than copied in
#   /etc/profile.d/higgs-runtime.sh   CUDA/PATH/LD_LIBRARY_PATH for JIT paths
#   /root/logs /root/pids         directories the service writes to
#
# Requirements: driver >= 580 (CUDA 13.0). torch is built for cu130; an older
# driver leaves torch.cuda.is_available() False and the pipeline cannot start.
#
# Dependencies are deliberately NOT installed from pyproject.toml's full
# dependency list: that list carries every model family sglang-omni supports
# (dots.tts, Ming-Omni, ZONOS2, gradio, whisper, nemo/pynini...), none of
# which the Higgs TTS pipeline imports, and some of which are painful to
# build. sglang is installed first and left to pin the numeric stack
# (torch / transformers / flashinfer[cu13]); the extras below are what
# higgs_tts itself needs.
#
# Usage:
#   SOURCE_ARCHIVE=/root/fork_deploy.tar.gz bash provision_higgs_base.sh
#   # or, if the source is already at /root/sglang-omni-fork, omit it.
set -euo pipefail

PY=/root/miniconda3/bin/python
SOURCE_DIR=${HIGGS_SOURCE_DIR:-/root/sglang-omni-fork}
MODEL_DIR=${HIGGS_MODEL_PATH:-/root/models/higgs-tts-3-4b}
MODEL_REPO=${HIGGS_MODEL_REPO:-bosonai/higgs-audio-v3-tts-4b}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.18}

echo "== 0. preflight =="
"$PY" -c "
import torch, sys
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'available', torch.cuda.is_available())
" || true
nvidia-smi --query-gpu=driver_version --format=csv,noheader

echo "== 1. directories =="
mkdir -p /root/models /root/logs /root/pids

echo "== 2. runtime environment =="
cat >/etc/profile.d/higgs-runtime.sh <<'PROFILE'
# CUDA 13.0 ships in the base image; flashinfer and torch.compile shell out to
# nvcc at runtime, and Miniconda must come first on PATH so subprocesses use
# the same interpreter as the worker that spawned them.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/root/miniconda3/bin:${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${CUDA_HOME}/targets/x86_64-linux/lib:/root/miniconda3/lib:${LD_LIBRARY_PATH:-}
export HIGGS_MODEL_PATH=/root/models/higgs-tts-3-4b
PROFILE

echo "== 3. dependencies =="
"$PY" -m pip install --no-cache-dir "sglang==${SGLANG_VERSION}"
# setuptools<81: pyworld still imports pkg_resources at module load, which
# setuptools removed in 81. Without the pin the voice-fusion path dies at
# import time inside the tts_engine process -- and only there, so plain TTS
# keeps working and the breakage presents as a fusion bug.
"$PY" -m pip install --no-cache-dir \
    "pyworld>=0.3.4" "setuptools<81" "soundfile>=0.12.0" "scipy>=1.10.0" \
    msgspec xxhash librosa huggingface_hub

echo "== 4. fork source =="
if [ -n "${SOURCE_ARCHIVE:-}" ]; then
    rm -rf "${SOURCE_DIR}"
    mkdir -p "${SOURCE_DIR}"
    tar xzf "${SOURCE_ARCHIVE}" -C "${SOURCE_DIR}"
fi
test -d "${SOURCE_DIR}" || { echo "no source at ${SOURCE_DIR}" >&2; exit 1; }
# --no-deps: step 3 already resolved the stack. A plain `pip install -e` here
# would drag in the full multi-model dependency list described above.
"$PY" -m pip install --no-cache-dir --no-deps -e "${SOURCE_DIR}"

echo "== 5. weights =="
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
    HF_HUB_DISABLE_XET=1 \
    "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='${MODEL_REPO}',
                        local_dir='${MODEL_DIR}', max_workers=4))
"
else
    echo "weights already present at ${MODEL_DIR}"
fi

echo "== 6. verify =="
"$PY" -c "
import sglang, sglang_omni, torch, pyworld
import sglang_omni.models.higgs_tts.stages          # noqa: F401
import sglang_omni.models.higgs_tts.fusion_reference  # noqa: F401
print('sglang_omni', sglang_omni.__version__, '|', sglang_omni.__file__)
print('sglang', sglang.__version__, '| torch', torch.__version__,
      '| cuda', torch.cuda.is_available())
"
echo "PROVISION_DONE -- run docker/higgs_image_acceptance.py before snapshotting"
