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

echo "== 6. background service dependencies =="
# The deployment runs background/main.py straight out of this image, so the
# service's own dependencies have to be here -- without them it dies at import
# on python-dotenv before any Higgs code runs. They used to arrive from
# background/Dockerfile's `pip install -e .`.
#
# Deliberately NOT background's whole dependency list. That list also carries
# its other worker types (aider-chat, playwright, codewithgpu, volcengine),
# and aider-chat in particular pins an older universe -- huggingface-hub 0.31,
# numpy 1.26, openai 1.x -- which pip will happily install over this image's
# stack, leaving transformers unimportable and the pipeline dead. The Higgs
# worker imports none of them. This list is what background/Dockerfile's own
# smoke test asserts a Higgs image must import, plus the extras those need.
#
# The constraints cover every package the two sides share, not just torch:
# a narrower list still lets pip walk numpy and huggingface-hub backwards.
cat >/root/higgs-stack-constraints.txt <<'CONSTRAINTS'
torch==2.13.0+cu130
torchvision==0.28.0
sglang==0.5.18
transformers==5.12.1
flashinfer-python==0.6.17
numpy>=2.1
huggingface-hub>=0.36.0
openai==2.6.1
CONSTRAINTS
"$PY" -m pip install --no-cache-dir -c /root/higgs-stack-constraints.txt     "python-dotenv>=1.0.1,<2.0.0"     "dramatiq[redis,watch]>=2.0.0,<3.0.0"     "redis>=5.2.1,<6.0.0"     "oss2>=2.19.1,<3.0.0"     "opentelemetry-sdk>=1.30.0,<2.0.0"     "opentelemetry-api>=1.30.0,<2.0.0"     "opentelemetry-exporter-otlp-proto-grpc>=1.30.0,<2.0.0"     "opentelemetry-exporter-otlp-proto-http>=1.30.0,<2.0.0"     "opentelemetry-instrumentation-httpx>=0.51b0,<1.0.0"     "opentelemetry-instrumentation-requests>=0.51b0,<1.0.0"     "ffmpeg-python>=0.2.0,<0.3.0"     "mutagen>=1.47.0,<2.0.0"     "httpx[socks]>=0.28.1,<0.29.0"     "pyyaml>=6.0.2,<7.0.0"     "pillow>=11.2.1,<12.0.0"     "json-repair>=0.47.7"     "websockets>=14.1.0,<15.0.0"

echo "== 7. verify =="
"$PY" -c "
import numpy, sglang, sglang_omni, torch, pyworld
import sglang_omni.models.higgs_tts.stages          # noqa: F401
import sglang_omni.models.higgs_tts.fusion_reference  # noqa: F401
# the service side must import too, in the same interpreter
import dotenv, dramatiq, redis, oss2, opentelemetry.sdk  # noqa: F401
import mutagen, ffmpeg, httpx, pydantic, yaml           # noqa: F401
print('numpy', numpy.__version__)
print('sglang_omni', sglang_omni.__version__, '|', sglang_omni.__file__)
print('sglang', sglang.__version__, '| torch', torch.__version__,
      '| cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'torch cannot see the GPU'
"
echo "PROVISION_DONE -- run docker/higgs_image_acceptance.py before snapshotting"
