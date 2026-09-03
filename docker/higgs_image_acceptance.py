"""GPU acceptance for the Higgs TTS image.

Two phases, because testing only the second one is how three deploys in a
row shipped an image that could not start:

1. Import the real service entrypoint (background.workers.higgs) in a
   subprocess. That module's import boots the pipeline through the actor's
   own warm-up, so it exercises the exact code path the deployment runs --
   including every third-party import the service pulls in. A hand-written
   `import dotenv, dramatiq, redis` list is not a substitute: it passed on a
   machine whose deployed counterpart still died on redis -> jwt ->
   cryptography.
2. Drive the pipeline directly for plain TTS and -- because this fork's
   reason for existing is voice fusion -- a two-reference fusion request,
   which exercises reference-space fusion end to end.

Set BACKGROUND_DIR to skip phase 1 when the service source is not mounted.
"""

import asyncio
import os
import subprocess
import sys
import time
import wave

import numpy as np

MODEL = os.environ.get("HIGGS_MODEL_PATH", "/root/models/higgs-tts-3-4b")
TEXT = "你好，这是推理端镜像的验收测试，音色融合功能已经就绪。"


def write_wav(chunks, out_path):
    sr, arrays = None, []
    for c in chunks:
        if getattr(c, "audio_data", None) is None:
            continue
        sr = sr or (c.sample_rate or 24000)
        arrays.append(np.asarray(c.audio_data))
    if not arrays:
        raise RuntimeError("no audio_data in chunks")
    arr = np.concatenate(arrays)
    if arr.dtype != np.int16:
        peak = max(abs(float(arr.max())), abs(float(arr.min())), 1e-8)
        arr = (arr / peak * 32767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr.tobytes())
    return len(arr) / sr


def check_service_entrypoint() -> None:
    """Phase 1: import what the deployment imports, in a throwaway process.

    Runs in a subprocess so the pipeline it boots releases the GPU before
    phase 2 starts its own. Credentials are placeholders: the module reads
    them at import time but nothing here talks to OSS.
    """
    bg = os.environ.get("BACKGROUND_DIR", "/autodl-fs/data/prod/background")
    if not os.path.isdir(bg):
        print(f"SERVICE_CHECK_SKIPPED (no service source at {bg})", flush=True)
        return
    env = {
        **os.environ,
        "PYTHONPATH": bg,
        "OSS_ACCESS_KEY_ID": "placeholder",
        "OSS_ACCESS_KEY_SECRET": "placeholder",
        "OSS_GLOBAL_ACCESS_KEY_ID": "placeholder",
        "OSS_GLOBAL_ACCESS_KEY_SECRET": "placeholder",
    }
    print("importing background.workers.higgs (boots the pipeline)...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", "import background.workers.higgs"],
        cwd=bg,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        output = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        tail = "\n".join(output)
        raise RuntimeError(
            "the deployment's own worker entrypoint does not import:\n" + tail
        )
    print(f"SERVICE_ENTRYPOINT_OK ({time.time() - t0:.1f}s)", flush=True)


async def main():
    import sglang
    import torch

    import sglang_omni
    from sglang_omni.client import Client, GenerateRequest, SamplingParams
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    print("sglang_omni", sglang_omni.__version__, sglang_omni.__file__, flush=True)
    print(
        "sglang",
        sglang.__version__,
        "| torch",
        torch.__version__,
        "| cuda",
        torch.cuda.is_available(),
        flush=True,
    )

    runner = MultiProcessPipelineRunner(HiggsTtsPipelineConfig(model_path=MODEL))
    t0 = time.time()
    await runner.start(timeout=900)
    print(f"PIPELINE_STARTED {time.time() - t0:.1f}s", flush=True)
    client = Client(runner.coordinator)

    def sampling():
        return SamplingParams(
            temperature=0.8,
            top_p=0.8,
            top_k=30,
            repetition_penalty=1.1,
            max_new_tokens=256,
            seed=42,
        )

    meta = {
        "task": "tts",
        "tts_params": {"voice": "default", "response_format": "wav", "speed": 1.0},
    }

    # 1) plain TTS
    t1 = time.time()
    chunks = [
        c
        async for c in client.generate(
            GenerateRequest(
                prompt=TEXT,
                sampling=sampling(),
                stream=False,
                output_modalities=["audio"],
                metadata=meta,
            )
        )
    ]
    dur = write_wav(chunks, "/root/acceptance_plain.wav")
    print(
        f"PLAIN_OK {len(chunks)} chunks, {dur:.2f}s audio, "
        f"{time.time() - t1:.1f}s wall",
        flush=True,
    )

    # 2) voice fusion: blend the two reference clips this fork was built for.
    refs = [p for p in (os.environ.get("REF_A"), os.environ.get("REF_B")) if p]
    if len(refs) == 2 and all(os.path.exists(p) for p in refs):
        t2 = time.time()
        prompt = {
            "text": TEXT,
            "references": [
                {"audio_path": refs[0], "weight": 0.5},
                {"audio_path": refs[1], "weight": 0.5},
            ],
        }
        chunks = [
            c
            async for c in client.generate(
                GenerateRequest(
                    prompt=prompt,
                    sampling=sampling(),
                    stream=False,
                    output_modalities=["audio"],
                    metadata=meta,
                )
            )
        ]
        dur = write_wav(chunks, "/root/acceptance_fusion.wav")
        print(
            f"FUSION_OK {len(chunks)} chunks, {dur:.2f}s audio, "
            f"{time.time() - t2:.1f}s wall (cold build incl.)",
            flush=True,
        )
    else:
        print("FUSION_SKIPPED (set REF_A/REF_B to two wav paths)", flush=True)

    try:
        await runner.stop()
    except Exception:
        pass
    print("ACCEPTANCE_DONE", flush=True)


if __name__ == "__main__":
    check_service_entrypoint()
    sys.exit(asyncio.run(main()))
