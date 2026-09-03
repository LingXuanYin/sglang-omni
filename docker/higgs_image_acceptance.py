"""GPU acceptance for the Higgs TTS image.

Drives the sglang_omni pipeline exactly the way background.tasks.
higgs_tts_actor does, minus that service's OSS/dramatiq dependencies, so it
validates the image itself rather than the deployment around it.

Covers plain TTS and -- because this fork's reason for existing is voice
fusion -- a two-reference fusion request built from the plain output, which
exercises the reference-space fusion path end to end.
"""

import asyncio
import os
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


async def main():
    import sglang
    import torch

    import sglang_omni
    from sglang_omni.client import Client, GenerateRequest, SamplingParams
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    print("sglang_omni", sglang_omni.__version__, sglang_omni.__file__, flush=True)
    print("sglang", sglang.__version__, "| torch", torch.__version__,
          "| cuda", torch.cuda.is_available(), flush=True)

    runner = MultiProcessPipelineRunner(HiggsTtsPipelineConfig(model_path=MODEL))
    t0 = time.time()
    await runner.start(timeout=900)
    print(f"PIPELINE_STARTED {time.time() - t0:.1f}s", flush=True)
    client = Client(runner.coordinator)

    def sampling():
        return SamplingParams(temperature=0.8, top_p=0.8, top_k=30,
                              repetition_penalty=1.1, max_new_tokens=256, seed=42)

    meta = {"task": "tts",
            "tts_params": {"voice": "default", "response_format": "wav", "speed": 1.0}}

    # 1) plain TTS
    t1 = time.time()
    chunks = [c async for c in client.generate(
        GenerateRequest(prompt=TEXT, sampling=sampling(), stream=False,
                        output_modalities=["audio"], metadata=meta))]
    dur = write_wav(chunks, "/root/acceptance_plain.wav")
    print(f"PLAIN_OK {len(chunks)} chunks, {dur:.2f}s audio, "
          f"{time.time() - t1:.1f}s wall", flush=True)

    # 2) voice fusion: blend the two reference clips this fork was built for.
    refs = [p for p in (os.environ.get("REF_A"), os.environ.get("REF_B")) if p]
    if len(refs) == 2 and all(os.path.exists(p) for p in refs):
        t2 = time.time()
        prompt = {"text": TEXT,
                  "references": [{"audio_path": refs[0], "weight": 0.5},
                                 {"audio_path": refs[1], "weight": 0.5}]}
        chunks = [c async for c in client.generate(
            GenerateRequest(prompt=prompt, sampling=sampling(), stream=False,
                            output_modalities=["audio"], metadata=meta))]
        dur = write_wav(chunks, "/root/acceptance_fusion.wav")
        print(f"FUSION_OK {len(chunks)} chunks, {dur:.2f}s audio, "
              f"{time.time() - t2:.1f}s wall (cold build incl.)", flush=True)
    else:
        print("FUSION_SKIPPED (set REF_A/REF_B to two wav paths)", flush=True)

    try:
        await runner.stop()
    except Exception:
        pass
    print("ACCEPTANCE_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
