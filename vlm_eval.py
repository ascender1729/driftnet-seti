#!/usr/bin/env python3
"""Zero-shot (and few-shot) VLM classification of dynamic spectrograms - the novel core.

Reuses the ILIAD crossfamily Bedrock-converse pattern, but with IMAGE input. Each VLM is shown
a rendered spectrogram PNG and asked to classify it as signal / rfi / noise, zero-shot (no
training, no examples) or few-shot (k labeled example images in context). Cloud models via
Bedrock converse; HF open-source VLMs are handled by hf_vlm_eval.py (separate, GPU/endpoint).

  vlm_eval.py --smoke                         # 9 images, 2 models, validate the loop
  vlm_eval.py --split data_test --models m1,m2 --shots 0 --out vlm_zeroshot.json
"""
import argparse, base64, json, os, re, sys, time
import boto3
from botocore.config import Config

REGION = "us-east-1"
br = boto3.client("bedrock-runtime", region_name=REGION,
                  config=Config(read_timeout=120, connect_timeout=20, retries={"max_attempts": 3}))

HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"
_HF_TOK = None
def _hf_token():
    global _HF_TOK
    if _HF_TOK is None:
        _HF_TOK = os.environ.get("HF_TOKEN")
        if not _HF_TOK:
            raise RuntimeError("set the HF_TOKEN environment variable to run the open-source VLM path")
    return _HF_TOK

CLASSES = ("signal", "rfi", "noise")
PROMPT = (
    "You are analyzing a radio-telescope dynamic spectrum (a 'waterfall': x-axis = frequency, "
    "y-axis = time, brightness = power). Classify the image into EXACTLY ONE class:\n"
    "- signal: a thin bright line that DRIFTS diagonally across the image (changes frequency over time)\n"
    "- rfi: a thin bright VERTICAL line at a constant frequency (does not drift)\n"
    "- noise: no coherent line, just background speckle\n"
    "Answer with ONLY one word: signal, rfi, or noise."
)

# Bedrock vision-capable model ids (converse, image input). Unavailable ones are skipped.
# 9 VLMs, 4 families, NO Nova (dropped per instruction). 7 Bedrock + 2 HuggingFace-PRO open.
DEFAULT_MODELS = [
    "us.anthropic.claude-opus-4-8",                 # Anthropic - frontier tier
    "us.anthropic.claude-sonnet-5",                 # Anthropic - mid tier
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",  # Anthropic - fast tier
    "qwen.qwen3-vl-235b-a22b",                       # Qwen (Bedrock)
    "google.gemma-3-27b-it",                         # Google
    "us.meta.llama4-maverick-17b-instruct-v1:0",     # Meta - Maverick
    "us.meta.llama4-scout-17b-instruct-v1:0",        # Meta - Scout
    "hf:Qwen/Qwen2.5-VL-72B-Instruct",               # HF PRO open
    "hf:Qwen/Qwen3-VL-30B-A3B-Instruct",             # HF PRO open
]


def _img_block(path):
    with open(path, "rb") as f:
        return {"image": {"format": "png", "source": {"bytes": f.read()}}}


def _parse(text):
    t = (text or "").lower()
    for c in CLASSES:                       # first class word that appears wins
        if re.search(r"\b" + c + r"\b", t):
            return c
    return "noise"                          # ponytail: default to the null class on a non-answer


def _img_uri(path):
    import base64
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _hf_classify(repo, png_path, shots):
    import urllib.request
    content = []
    for ex_path, ex_cls in shots:
        content.append({"type": "image_url", "image_url": {"url": _img_uri(ex_path)}})
        content.append({"type": "text", "text": f"Example - this one is: {ex_cls}"})
    content.append({"type": "image_url", "image_url": {"url": _img_uri(png_path)}})
    content.append({"type": "text", "text": PROMPT})
    body = json.dumps({"model": repo, "max_tokens": 16, "temperature": 0.0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(HF_ROUTER, data=body,
                                 headers={"Authorization": "Bearer " + _hf_token(), "Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=90).read())
    return _parse(r["choices"][0]["message"]["content"])


def classify(model_id, png_path, shots):
    if model_id.startswith("hf:"):          # HuggingFace open VLM via PRO Inference Providers
        return _hf_classify(model_id[3:], png_path, shots)
    content = []
    for ex_path, ex_cls in shots:           # few-shot: labeled examples first
        content.append(_img_block(ex_path))
        content.append({"text": f"Example - this one is: {ex_cls}"})
    content.append(_img_block(png_path))
    content.append({"text": PROMPT})
    msgs = [{"role": "user", "content": content}]
    try:                                                    # newest Claude (opus-4-8, sonnet-5) reject temperature
        r = br.converse(modelId=model_id, messages=msgs,
                        inferenceConfig={"maxTokens": 12, "temperature": 0.0})
    except Exception as e:
        if "temperature" in str(e).lower():
            r = br.converse(modelId=model_id, messages=msgs, inferenceConfig={"maxTokens": 12})
        else:
            raise
    return _parse("".join(b.get("text", "") for b in r["output"]["message"]["content"]))


def pick_shots(split, k, seed=1):
    if k == 0:
        return []
    import random
    lab = json.load(open(f"{split}/labels.json"))
    rng = random.Random(seed)
    shots = []
    for c in CLASSES:                        # one clear (high-SNR) example per class, x (k per class handled by caller)
        pool = [i for i, m in lab.items() if m["cls"] == c and (m["snr_db"] >= 14 or c == "noise")]
        for _id in rng.sample(pool, min(k, len(pool))):
            shots.append((f"{split}/png/{_id}.png", c))
    rng.shuffle(shots)
    return shots


def run(split, models, shots_k, out, limit=None):
    lab = json.load(open(f"{split}/labels.json"))
    ids = list(lab)[:limit] if limit else list(lab)
    shot_ids = {p for p, _ in pick_shots(split, shots_k)}
    ids = [i for i in ids if f"{split}/png/{i}.png" not in shot_ids]   # never test on a shot example
    shots = pick_shots(split, shots_k)
    results = {}
    for model_id in models:
        per, correct, err = {}, 0, 0
        t0 = time.time()
        for _id in ids:
            try:
                p = classify(model_id, f"{split}/png/{_id}.png", shots)
            except Exception as e:
                p = "noise"; err += 1
                if err <= 2:
                    print(f"  [{model_id.split('.')[-1][:20]}] err: {type(e).__name__} {str(e)[:80]}", flush=True)
            true = lab[_id]["cls"]
            per[_id] = {"true": true, "pred": p, "snr_db": lab[_id]["snr_db"]}
            correct += int(p == true)
        acc = correct / max(len(ids), 1)
        results[model_id] = {"shots": shots_k, "acc": acc, "n": len(ids), "errors": err,
                             "secs": round(time.time() - t0, 1), "per": per}
        print(f"[{model_id}] shots={shots_k} acc={acc:.3f} n={len(ids)} err={err} ({results[model_id]['secs']}s)", flush=True)
    if out:
        json.dump(results, open(out, "w"), indent=1)
        print(f"-> {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--split", default="data_test")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        run("data_test", DEFAULT_MODELS[:2], 0, None, limit=9)
    else:
        run(a.split, a.models.split(","), a.shots, a.out, a.limit)
