# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25", "requests"]
# ///
"""Normalizer and tokenizer type of the top LIMIT (10,000) Hub models by 30-day downloads.

Reads the start of each tokenizer.json and prints the results as gzip+base64 JSON in numbered
`RESULT` lines, decoded by analyze.py.
"""
import base64
import concurrent.futures as cf
import gzip
import json
import os
import time

import requests
from huggingface_hub import HfApi

LIMIT = int(os.environ.get("LIMIT", 10000))
TOKEN = os.environ.get("HF_TOKEN")
HEAD = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
DEC = json.JSONDecoder()


def signature(n):
    """A normalizer config → a short, stable description of the chain."""
    if n is None:
        return "none"
    t = n.get("type", "?")
    if t == "Sequence":
        return "Sequence[" + ", ".join(signature(x) for x in n.get("normalizers", [])) + "]"
    if t == "BertNormalizer":
        lower = n.get("lowercase", True)
        strip = n.get("strip_accents")
        strip = lower if strip is None else strip
        return f"BertNormalizer({'lowercase' if lower else 'cased'}{', strip accents' if strip else ''})"
    return t


MODEL_TYPE = __import__("re").compile(r'"model"\s*:\s*\{\s*"type"\s*:\s*"(\w+)"')


def model_type(text):
    """The tokenizer algorithm (BPE, WordPiece, Unigram, WordLevel): `"model": {"type": ...}`, or, in
    older files without "type", inferred from the fields that come before the vocabulary."""
    m = MODEL_TYPE.search(text)
    if m:
        return m.group(1)
    i = text.find('"model"')
    if i < 0:
        return None
    j = text.find('"vocab"', i)
    head = text[i:j if j > 0 else i + 2000]
    if '"max_input_chars_per_word"' in head:
        return "WordPiece"
    if '"dropout"' in head or '"fuse_unk"' in head or '"merges"' in head:
        return "BPE"
    if '"unk_id"' in head:
        return "Unigram"
    if j > 0 and '"unk_token"' in head:
        return "WordLevel"
    return None


def normalizer_of(repo):
    """→ (status, normalizer chain, tokenizer model type)."""
    url = f"https://huggingface.co/{repo}/resolve/main/tokenizer.json"
    for size in (2 << 20, 16 << 20, None):
        h = dict(HEAD, **({"Range": f"bytes=0-{size - 1}"} if size else {}))
        for attempt in range(6):
            r = requests.get(url, headers=h, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            break
        if r.status_code == 404:
            return "no tokenizer.json", None, None
        if r.status_code in (401, 403):
            return "gated", None, None
        if r.status_code not in (200, 206):
            return f"http {r.status_code}", None, None
        text = r.content.decode("utf-8", "ignore")
        i = text.find('"normalizer"')
        if i < 0:
            if r.status_code == 200:
                return "ok", "none", model_type(text)
            continue
        j = text.index(":", i) + 1
        while text[j] in " \n\r\t":
            j += 1
        try:
            value, _ = DEC.raw_decode(text, j)
        except ValueError:
            continue  # cut mid-value: read more
        kind = model_type(text)
        if kind is None and r.status_code == 206 and size is not None:
            continue  # "model" lies past this range: read more
        return "ok", signature(value), kind
    return "unparsed", None, None


def main():
    api = HfApi(token=TOKEN)
    models = list(api.list_models(sort="downloads", limit=LIMIT, expand=["downloads", "pipeline_tag", "library_name"]))
    print(f"listed {len(models)} models", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(16) as pool:
        futs = {pool.submit(normalizer_of, m.id): m for m in models}
        for k, f in enumerate(cf.as_completed(futs)):
            m = futs[f]
            try:
                status, chain, kind = f.result()
            except Exception as e:  # noqa: BLE001
                status, chain, kind = f"error {type(e).__name__}", None, None
            rows.append(dict(id=m.id, downloads=m.downloads or 0, pipeline=m.pipeline_tag, library=m.library_name,
                             status=status, chain=chain, model=kind))
            if k % 500 == 0:
                print(f"{k}/{len(models)}", flush=True)
    rows.sort(key=lambda r: -r["downloads"])
    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    print("== status", json.dumps(status), flush=True)
    # Log lines are capped at ~16 KB: the result goes out in numbered chunks.
    blob = base64.b64encode(gzip.compress(json.dumps(rows).encode())).decode()
    chunks = [blob[i:i + 8000] for i in range(0, len(blob), 8000)]
    for i, chunk in enumerate(chunks):
        print(f"RESULT\t{i}\t{len(chunks)}\t{chunk}", flush=True)
    print("== done", flush=True)


if __name__ == "__main__":
    main()
