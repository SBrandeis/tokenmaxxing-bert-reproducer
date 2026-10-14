#!/usr/bin/env python3
"""Share of ASCII bytes and characters in each benchmark corpus → ascii_share.json."""
import json
import pathlib
import subprocess
import tempfile
import unicodedata

REVISION = "5f5c6524084c42a35c28c0184bea37ec2530fd7c"
FILES = {"eng_Latn": "fixtures/lang/eng_Latn.txt", "code_mixed": "fixtures/modalities/code_mixed.txt",
         "rus_Cyrl": "fixtures/lang/rus_Cyrl.txt", "ell_Grek": "fixtures/lang/ell_Grek.txt",
         "arb_Arab": "fixtures/lang/arb_Arab.txt", "cmn_Hani": "fixtures/lang/cmn_Hani.txt"}

out = {"revision": REVISION}
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["hf", "download", "hf-internal-testing/tokenizers-test-data", *FILES.values(), "--repo-type", "dataset",
                    "--revision", REVISION, "--local-dir", tmp], check=True, capture_output=True)
    for name, rel in FILES.items():
        text = (pathlib.Path(tmp) / rel).read_text()
        texts = {name: text}
        if name == "ell_Grek":
            texts["ell_Grek_nfd"] = unicodedata.normalize("NFD", text)
        for n, t in texts.items():
            b = t.encode()
            out[n] = {"ascii_bytes": round(sum(x < 0x80 for x in b) / len(b), 4),
                      "ascii_chars": round(sum(c.isascii() for c in t) / len(t), 4), "bytes": len(b)}
path = pathlib.Path(__file__).parent / "ascii_share.json"
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
