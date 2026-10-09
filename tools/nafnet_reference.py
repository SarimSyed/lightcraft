#!/usr/bin/env python3
"""Development only: convert the author checkpoint and produce independent reference tensors.

Install torch/numpy/safetensors in a venv. Arguments: checkpoint, pinned upstream
NAFNet_arch.py, pinned arch_util.py, output directory. No product Python dependency.
"""
import ast
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import save_file

REVISION = "2b4af71ebe098a92a75910c233a3965a3e93ede4"
SOURCE_HASH = "89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4"


def main():
    checkpoint, network, util, output = map(pathlib.Path, sys.argv[1:])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == SOURCE_HASH
    assert hashlib.sha256(network.read_bytes()).hexdigest() == "01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967"
    assert hashlib.sha256(util.read_bytes()).hexdigest() == "5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005"
    output.mkdir(parents=True, exist_ok=True)
    scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    # Execute the author's normalization and network, without BasicSR's training dependencies.
    nodes = [n for n in ast.parse(util.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in ("LayerNormFunction", "LayerNorm2d")]
    nodes += [n for n in ast.parse(network.read_text()).body
              if isinstance(n, ast.ClassDef) and n.name in ("SimpleGate", "NAFBlock", "NAFNet")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(network), "exec"), scope)
    model = scope["NAFNet"](width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).eval()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["params"]
    model.load_state_dict(weights, strict=True)
    bundle = output / "nafnet-sidd-width32-v1.safetensors"
    header, chunks, offset = {}, [], 0
    for key, value in sorted(weights.items()):
        data = value.contiguous().numpy().astype("<f4").tobytes()
        header[key] = {"dtype": "F32", "shape": list(value.shape), "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    header["__metadata__"] = {"upstream_revision": REVISION, "source_sha256": SOURCE_HASH}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    bundle.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"".join(chunks))
    torch.set_num_threads(4)
    refs = {}
    with torch.no_grad():
        for name, h, w in [("tiny", 1, 1), ("padding", 17, 29), ("tiles", 193, 257)]:
            y, x = np.indices((h, w))
            clean = np.stack([.25 + x / w * .25, .35 + y / h * .1, .45 + x / w * .1], axis=-1).astype(np.float32)
            # Fixed procedural noise, identical across languages via stored public input.
            noise = (((x * 17 + y * 31) % 23) - 11).astype(np.float32)[..., None] * .004
            noisy = np.clip(clean + noise, 0, 1)
            inp = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).contiguous()
            refs[name + ".input"] = inp
            refs[name + ".clean"] = torch.from_numpy(clean).permute(2, 0, 1).unsqueeze(0).contiguous()
            if name != "tiles":
                refs[name + ".output"] = model(inp).contiguous()
                continue
            result = torch.zeros_like(inp)
            normalization = torch.zeros((1, 1, h, w))
            for sy in range(0, h, 192):
                for sx in range(0, w, 192):
                    th, tw = min(256, h - sy), min(256, w - sx)
                    tile = torch.zeros((1, 3, 256, 256))
                    tile[:, :, :th, :tw] = inp[:, :, sy:sy + th, sx:sx + tw]
                    prediction = model(tile)[:, :, :th, :tw]
                    taper = np.minimum(np.minimum((np.arange(256) + .5) / 64, (255.5 - np.arange(256)) / 64), 1).astype(np.float32)
                    weight = torch.from_numpy(taper[:th, None] * taper[None, :tw])[None, None]
                    result[:, :, sy:sy + th, sx:sx + tw] += prediction * weight
                    normalization[:, :, sy:sy + th, sx:sx + tw] += weight
            refs[name + ".output"] = (result / normalization).contiguous()
    save_file(refs, str(output / "reference.safetensors"))
    provenance = {"model": "nafnet-sidd-width32", "upstream_revision": REVISION,
                  "source_url": "https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view",
                  "source_sha256": SOURCE_HASH, "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                  "bundle_size": bundle.stat().st_size, "torch": torch.__version__,
                  "conversion": "params, unchanged names and F32 values; safetensors; ordinary NAFNet global pooling"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
