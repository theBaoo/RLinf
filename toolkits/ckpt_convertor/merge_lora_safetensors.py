#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge LoRA tensors in a safetensors checkpoint into full weights.

This utility is designed for checkpoints where keys may look like:
- base_model.model.xxx.base_layer.weight
- base_model.model.xxx.lora_A.default.weight
- base_model.model.xxx.lora_B.default.weight

It can optionally strip a top-level prefix (e.g. "base_model.model.")
and writes a new safetensors file plus an HF-style index json.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import safetensors.torch
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA safetensors checkpoint")
    parser.add_argument("--input", required=True, help="Path to input .safetensors")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write merged checkpoint files",
    )
    parser.add_argument(
        "--output-name",
        default="model-00001-of-00001.safetensors",
        help="Output safetensors filename",
    )
    parser.add_argument(
        "--strip-prefix",
        default="",
        help="Strip this prefix from every output key, e.g. base_model.model.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=1.0,
        help="LoRA alpha used for scaling",
    )
    parser.add_argument(
        "--lora-rank",
        type=float,
        default=1.0,
        help="LoRA rank r used for scaling alpha/r",
    )
    parser.add_argument(
        "--keep-lora-keys",
        action="store_true",
        help="Keep original LoRA keys in output (default: drop them)",
    )
    return parser.parse_args()


def maybe_strip_prefix(key: str, prefix: str) -> str:
    if prefix and key.startswith(prefix):
        return key[len(prefix) :]
    return key


def normalize_base_layer_key(key: str) -> str:
    if key.endswith(".base_layer.weight"):
        return key[: -len(".base_layer.weight")] + ".weight"
    if key.endswith(".base_layer.bias"):
        return key[: -len(".base_layer.bias")] + ".bias"
    return key


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    src = safetensors.torch.load_file(args.input)

    lora_a = {}
    lora_b = {}
    base_layer_w = {}

    for key, tensor in src.items():
        if key.endswith(".lora_A.default.weight"):
            stem = key[: -len(".lora_A.default.weight")]
            lora_a[stem] = tensor
        elif key.endswith(".lora_B.default.weight"):
            stem = key[: -len(".lora_B.default.weight")]
            lora_b[stem] = tensor
        elif key.endswith(".base_layer.weight"):
            stem = key[: -len(".base_layer.weight")]
            base_layer_w[stem] = tensor

    out = {}
    scale = args.lora_alpha / args.lora_rank

    merged_count = 0
    missing_a_or_b = defaultdict(list)

    for key, tensor in src.items():
        # Optionally skip raw LoRA tensors.
        if not args.keep_lora_keys and (
            key.endswith(".lora_A.default.weight")
            or key.endswith(".lora_B.default.weight")
        ):
            continue

        if key.endswith(".base_layer.weight"):
            stem = key[: -len(".base_layer.weight")]
            out_key = maybe_strip_prefix(stem + ".weight", args.strip_prefix)

            if stem in lora_a and stem in lora_b:
                a = lora_a[stem].float()
                b = lora_b[stem].float()
                base = tensor.float()
                delta = torch.matmul(b, a) * scale
                merged = (base + delta).to(dtype=tensor.dtype)
                out[out_key] = merged
                merged_count += 1
            else:
                # Fall back to base layer weight if A/B pair is incomplete.
                if stem not in lora_a:
                    missing_a_or_b[stem].append("A")
                if stem not in lora_b:
                    missing_a_or_b[stem].append("B")
                out[out_key] = tensor
            continue

        if key.endswith(".base_layer.bias"):
            out_key = maybe_strip_prefix(normalize_base_layer_key(key), args.strip_prefix)
            out[out_key] = tensor
            continue

        out_key = maybe_strip_prefix(key, args.strip_prefix)
        out[out_key] = tensor

    output_st = os.path.join(args.output_dir, args.output_name)
    safetensors.torch.save_file(out, output_st)

    # Build HF-style index for compatibility.
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in out.values())},
        "weight_map": {k: args.output_name for k in out.keys()},
    }
    index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Input keys: {len(src)}")
    print(f"Output keys: {len(out)}")
    print(f"Merged LoRA layers: {merged_count}")
    if missing_a_or_b:
        print("Layers with incomplete LoRA pairs (kept base_layer.weight):")
        for stem, miss in missing_a_or_b.items():
            print(f"  - {stem}: missing {','.join(miss)}")
    print(f"Wrote: {output_st}")
    print(f"Wrote: {index_path}")


if __name__ == "__main__":
    main()

