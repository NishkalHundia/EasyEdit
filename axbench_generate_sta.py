#!/usr/bin/env python3
"""Generate STA vectors from AxBench Concept500 using contrastive pairs.

Builds the same pairs as CAA and then invokes the STA generator with
GemmaScope SAE paths auto-inferred from model and layer selections.
"""

import os
import random
from typing import Dict, List, Tuple

from datasets import load_dataset
from omegaconf import OmegaConf

from steer.vector_generators.vector_generators import BaseVectorGenerator


def _build_contrastive_pairs(
    concept_id: int,
    train_rows: List[Dict],
    limit: int = None,
    seed: int = 0,
) -> List[Dict[str, str]]:
    neg_index: Dict[Tuple[str, str], List[Dict]] = {}
    for row in train_rows:
        if str(row.get("category", "")).lower() == "negative":
            key = (row.get("input", ""), row.get("concept_genre", ""))
            neg_index.setdefault(key, []).append(row)

    pairs: List[Dict[str, str]] = []
    for row in train_rows:
        try:
            cid = int(row.get("concept_id", -1))
        except Exception:
            cid = -1
        if cid != concept_id or str(row.get("category", "")).lower() != "positive":
            continue
        inp = row.get("input", "")
        genre = row.get("concept_genre", "")
        pos_out = row.get("output", "")
        if not inp:
            continue
        negs = neg_index.get((inp, genre), [])
        if not negs:
            continue
        neg_out = negs[0].get("output", "")
        if not neg_out:
            continue
        pairs.append({
            "question": inp,
            "matching": pos_out,
            "not_matching": neg_out,
        })

    if limit and len(pairs) > limit:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, k=limit)
    return pairs


def _infer_gemma_sae_paths(model: str, layers: List[int], sae_width: str) -> List[str]:
    model_lower = model.lower()
    paths = []
    for layer in layers:
        if "gemma" in model_lower:
            if sae_width == "131k":
                if "2-2b" in model_lower or "2b" in model_lower:
                    repo = "gemma-scope-2b-pt-res"
                else:
                    repo = "gemma-scope-9b-pt-res"
                paths.append(f"hugging_cache/{repo}/layer_{layer}/width_131k/average_l0_71")
            else:
                if "2-2b" in model_lower or "2b" in model_lower:
                    repo = "gemma-scope-2b-pt-res-canonical"
                else:
                    repo = "gemma-scope-9b-pt-res-canonical"
                paths.append(f"hugging_cache/{repo}/layer_{layer}/width_16k/canonical")
        else:
            raise ValueError("STA currently only supports Gemma models in this script")
    return paths


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sae_width", default="16k", choices=["16k", "131k"])
    parser.add_argument("--trim", type=float, default=0.65)
    parser.add_argument("--mode", default="act_and_freq")
    args = parser.parse_args()

    print("Loading Concept500 (train) ...")
    ds_train = load_dataset("pyvene/axbench-concept500", split="train", ignore_verifications=True)
    train_rows = [dict(r) for r in ds_train]

    pairs = _build_contrastive_pairs(
        concept_id=args.concept_id,
        train_rows=train_rows,
        limit=(None if args.train_limit in (None, 0) else args.train_limit),
        seed=args.seed,
    )
    print(f"Built {len(pairs)} contrastive pairs for concept_id={args.concept_id}")
    if not pairs:
        raise RuntimeError("No contrastive pairs found. Check concept_id or dataset filters.")

    sae_paths = _infer_gemma_sae_paths(args.model, args.layers, args.sae_width)

    model_tag = args.model.split("/")[-1]
    dataset_label = f"concept_{args.concept_id}"
    vector_base_dir = f"vectors/axbench/{model_tag}"

    gen_config = {
        "alg_name": "sta",
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "layers": args.layers,
        "sae_paths": sae_paths,
        "trims": [args.trim],
        "mode": args.mode,
        "multiple_choice": False,
        "save_vectors": True,
        "steer_vector_output_dirs": [vector_base_dir],
        "steer_train_hparam_paths": ["hparams/Steer/sta_hparams/generate_sta.yaml"],
        "steer_train_dataset": [dataset_label],
    }

    print("Generating STA vectors ...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({dataset_label: pairs})

    out_dir = os.path.join(vector_base_dir, dataset_label, "sta_vector")
    print(f"\nVectors saved to {out_dir}")
    layers_str = " ".join(map(str, args.layers))
    print("\nTo apply these vectors:")
    print(
        f"python EasyEdit/axbench_apply_sta.py --concept_id {args.concept_id} --model {args.model} --layers {layers_str} --multipliers 0.5 1.0 2.0"
    )


if __name__ == "__main__":
    main()


