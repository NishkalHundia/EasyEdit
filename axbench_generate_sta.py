#!/usr/bin/env python3
"""Generate STA vectors from AxBench Concept500 dataset.

Builds contrastive pairs identical to the CAA generator and then computes
STA vectors using GemmaScope SAEs. Mirrors dialz STA generator behavior.
"""

import os
import random
from typing import List, Dict
from datasets import load_dataset
from omegaconf import OmegaConf
from steer.vector_generators.vector_generators import BaseVectorGenerator


def build_contrastive_pairs(
    concept_id: int,
    split: str = "train",
    seed: int = 0,
    limit: int = None,
) -> List[Dict[str, str]]:
    # Load only the requested split
    dsplit = load_dataset("pyvene/axbench-concept500", split=split)

    pos_rows = [r for r in dsplit if r.get("concept_id") == concept_id and r.get("category") == "positive"]
    if not pos_rows:
        raise ValueError(f"No positive rows found for concept_id={concept_id} in split='{split}'")
    genre = pos_rows[0].get("concept_genre")

    neg_by_input = {}
    for r in dsplit:
        if r.get("concept_genre") == genre and r.get("category") == "negative":
            key = r.get("input", "")
            if key and key not in neg_by_input:
                neg_by_input[key] = r

    pairs: List[Dict[str, str]] = []
    for r in pos_rows:
        inp = r.get("input", "")
        if not inp:
            continue
        neg = neg_by_input.get(inp)
        if not neg:
            continue
        pairs.append({
            "question": inp,
            "matching": r.get("output", ""),
            "not_matching": neg.get("output", ""),
        })

    if not pairs:
        raise ValueError(f"No contrastive pairs found for concept_id={concept_id} (genre={genre})")

    if limit and len(pairs) > limit:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, k=limit)
    return pairs


def resolve_gemmascope_sae_paths(model_name: str, layers: List[int], sae_width: str) -> List[str]:
    model_lower = model_name.lower()
    if "gemma" not in model_lower:
        raise ValueError("STA currently only supports Gemma models")
    if "2-2b" in model_lower or "2b" in model_lower:
        repo = "gemma-scope-2b-pt-res-canonical"
    else:
        repo = "gemma-scope-9b-pt-res-canonical"

    sae_paths = []
    for layer in layers:
        if sae_width == "131k":
            repo_131k = repo.replace("-canonical", "")
            sae_paths.append(f"hugging_cache/{repo_131k}/layer_{layer}/width_131k/average_l0_71")
        else:
            sae_paths.append(f"hugging_cache/{repo}/layer_{layer}/width_16k/canonical")
    return sae_paths


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sae_width", default="16k", help="SAE width (16k or 131k)")
    parser.add_argument("--trim", type=float, default=0.65)
    parser.add_argument("--mode", default="act_and_freq", choices=["act_and_freq","only_act","only_freq"])
    args = parser.parse_args()

    print(f"Building contrastive pairs from Concept500 for concept_id={args.concept_id} split={args.split} ...")
    train_data = build_contrastive_pairs(args.concept_id, split=args.split, seed=args.seed, limit=args.train_limit)
    print(f"Built {len(train_data)} training pairs")

    sae_paths = resolve_gemmascope_sae_paths(args.model, args.layers, args.sae_width)

    model_tag = args.model.split("/")[-1]
    vector_base_dir = f"vectors/axbench/{model_tag}/concept_{args.concept_id}"
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
        "steer_train_dataset": [f"axbench_concept_{args.concept_id}"],
    }

    print("Generating STA vectors...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({f"axbench_concept_{args.concept_id}": train_data})
    print(f"\nVectors saved under {vector_base_dir}/sta_vector")
    print("\nTo apply these vectors, run:")
    print(
        f"python axbench_apply_sta.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0 2.0"
    )


if __name__ == "__main__":
    main()


