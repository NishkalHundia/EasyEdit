#!/usr/bin/env python3
"""Sanity-check steering on training prompts used for vector calc.

Loads the same positive/negative-matched prompts from Concept500 train for a
given concept_id and runs generation with the trained vectors, to verify
steering on seen prompts.
"""

import os
import random
from typing import Dict, List, Tuple

from datasets import load_dataset
from omegaconf import OmegaConf

from steer.vector_appliers.vector_applier import BaseVectorApplier


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

    items: List[Dict[str, str]] = []
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
        # Return just the prompt for generation; we can include expected outputs for reference
        items.append({
            "input": inp,
            "reference_response": pos_out,
        })

    if limit and len(items) > limit:
        rng = random.Random(seed)
        items = rng.sample(items, k=limit)
    return items


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--method", choices=["caa", "sta"], required=True)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--trim", type=float, default=0.65)
    parser.add_argument("--mode", default="act_and_freq")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    print("Loading Concept500 (train) ...")
    ds_train = load_dataset("pyvene/axbench-concept500", split="train", verification_mode="no_checks")
    train_rows = [dict(r) for r in ds_train]
    items = _build_contrastive_pairs(args.concept_id, train_rows, limit=args.limit)
    if not items:
        raise RuntimeError("No training items found for given concept_id")

    model_tag = args.model.split("/")[-1]
    dataset_label = f"concept_{args.concept_id}"
    vector_dir = f"vectors/axbench/{model_tag}/{dataset_label}/{args.method}_vector"
    if not os.path.exists(vector_dir):
        raise FileNotFoundError(f"Vectors not found at {vector_dir}. Generate first.")

    apply_cfg = {
        "alg_name": args.method,
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "apply_steer_hparam_paths": [
            "hparams/Steer/caa_hparams/apply_caa.yaml" if args.method == "caa" else "hparams/Steer/sta_hparams/apply_sta.yaml"
        ],
        "steer_vector_load_dir": [vector_dir],
        "generation_data": ["train"],
        "generation_data_size": None,
        "generation_output_dir": f"generation/axbench/{model_tag}/{dataset_label}/sanity_{args.method}",
        "steer_from_end_position": True,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "vllm_enable": False,
    }

    cfg = OmegaConf.create(apply_cfg)
    applier = BaseVectorApplier(cfg)

    if args.method == "sta":
        applier.hparams_dict["sta"].mode = args.mode
        applier.hparams_dict["sta"].trims = [args.trim for _ in args.layers]
        applier.hparams_dict["sta"].multipliers = [args.multiplier for _ in args.layers]
    else:
        applier.hparams_dict["caa"].multipliers = [args.multiplier for _ in args.layers]

    applier.apply_vectors()
    applier.generate({"train": items})
    applier.model.reset_all()

    print(f"\nDone. Results saved to {apply_cfg['generation_output_dir']}")


if __name__ == "__main__":
    main()


