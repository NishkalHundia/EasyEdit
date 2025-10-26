#!/usr/bin/env python3
"""Generate CAA vectors from AxBench Concept500 using contrastive pairs.

This mirrors the dialz CAA generation flow but builds pairs from
pyvene/axbench-concept500 train split by concept_id and genre, pairing
positives with exact-input-matched negatives.
"""

import os
import random
from typing import Dict, List, Tuple

import pandas as pd
from omegaconf import OmegaConf

from steer.vector_generators.vector_generators import BaseVectorGenerator


def _build_contrastive_pairs(
    concept_id: int,
    train_rows: List[Dict],
    limit: int = None,
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Construct contrastive pairs for a concept.

    For each positive row with the given concept_id, find a negative row
    with the same concept_genre and the exact same input. Return a list of
    dicts with keys: question, matching, not_matching.
    """
    # Index negatives by (input, concept_genre)
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
        if cid != concept_id:
            continue
        if str(row.get("category", "")).lower() != "positive":
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True, help="Concept ID to train on")
    parser.add_argument("--model", default="google/gemma-2-9b-it", help="HF model id/path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=720, help="Max training pairs; 0 means all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading Concept500 (train) ...")
    # Download train parquet directly from HF to avoid multi-split schema conflicts
    import tempfile
    import subprocess
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    train_cache_path = os.path.join(cache_dir, "pyvene___axbench-concept500", "default", "train", "0.0.0", "cache-train.arrow")
    
    # Check if already cached
    if os.path.exists(train_cache_path):
        print("Using cached train data...")
        from datasets import load_from_disk
        ds_train = load_from_disk(os.path.dirname(train_cache_path))
        train_rows = [dict(r) for r in ds_train]
    else:
        print("Downloading train split...")
        # Direct download from HF structure: /9b/l20/train/data.parquet
        train_url = "https://huggingface.co/datasets/pyvene/axbench-concept500/resolve/main/9b/l20/train/data.parquet"
        tmp_path = tempfile.mktemp(suffix=".parquet")
        subprocess.run(["wget", "-q", "-O", tmp_path, train_url], check=True)
        train_rows = pd.read_parquet(tmp_path).to_dict('records')
        os.unlink(tmp_path)

    pairs = _build_contrastive_pairs(
        concept_id=args.concept_id,
        train_rows=train_rows,
        limit=(None if args.train_limit in (None, 0) else args.train_limit),
        seed=args.seed,
    )
    print(f"Built {len(pairs)} contrastive pairs for concept_id={args.concept_id}")
    if not pairs:
        raise RuntimeError("No contrastive pairs found. Check concept_id or dataset filters.")

    # Configure vector generation
    model_tag = args.model.split("/")[-1]
    dataset_label = f"concept_{args.concept_id}"
    vector_base_dir = f"vectors/axbench/{model_tag}"
    gen_config = {
        "alg_name": "caa",
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "layers": args.layers,
        "save_vectors": True,
        "steer_vector_output_dirs": [vector_base_dir],
        "steer_train_hparam_paths": ["hparams/Steer/caa_hparams/generate_caa.yaml"],
        "steer_train_dataset": [dataset_label],
    }

    print("Generating CAA vectors ...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({dataset_label: pairs})

    out_dir = os.path.join(vector_base_dir, dataset_label, "caa_vector")
    print(f"\nVectors saved to {out_dir}")
    layers_str = " ".join(map(str, args.layers))
    print("\nTo apply these vectors:")
    print(
        f"python EasyEdit/axbench_apply_caa.py --concept_id {args.concept_id} --model {args.model} --layers {layers_str} --multipliers 0.5 1.0 2.0"
    )


if __name__ == "__main__":
    main()


