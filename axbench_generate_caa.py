#!/usr/bin/env python3
"""Generate CAA vectors from AxBench Concept500 dataset.

This mirrors the dialz CAA generator but builds contrastive pairs from
`pyvene/axbench-concept500` by:
- selecting positive rows for a given concept_id from the train split
- determining the concept's genre
- collecting negative rows of the same genre with identical input prompts
  whose outputs are negative (no concept) to form (prompt, matching, not_matching) pairs
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
    """Create contrastive pairs from Concept500 for a specific concept_id.

    Rules:
    - positives: rows with given concept_id and category == 'positive' in the given split
    - genre: inferred from positive rows' `concept_genre`
    - negatives: rows with same genre, same input text, category == 'negative'
    - pair each positive row to the negative row with exactly matching input
    """
    # Stream only the requested split to avoid preparing other splits
    stream = load_dataset("pyvene/axbench-concept500", split=split, streaming=True, verification_mode="no_checks")

    # First pass: gather positives and determine genre
    pos_by_input = {}
    genre = None
    for row in stream:
        if row.get("concept_id") == concept_id and row.get("category") == "positive":
            if genre is None:
                genre = row.get("concept_genre")
            inp = row.get("input", "")
            if inp and inp not in pos_by_input:
                pos_by_input[inp] = row.get("output", "")
                if limit and len(pos_by_input) >= limit:
                    break

    if not pos_by_input:
        raise ValueError(f"No positive rows found for concept_id={concept_id} in split='{split}'")

    # Second pass: gather negatives for those inputs with same genre
    stream2 = load_dataset("pyvene/axbench-concept500", split=split, streaming=True, verification_mode="no_checks")
    neg_by_input = {}
    for row in stream2:
        if row.get("concept_genre") == genre and row.get("category") == "negative":
            inp = row.get("input", "")
            if inp and inp in pos_by_input and inp not in neg_by_input:
                neg_by_input[inp] = row.get("output", "")

    # Build pairs in insertion order up to limit
    pairs: List[Dict[str, str]] = []
    for inp, pos_out in pos_by_input.items():
        neg_out = neg_by_input.get(inp)
        if not neg_out:
            continue
        pairs.append({
            "question": inp,
            "matching": pos_out,
            "not_matching": neg_out,
        })
        if limit and len(pairs) >= limit:
            break

    if not pairs:
        raise ValueError(f"No contrastive pairs found for concept_id={concept_id} (genre={genre})")

    if limit and len(pairs) > limit:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, k=limit)

    return pairs


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
    args = parser.parse_args()

    # Load training data (contrastive pairs)
    print(f"Building contrastive pairs from Concept500 for concept_id={args.concept_id} split={args.split} ...")
    train_data = build_contrastive_pairs(args.concept_id, split=args.split, seed=args.seed, limit=args.train_limit)
    print(f"Built {len(train_data)} training pairs")

    # Config for vector generation
    model_tag = args.model.split("/")[-1]
    vector_base_dir = f"vectors/axbench/{model_tag}/concept_{args.concept_id}"
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
        "steer_train_dataset": [f"axbench_concept_{args.concept_id}"],
    }

    # Generate vectors
    print("Generating CAA vectors...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({f"axbench_concept_{args.concept_id}": train_data})
    print(f"\nVectors saved under {vector_base_dir}/caa_vector")
    print("\nTo apply these vectors, run:")
    print(
        f"python axbench_apply_caa.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0 2.0"
    )


if __name__ == "__main__":
    main()


