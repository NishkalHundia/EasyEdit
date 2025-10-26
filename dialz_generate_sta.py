#!/usr/bin/env python3
"""Generate STA vectors from dialz datasets."""

import json
import os
import random
from omegaconf import OmegaConf
from steer.vector_generators.vector_generators import BaseVectorGenerator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dialz_dataset(dataset_name, limit=None, seed=0):
    """Load dialz dataset and format for STA"""
    dataset_path = os.path.join(ROOT_DIR, "dialz_data/datasets/load", f"{dataset_name}.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Normalize
    normalized = []
    for item in raw_data:
        question = item.get("question", "")
        positive = item.get("positive", item.get("matching"))
        negative = item.get("negative", item.get("not_matching"))
        
        if positive and negative:
            normalized.append({
                "question": question,
                "matching": positive,
                "not_matching": negative
            })
    
    # Random sample if limit specified
    if limit and len(normalized) > limit:
        rng = random.Random(seed)
        normalized = rng.sample(normalized, k=limit)
    
    return normalized


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sae_width", default="16k", help="SAE width (16k or 131k)")
    parser.add_argument("--trim", type=float, default=0.65)
    args = parser.parse_args()
    
    # Determine SAE paths
    model_lower = args.model.lower()
    if "gemma" in model_lower:
        if "2-2b" in model_lower or "2b" in model_lower:
            repo = "gemma-scope-2b-pt-res-canonical"
        else:
            repo = "gemma-scope-9b-pt-res-canonical"
        
        sae_paths = []
        for layer in args.layers:
            if args.sae_width == "131k":
                repo_131k = repo.replace("-canonical", "")
                sae_paths.append(f"hugging_cache/{repo_131k}/layer_{layer}/width_131k/average_l0_71")
            else:
                sae_paths.append(f"hugging_cache/{repo}/layer_{layer}/width_16k/canonical")
    else:
        raise ValueError("STA currently only supports Gemma models")
    
    # Load training data
    print(f"Loading {args.dataset} dataset...")
    train_data = load_dialz_dataset(args.dataset, limit=args.train_limit, seed=args.seed)
    print(f"Loaded {len(train_data)} training examples")
    
    # Config for vector generation
    vector_base_dir = f"vectors/dialz/{args.model.split('/')[-1]}"
    vector_dir = f"{vector_base_dir}/{args.dataset}/sta_vector"
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
        "mode": "act_and_freq",
        "multiple_choice": False,
        "save_vectors": True,
        "steer_vector_output_dirs": [vector_base_dir],
        "steer_train_hparam_paths": ["hparams/Steer/sta_hparams/generate_sta.yaml"],
        "steer_train_dataset": [args.dataset],
    }
    
    # Generate vectors
    print("Generating STA vectors...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({args.dataset: train_data})
    print(f"\nVectors saved to {vector_dir}")
    print(f"\nTo apply these vectors, run:")
    print(f"python dialz_apply_sta.py --dataset {args.dataset} --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0 2.0")


if __name__ == "__main__":
    main()





