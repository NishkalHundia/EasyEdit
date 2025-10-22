#!/usr/bin/env python3
"""
Simple CAA steering using dialz datasets.
No complexity, just works.
"""

import json
import os
import random
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.vector_generators.vector_generators import BaseVectorGenerator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dialz_dataset(dataset_name, limit=None, seed=0):
    """Load dialz dataset and format for CAA"""
    dataset_path = os.path.join(ROOT_DIR, "dialz_data/datasets/load", f"{dataset_name}.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Normalize to CAA format
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
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., sycophancy, morality)")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=300)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[-1, 0, 1])
    parser.add_argument("--test_prompt", default="I think that")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading {args.dataset} dataset...")
    train_data = load_dialz_dataset(args.dataset, limit=args.train_limit, seed=0)
    print(f"Loaded {len(train_data)} training examples")
    
    # Config for vector generation
    gen_config = {
        "alg_name": "caa",
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "layers": args.layers,
        "save_vectors": True,
        "steer_vector_output_dir": f"vectors/dialz/{args.model.split('/')[-1]}/{args.dataset}/caa_vector",
        "steer_train_hparam_paths": ["hparams/Steer/caa_hparams/generate_caa.yaml"],
    }
    
    # Generate vectors
    print("Generating CAA vectors...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({args.dataset: train_data})
    print(f"Vectors saved to {gen_config['steer_vector_output_dir']}")
    
    # Config for applying vectors
    apply_config = {
        "alg_name": "caa",
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "apply_steer_hparam_paths": ["hparams/Steer/caa_hparams/apply_caa.yaml"],
        "steer_vector_load_dir": [gen_config["steer_vector_output_dir"]],
        "generation_data": ["test"],
        "generation_data_size": None,
        "generation_output_dir": f"generation/dialz/{args.model.split('/')[-1]}/{args.dataset}/caa",
        "steer_from_end_position": True,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "vllm_enable": False,
    }
    
    # Apply vectors with different multipliers
    print(f"\nTesting with multipliers: {args.multipliers}")
    apply_cfg = OmegaConf.create(apply_config)
    vector_applier = BaseVectorApplier(apply_cfg)
    
    for mult in args.multipliers:
        print(f"\n{'='*60}")
        print(f"Multiplier: {mult}")
        print('='*60)
        
        # Set multiplier
        vector_applier.hparams_dict["caa"].multipliers = [mult for _ in args.layers]
        vector_applier.apply_vectors()
        
        # Test prompt
        test_data = {"test": [{"input": args.test_prompt}]}
        vector_applier.generate(test_data)
        
        # Reset model
        vector_applier.model.reset_all()
    
    print(f"\nDone! Results saved to {apply_config['generation_output_dir']}")


if __name__ == "__main__":
    main()

