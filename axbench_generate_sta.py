#!/usr/bin/env python3
"""Generate STA vectors from axbench-concept500 dataset."""

import json
import os
import random
from omegaconf import OmegaConf
from steer.vector_generators.vector_generators import BaseVectorGenerator
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_concept_data(concept_id, limit=None, seed=0):
    """Load axbench-concept500 dataset and create contrastive pairs for a concept."""
    print(f"Loading axbench-concept500 dataset for concept_id={concept_id}...")
    
    # Load train split
    dataset = load_dataset("pyvene/axbench-concept500", split="train")
    
    # Find positive examples for this concept
    positive_examples = [ex for ex in dataset if ex['concept_id'] == concept_id and ex['category'] == 'positive']
    
    if not positive_examples:
        raise ValueError(f"No positive examples found for concept_id={concept_id}")
    
    # Get genre for this concept
    concept_genre = positive_examples[0]['concept_genre']
    concept_name = positive_examples[0]['output_concept']
    print(f"Concept: {concept_name}, Genre: {concept_genre}")
    print(f"Found {len(positive_examples)} positive examples")
    
    # Get all negative examples with same genre
    negative_examples = [
        ex for ex in dataset 
        if ex['category'] == 'negative' and ex['concept_genre'] == concept_genre
    ]
    print(f"Found {len(negative_examples)} negative examples in genre '{concept_genre}'")
    
    # Create mapping of input prompts to negative examples
    negative_by_input = {}
    for neg_ex in negative_examples:
        input_text = neg_ex['input']
        if input_text not in negative_by_input:
            negative_by_input[input_text] = neg_ex
    
    # Create contrastive pairs by matching positive and negative examples with same input
    contrastive_pairs = []
    for pos_ex in positive_examples:
        input_text = pos_ex['input']
        if input_text in negative_by_input:
            contrastive_pairs.append({
                "question": input_text,
                "matching": pos_ex['output'],
                "not_matching": negative_by_input[input_text]['output']
            })
    
    print(f"Created {len(contrastive_pairs)} contrastive pairs")
    
    if not contrastive_pairs:
        raise ValueError(f"No contrastive pairs found for concept_id={concept_id}")
    
    # Random sample if limit specified
    if limit and len(contrastive_pairs) > limit:
        rng = random.Random(seed)
        contrastive_pairs = rng.sample(contrastive_pairs, k=limit)
        print(f"Sampled {len(contrastive_pairs)} pairs (limit={limit})")
    
    return contrastive_pairs, concept_name


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", required=True, type=int, help="Concept ID from axbench-concept500")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=None, help="Limit number of training pairs")
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
    train_data, concept_name = load_concept_data(args.concept_id, limit=args.train_limit, seed=args.seed)
    
    # Config for vector generation
    dataset_name = f"concept_{args.concept_id}"
    vector_base_dir = f"vectors/axbench/{args.model.split('/')[-1]}"
    vector_dir = f"{vector_base_dir}/{dataset_name}/sta_vector"
    
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
        "steer_train_dataset": [dataset_name],
    }
    
    # Generate vectors
    print("\nGenerating STA vectors...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({dataset_name: train_data})
    
    # Save training pairs for sanity check
    sanity_dir = f"vectors/axbench/{args.model.split('/')[-1]}/{dataset_name}"
    os.makedirs(sanity_dir, exist_ok=True)
    with open(os.path.join(sanity_dir, "train_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Vectors saved to: {vector_dir}")
    print(f"Training pairs saved to: {sanity_dir}/train_pairs.json")
    print(f"Concept: {concept_name} (ID: {args.concept_id})")
    print(f"{'='*60}")
    print(f"\nTo apply these vectors, run:")
    print(f"python axbench_apply_sta.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0 2.0")
    print(f"\nTo run sanity check, run:")
    print(f"python axbench_sanity_check.py --concept_id {args.concept_id} --method sta --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0")


if __name__ == "__main__":
    main()


