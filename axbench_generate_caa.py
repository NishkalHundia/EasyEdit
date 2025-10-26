#!/usr/bin/env python3
"""Generate CAA vectors from axbench-concept500 dataset."""

import os
import random
from omegaconf import OmegaConf
from steer.vector_generators.vector_generators import BaseVectorGenerator
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_axbench_concept(concept_id, limit=None, seed=0):
    """Load axbench-concept500 dataset and create contrastive pairs for a specific concept."""
    print(f"Loading axbench-concept500 dataset from HuggingFace...")
    
    # Load train split with explicit download mode
    try:
        dataset = load_dataset("pyvene/axbench-concept500", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading method...")
        dataset = load_dataset("pyvene/axbench-concept500", split="train", download_mode="force_redownload", trust_remote_code=True)
    
    # Find positive examples for this concept_id
    positive_examples = []
    concept_genre = None
    concept_name = None
    
    for item in dataset:
        if item['concept_id'] == concept_id and item['category'] == 'positive':
            positive_examples.append({
                'input': item['input'],
                'output': item['output'],
                'genre': item['concept_genre'],
                'concept': item['output_concept']
            })
            if concept_genre is None:
                concept_genre = item['concept_genre']
                concept_name = item['output_concept']
    
    if not positive_examples:
        raise ValueError(f"No positive examples found for concept_id {concept_id}")
    
    print(f"Found {len(positive_examples)} positive examples for concept '{concept_name}' (ID: {concept_id}, genre: {concept_genre})")
    
    # Create a mapping of input prompts from positive examples
    positive_inputs = {ex['input'] for ex in positive_examples}
    
    # Find matching negative examples (same genre, same input prompt)
    negative_lookup = {}
    for item in dataset:
        if (item['category'] == 'negative' and 
            item['concept_genre'] == concept_genre and
            item['input'] in positive_inputs):
            negative_lookup[item['input']] = item['output']
    
    print(f"Found {len(negative_lookup)} matching negative examples")
    
    # Create contrastive pairs
    contrastive_pairs = []
    for pos_ex in positive_examples:
        if pos_ex['input'] in negative_lookup:
            contrastive_pairs.append({
                "question": pos_ex['input'],
                "matching": pos_ex['output'],  # Contains the concept
                "not_matching": negative_lookup[pos_ex['input']]  # Doesn't contain concept
            })
    
    if not contrastive_pairs:
        raise ValueError(f"No contrastive pairs created for concept_id {concept_id}")
    
    print(f"Created {len(contrastive_pairs)} contrastive pairs")
    
    # Random sample if limit specified
    if limit and len(contrastive_pairs) > limit:
        rng = random.Random(seed)
        contrastive_pairs = rng.sample(contrastive_pairs, k=limit)
        print(f"Sampled {limit} pairs")
    
    return contrastive_pairs, concept_name, concept_genre


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True, help="Concept ID from axbench-concept500")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=None, help="Limit number of training examples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    # Load training data
    train_data, concept_name, concept_genre = load_axbench_concept(
        args.concept_id, 
        limit=args.train_limit, 
        seed=args.seed
    )
    
    dataset_name = f"concept_{args.concept_id}"
    
    # Config for vector generation
    vector_base_dir = f"vectors/axbench/{args.model.split('/')[-1]}"
    vector_dir = f"{vector_base_dir}/{dataset_name}/caa_vector"
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
        "steer_train_dataset": [dataset_name],
    }
    
    # Generate vectors
    print(f"\nGenerating CAA vectors for concept '{concept_name}' (genre: {concept_genre})...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({dataset_name: train_data})
    print(f"\nVectors saved to {vector_dir}")
    print(f"\nTo apply these vectors, run:")
    print(f"python axbench_apply_caa.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0 2.0")
    print(f"\nFor sanity check, run:")
    print(f"python axbench_sanity_check.py --concept_id {args.concept_id} --model {args.model} --method caa --layers {' '.join(map(str, args.layers))} --multipliers 1.0")


if __name__ == "__main__":
    main()

