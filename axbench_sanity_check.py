#!/usr/bin/env python3
"""Sanity check for axbench steering vectors using training prompts."""

import os
import random
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_training_prompts(concept_id, num_examples=5, seed=0):
    """Load training prompts used to create the steering vectors."""
    print(f"Loading axbench-concept500 train split from HuggingFace...")
    
    # Load only the train split
    dataset = load_dataset("pyvene/axbench-concept500", split="train")
    
    # Find positive examples for this concept_id
    positive_examples = []
    concept_genre = None
    concept_name = None
    
    for item in dataset:
        if item['concept_id'] == concept_id and item['category'] == 'positive':
            positive_examples.append({
                'input': item['input'],
                'matching_output': item['output'],
                'genre': item['concept_genre'],
                'concept': item['output_concept']
            })
            if concept_genre is None:
                concept_genre = item['concept_genre']
                concept_name = item['output_concept']
    
    if not positive_examples:
        raise ValueError(f"No positive examples found for concept_id {concept_id}")
    
    # Create a mapping of input prompts from positive examples
    positive_inputs = {ex['input'] for ex in positive_examples}
    
    # Find matching negative examples (same genre, same input prompt)
    for pos_ex in positive_examples:
        for item in dataset:
            if (item['category'] == 'negative' and 
                item['concept_genre'] == concept_genre and
                item['input'] == pos_ex['input']):
                pos_ex['not_matching_output'] = item['output']
                break
    
    # Filter to only pairs with both matching and not_matching
    contrastive_pairs = [ex for ex in positive_examples if 'not_matching_output' in ex]
    
    # Random sample
    if len(contrastive_pairs) > num_examples:
        rng = random.Random(seed)
        contrastive_pairs = rng.sample(contrastive_pairs, k=num_examples)
    
    print(f"Loaded {len(contrastive_pairs)} training prompt pairs for concept '{concept_name}' (genre: {concept_genre})")
    
    return contrastive_pairs, concept_name, concept_genre


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True, help="Concept ID from axbench-concept500")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=["caa", "sta"], required=True, help="Steering method to test")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.0, 1.0, 2.0], 
                       help="Multipliers to test (include 0.0 for baseline)")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_examples", type=int, default=5, help="Number of training prompts to test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    dataset_name = f"concept_{args.concept_id}"
    vector_dir = f"vectors/axbench/{args.model.split('/')[-1]}/{dataset_name}/{args.method}_vector"
    
    if not os.path.exists(vector_dir):
        print(f"Error: Vectors not found at {vector_dir}")
        print(f"Generate them first with:")
        print(f"python axbench_generate_{args.method}.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))}")
        return
    
    # Load training prompts
    training_pairs, concept_name, concept_genre = load_training_prompts(
        args.concept_id, 
        num_examples=args.num_examples,
        seed=args.seed
    )
    
    # Config for applying vectors
    hparam_path = f"hparams/Steer/{args.method}_hparams/apply_{args.method}.yaml"
    apply_config = {
        "alg_name": args.method,
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "apply_steer_hparam_paths": [hparam_path],
        "steer_vector_load_dir": [vector_dir],
        "generation_data": ["test"],
        "generation_data_size": None,
        "generation_output_dir": f"generation/axbench/{args.model.split('/')[-1]}/{dataset_name}/{args.method}_sanity",
        "steer_from_end_position": True,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "vllm_enable": False,
    }
    
    print(f"\n{'='*80}")
    print(f"SANITY CHECK: {args.method.upper()} Steering")
    print(f"Concept: '{concept_name}' (ID: {args.concept_id}, genre: {concept_genre})")
    print(f"Testing {len(training_pairs)} training prompts")
    print('='*80)
    
    apply_cfg = OmegaConf.create(apply_config)
    vector_applier = BaseVectorApplier(apply_cfg)
    
    for mult in args.multipliers:
        print(f"\n{'='*80}")
        print(f"Multiplier: {mult} {'(BASELINE - NO STEERING)' if mult == 0.0 else ''}")
        print('='*80)
        
        # Set multiplier
        vector_applier.hparams_dict[args.method].multipliers = [mult for _ in args.layers]
        vector_applier.apply_vectors()
        
        # Test on training prompts
        test_data = {"test": [{"input": pair['input']} for pair in training_pairs]}
        outputs = vector_applier.generate(test_data)
        
        # Display results
        print("\nResults on Training Prompts:")
        print("-" * 80)
        for i, (pair, output) in enumerate(zip(training_pairs, outputs if outputs else [])):
            print(f"\n[Training Pair {i+1}]")
            print(f"Prompt: {pair['input'][:100]}{'...' if len(pair['input']) > 100 else ''}")
            print(f"\nExpected (Matching - contains '{concept_name}'):")
            print(f"{pair['matching_output'][:150]}{'...' if len(pair['matching_output']) > 150 else ''}")
            print(f"\nExpected (Not Matching - no concept):")
            print(f"{pair['not_matching_output'][:150]}{'...' if len(pair['not_matching_output']) > 150 else ''}")
            if outputs:
                print(f"\nGenerated Output:")
                print(f"{output}")
            print("-" * 80)
        
        # Reset model
        vector_applier.model.reset_all()
    
    print(f"\n{'='*80}")
    print("SANITY CHECK COMPLETE")
    print(f"Expected behavior: Higher multipliers should steer outputs toward the concept.")
    print(f"Results saved to {apply_config['generation_output_dir']}")
    print('='*80)


if __name__ == "__main__":
    main()

