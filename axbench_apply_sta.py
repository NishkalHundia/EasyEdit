#!/usr/bin/env python3
"""Apply STA vectors and generate on axbench-concept500 test data."""

import json
import os
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_test_data(concept_id):
    """Load test data for a specific concept from axbench-concept500."""
    print(f"Loading test data for concept_id={concept_id}...")
    
    # Load entire dataset first to avoid schema mismatch issues between splits
    full_dataset = load_dataset("pyvene/axbench-concept500")
    dataset = full_dataset["test"]
    
    # Find all examples for this concept (positive ones)
    test_examples = [
        ex for ex in dataset 
        if ex['concept_id'] == concept_id and ex['category'] == 'positive'
    ]
    
    if not test_examples:
        raise ValueError(f"No test examples found for concept_id={concept_id}")
    
    # Get concept name and genre
    concept_name = test_examples[0]['output_concept']
    concept_genre = test_examples[0]['concept_genre']
    
    print(f"Concept: {concept_name}, Genre: {concept_genre}")
    print(f"Found {len(test_examples)} test examples")
    
    # Format for generation
    formatted_data = []
    for ex in test_examples:
        formatted_data.append({
            "input": ex['input'],
            "expected_output": ex['output']
        })
    
    return formatted_data, concept_name


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", required=True, type=int, help="Concept ID from axbench-concept500")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--test_limit", type=int, default=None, help="Limit number of test examples")
    args = parser.parse_args()
    
    # Config for applying vectors
    dataset_name = f"concept_{args.concept_id}"
    vector_dir = f"vectors/axbench/{args.model.split('/')[-1]}/{dataset_name}/sta_vector"
    
    if not os.path.exists(vector_dir):
        print(f"Error: Vectors not found at {vector_dir}")
        print(f"Generate them first with:")
        print(f"python axbench_generate_sta.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))}")
        return
    
    # Load test data
    test_data, concept_name = load_test_data(args.concept_id)
    
    if args.test_limit:
        test_data = test_data[:args.test_limit]
    
    print(f"Testing on {len(test_data)} examples\n")
    
    apply_config = {
        "alg_name": "sta",
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "apply_steer_hparam_paths": ["hparams/Steer/sta_hparams/apply_sta.yaml"],
        "steer_vector_load_dir": [vector_dir],
        "generation_data": ["test"],
        "generation_data_size": None,
        "generation_output_dir": f"generation/axbench/{args.model.split('/')[-1]}/{dataset_name}/sta",
        "steer_from_end_position": True,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "vllm_enable": False,
    }
    
    # Create output directory
    output_dir = apply_config['generation_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing with multipliers: {args.multipliers}")
    print(f"Concept: {concept_name} (ID: {args.concept_id})")
    print(f"Layers: {args.layers}\n")
    
    apply_cfg = OmegaConf.create(apply_config)
    vector_applier = BaseVectorApplier(apply_cfg)
    
    all_results = []
    
    for mult in args.multipliers:
        print(f"\n{'='*80}")
        print(f"Multiplier: {mult}")
        print('='*80)
        
        # Set multiplier
        vector_applier.hparams_dict["sta"].multipliers = [mult for _ in args.layers]
        vector_applier.apply_vectors()
        
        # Generate for each test example
        results = []
        for idx, example in enumerate(test_data):
            print(f"\n[{idx+1}/{len(test_data)}] Input: {example['input'][:100]}...")
            
            # Generate with steering
            test_batch = {"test": [{"input": example['input']}]}
            outputs = vector_applier.generate(test_batch)
            
            generated = outputs['test'][0]['output'] if outputs and 'test' in outputs else ""
            
            result = {
                "index": idx,
                "multiplier": mult,
                "input": example['input'],
                "expected_output": example['expected_output'],
                "generated_output": generated,
            }
            results.append(result)
            
            print(f"Expected: {example['expected_output'][:100]}...")
            print(f"Generated: {generated[:100]}...")
        
        # Save results for this multiplier
        mult_str = str(mult).replace('.', '_')
        output_file = os.path.join(output_dir, f"results_mult_{mult_str}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
        all_results.extend(results)
        
        # Reset model
        vector_applier.model.reset_all()
    
    # Save all results
    all_results_file = os.path.join(output_dir, "all_results.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


