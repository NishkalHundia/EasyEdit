#!/usr/bin/env python3
"""Sanity check: Apply steering vectors to the same training prompts they were trained on."""

import json
import os
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_train_pairs(concept_id, model):
    """Load the training pairs that were used to create the vectors."""
    train_pairs_path = f"vectors/axbench/{model.split('/')[-1]}/concept_{concept_id}/train_pairs.json"
    
    if not os.path.exists(train_pairs_path):
        raise FileNotFoundError(
            f"Training pairs not found at {train_pairs_path}. "
            f"Generate vectors first with axbench_generate_sta.py or axbench_generate_caa.py"
        )
    
    with open(train_pairs_path, 'r', encoding='utf-8') as f:
        train_pairs = json.load(f)
    
    return train_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", required=True, type=int, help="Concept ID from axbench-concept500")
    parser.add_argument("--method", required=True, choices=["sta", "caa"], help="Steering method")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--sample_limit", type=int, default=10, help="Limit number of training pairs to test")
    args = parser.parse_args()
    
    # Config for applying vectors
    dataset_name = f"concept_{args.concept_id}"
    vector_dir = f"vectors/axbench/{args.model.split('/')[-1]}/{dataset_name}/{args.method}_vector"
    
    if not os.path.exists(vector_dir):
        print(f"Error: Vectors not found at {vector_dir}")
        print(f"Generate them first with:")
        print(f"python axbench_generate_{args.method}.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))}")
        return
    
    # Load training pairs
    print(f"Loading training pairs for concept_id={args.concept_id}...")
    train_pairs = load_train_pairs(args.concept_id, args.model)
    
    # Sample if needed
    if args.sample_limit and len(train_pairs) > args.sample_limit:
        train_pairs = train_pairs[:args.sample_limit]
    
    print(f"Testing on {len(train_pairs)} training examples (sanity check)\n")
    
    # Prepare apply config
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
        "generation_data": ["sanity"],
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
    
    # Create output directory
    output_dir = apply_config['generation_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Method: {args.method.upper()}")
    print(f"Multipliers: {args.multipliers}")
    print(f"Layers: {args.layers}\n")
    
    apply_cfg = OmegaConf.create(apply_config)
    vector_applier = BaseVectorApplier(apply_cfg)
    
    all_results = []
    
    for mult in args.multipliers:
        print(f"\n{'='*80}")
        print(f"Multiplier: {mult}")
        print('='*80)
        
        # Set multiplier
        vector_applier.hparams_dict[args.method].multipliers = [mult for _ in args.layers]
        vector_applier.apply_vectors()
        
        # Generate for each training pair
        results = []
        for idx, pair in enumerate(train_pairs):
            question = pair['question']
            matching = pair['matching']
            not_matching = pair['not_matching']
            
            print(f"\n[{idx+1}/{len(train_pairs)}] Question: {question[:80]}...")
            
            # Generate with steering
            test_batch = {"sanity": [{"input": question}]}
            outputs = vector_applier.generate(test_batch)
            
            generated = outputs['sanity'][0]['output'] if outputs and 'sanity' in outputs else ""
            
            result = {
                "index": idx,
                "multiplier": mult,
                "question": question,
                "matching_output": matching,
                "not_matching_output": not_matching,
                "generated_output": generated,
            }
            results.append(result)
            
            print(f"Matching (expected): {matching[:80]}...")
            print(f"Not matching: {not_matching[:80]}...")
            print(f"Generated: {generated[:80]}...")
        
        # Save results for this multiplier
        mult_str = str(mult).replace('.', '_')
        output_file = os.path.join(output_dir, f"sanity_mult_{mult_str}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
        all_results.extend(results)
        
        # Reset model
        vector_applier.model.reset_all()
    
    # Save all results
    all_results_file = os.path.join(output_dir, "sanity_all_results.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Sanity check complete!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}")
    print("\nThis sanity check tests if steering works on the same prompts used for training.")
    print("If steering is effective, the generated outputs should be closer to 'matching_output'.")


if __name__ == "__main__":
    main()


