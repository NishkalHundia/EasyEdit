#!/usr/bin/env python3
"""Simple CAA (Contrastive Activation Addition) script for Google Gemma 2 9B.

This script allows you to easily run CAA by specifying your own:
- Prompts/questions
- Matching responses (desired behavior)
- Non-matching responses (undesired behavior)

Usage:
    python simple_caa.py --examples examples.json --layers 20 --multipliers 1.0 2.0
    
Or use inline examples:
    python simple_caa.py --inline --layers 20
"""

import os
import json
import argparse
from typing import List, Dict
from omegaconf import OmegaConf

from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier


def load_examples_from_file(filepath: str) -> List[Dict[str, str]]:
    """Load contrastive examples from a JSON file.
    
    Expected format:
    [
        {
            "question": "Your prompt here",
            "matching": "Desired response",
            "not_matching": "Undesired response"
        },
        ...
    ]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    # Validate format
    for i, ex in enumerate(examples):
        if not all(k in ex for k in ['question', 'matching', 'not_matching']):
            raise ValueError(
                f"Example {i} missing required keys. "
                "Each example must have: 'question', 'matching', 'not_matching'"
            )
    
    return examples


def get_inline_examples() -> List[Dict[str, str]]:
    """Get examples through interactive input."""
    examples = []
    print("\n" + "="*60)
    print("Enter your contrastive examples")
    print("Press Ctrl+C when done adding examples")
    print("="*60 + "\n")
    
    while True:
        try:
            print(f"\n--- Example {len(examples) + 1} ---")
            question = input("Question/Prompt: ").strip()
            if not question:
                print("Question cannot be empty. Try again.")
                continue
                
            matching = input("Matching (desired) response: ").strip()
            if not matching:
                print("Matching response cannot be empty. Try again.")
                continue
                
            not_matching = input("Not matching (undesired) response: ").strip()
            if not not_matching:
                print("Not matching response cannot be empty. Try again.")
                continue
            
            examples.append({
                "question": question,
                "matching": matching,
                "not_matching": not_matching
            })
            
            print(f"✓ Added example {len(examples)}")
            
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break
    
    if not examples:
        print("No examples provided. Exiting.")
        exit(1)
    
    return examples


def generate_caa_vectors(
    examples: List[Dict[str, str]],
    model: str,
    device: str,
    layers: List[int],
    output_dir: str,
    dataset_label: str = "custom_caa"
):
    """Generate CAA vectors from contrastive examples."""
    
    print(f"\n{'='*60}")
    print(f"Generating CAA vectors")
    print(f"Model: {model}")
    print(f"Layers: {layers}")
    print(f"Examples: {len(examples)}")
    print(f"Output: {output_dir}")
    print('='*60 + "\n")
    
    # Configure vector generation
    gen_config = {
        "alg_name": "caa",
        "model_name_or_path": model,
        "device": device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "layers": layers,
        "save_vectors": True,
        "steer_vector_output_dirs": [output_dir],
        "steer_train_hparam_paths": ["hparams/Steer/caa_hparams/generate_caa.yaml"],
        "steer_train_dataset": [dataset_label],
    }
    
    # Generate vectors
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({dataset_label: examples})
    
    vector_path = os.path.join(output_dir, dataset_label, "caa_vector")
    print(f"\n✓ Vectors saved to: {vector_path}")
    return vector_path


def apply_caa_vectors(
    vector_path: str,
    model: str,
    device: str,
    layers: List[int],
    multipliers: List[float],
    test_prompts: List[str] = None
):
    """Apply CAA vectors to test prompts."""
    
    if test_prompts is None:
        test_prompts = [
            "Who is U2"
        ]
    
    print(f"\n{'='*60}")
    print(f"Applying CAA vectors")
    print(f"Vector path: {vector_path}")
    print(f"Multipliers: {multipliers}")
    print(f"Test prompts: {len(test_prompts)}")
    print('='*60 + "\n")
    
    # Prepare dataset for generation
    test_dataset = [{"input": p} for p in test_prompts]
    results_dir = os.path.join(vector_path, "results")
    
    # Baseline generation (no vectors applied)
    print("\n[Baseline (no steering)]")
    baseline_cfg = OmegaConf.create({
        "model_name_or_path": model,
        "device": device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "generation_output_dir": results_dir,
        "generation_data_size": None,
        "num_responses": 1,
        "generation_params": {"max_new_tokens": 128, "do_sample": True, "temperature": 0.7, "top_p": 0.9},
    })
    baseline_applier = BaseVectorApplier(baseline_cfg)
    baseline_results = baseline_applier.generate({"test": test_dataset}, save_results=False)
    for i, res in enumerate(baseline_results):
        print(f"\nPrompt {i+1}: {res['input']}")
        print(f"Baseline: {res['pred'][0] if res['pred'] else ''}")
    
    # Steered generations for each multiplier
    for mult in multipliers:
        print(f"\n[With CAA (multiplier={mult})]")
        apply_cfg = OmegaConf.create({
            "apply_steer_hparam_paths": ["hparams/Steer/caa_hparams/apply_caa.yaml"],
            "steer_vector_load_dir": [vector_path],
            "model_name_or_path": model,
            "device": device,
            "torch_dtype": "bfloat16",
            "use_chat_template": False,
            "system_prompt": "",
            "layers": layers,
            "multipliers": [mult],
            "generation_output_dir": results_dir,
            "generation_data_size": None,
            "num_responses": 1,
            "generation_params": {"max_new_tokens": 128, "do_sample": True, "temperature": 0.7, "top_p": 0.9},
        })
        applier = BaseVectorApplier(apply_cfg)
        applier.apply_vectors()
        steered_results = applier.generate({"test": test_dataset}, save_results=False)
        for i, res in enumerate(steered_results):
            print(f"\nPrompt {i+1}: {res['input']}")
            print(f"CAA x{mult}: {res['pred'][0] if res['pred'] else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple CAA script for Google Gemma 2 9B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a JSON file with examples
  python simple_caa.py --examples my_examples.json --layers 20 --multipliers 1.0 2.0
  
  # Enter examples interactively
  python simple_caa.py --inline --layers 20
  
  # Generate only (skip application)
  python simple_caa.py --examples my_examples.json --generate-only
  
  # Apply existing vectors
  python simple_caa.py --apply-only --vector-path vectors/custom/custom_caa/caa_vector --layers 20
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--examples",
        type=str,
        help="Path to JSON file with contrastive examples"
    )
    input_group.add_argument(
        "--inline",
        action="store_true",
        help="Enter examples interactively"
    )
    input_group.add_argument(
        "--apply-only",
        action="store_true",
        help="Only apply existing vectors (use with --vector-path)"
    )
    
    # Model options
    parser.add_argument(
        "--model",
        default="google/gemma-2-9b-it",
        help="HuggingFace model ID (default: google/gemma-2-9b-it)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[20],
        help="Layer(s) to apply CAA to (default: 20)"
    )
    
    # Generation options
    parser.add_argument(
        "--output-dir",
        default="vectors/custom",
        help="Output directory for vectors (default: vectors/custom)"
    )
    parser.add_argument(
        "--dataset-label",
        default="custom_caa",
        help="Label for this dataset (default: custom_caa)"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate vectors, don't apply them"
    )
    
    # Application options
    parser.add_argument(
        "--vector-path",
        type=str,
        help="Path to existing CAA vectors (required for --apply-only)"
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[1.0, 2.0],
        help="Steering multipliers to test (default: 1.0 2.0)"
    )
    parser.add_argument(
        "--test-prompts",
        nargs="+",
        type=str,
        help="Custom test prompts for application"
    )
    
    args = parser.parse_args()
    
    # Default behavior: if no input mode provided, try default examples file
    if not (args.examples or args.inline or args.apply_only):
        default_examples = os.path.join(os.path.dirname(__file__), "example_caa_data.json")
        if os.path.exists(default_examples):
            args.examples = default_examples
            print(f"No input flags provided. Using default examples file: {default_examples}")
        else:
            args.inline = True
            print("No input flags provided and no default examples found. Entering interactive mode.")
    
    # Validate arguments
    if args.apply_only and not args.vector_path:
        parser.error("--apply-only requires --vector-path")
    
    # Load or get examples
    if args.apply_only:
        examples = None
        vector_path = args.vector_path
    else:
        if args.examples:
            print(f"Loading examples from: {args.examples}")
            examples = load_examples_from_file(args.examples)
            print(f"Loaded {len(examples)} examples")
        else:
            examples = get_inline_examples()
        
        # Show examples summary
        print(f"\n{'='*60}")
        print("Examples Summary:")
        print('='*60)
        for i, ex in enumerate(examples[:3], 1):  # Show first 3
            print(f"\nExample {i}:")
            print(f"  Q: {ex['question'][:60]}...")
            print(f"  ✓ Matching: {ex['matching'][:50]}...")
            print(f"  ✗ Not matching: {ex['not_matching'][:50]}...")
        if len(examples) > 3:
            print(f"\n... and {len(examples) - 3} more examples")
        print('='*60)
        
        # Generate vectors
        vector_path = generate_caa_vectors(
            examples=examples,
            model=args.model,
            device=args.device,
            layers=args.layers,
            output_dir=args.output_dir,
            dataset_label=args.dataset_label
        )
    
    # Apply vectors
    if not args.generate_only:
        apply_caa_vectors(
            vector_path=vector_path,
            model=args.model,
            device=args.device,
            layers=args.layers,
            multipliers=args.multipliers,
            test_prompts=args.test_prompts
        )
    
    print("\n" + "="*60)
    print("✓ Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

