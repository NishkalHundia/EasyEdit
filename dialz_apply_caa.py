#!/usr/bin/env python3
"""Apply CAA vectors to generate text."""

import os
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--test_prompt", default="I think that")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()
    
    # Config for applying vectors
    vector_dir = f"vectors/dialz/{args.model.split('/')[-1]}/{args.dataset}/caa_vector"
    
    if not os.path.exists(vector_dir):
        print(f"Error: Vectors not found at {vector_dir}")
        print(f"Generate them first with:")
        print(f"python dialz_generate_caa.py --dataset {args.dataset} --model {args.model} --layers {' '.join(map(str, args.layers))}")
        return
    
    apply_config = {
        "alg_name": "caa",
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "apply_steer_hparam_paths": ["hparams/Steer/caa_hparams/apply_caa.yaml"],
        "steer_vector_load_dir": [vector_dir],
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
    print(f"Testing with multipliers: {args.multipliers}")
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





