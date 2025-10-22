import argparse
import json
import os
from typing import Dict, List

from omegaconf import OmegaConf

from steer.vector_appliers.vector_applier import BaseVectorApplier


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def load_test_prompts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list) and obj and isinstance(obj[0], str):
        return [{"input": s} for s in obj]
    return [{"input": x.get("input", "")} for x in obj]


def main():
    parser = argparse.ArgumentParser(description="Apply steering vectors with multiple multipliers and generate outputs.")
    parser.add_argument("--method", choices=["caa", "sta"], required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name used in vectors path")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--multipliers", nargs="+", type=float, required=True, help="List of multipliers, e.g., 0.5 1.0 2.0")
    parser.add_argument("--test_prompts", required=True, help="JSON file with test prompts (list[str] or list[{input: str}])")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    vector_dir = os.path.join(
        "vectors", "dialz", "gemma-2-9b", args.dataset, f"{args.method}_vector"
    )

    apply_hparam_path = f"hparams/Steer/{args.method}_hparams/apply_{args.method}.yaml"

    test_entries = load_test_prompts(args.test_prompts)

    base_output_dir = os.path.join(
        "generation", "dialz", "gemma-2-9b", args.dataset, args.method
    )

    for mult in args.multipliers:
        top_cfg = OmegaConf.create({
            "model_name_or_path": args.model,
            "torch_dtype": "bfloat16",
            "device": args.device,
            "seed": 0,
            "use_chat_template": False,
            "system_prompt": "",
            "apply_steer_hparam_paths": [apply_hparam_path],
            "steer_vector_load_dir": [vector_dir],
            "generation_data": ["dialz_test"],
            "generation_data_size": None,
            "generation_output_dir": os.path.join(base_output_dir, f"mult_{mult}"),
            "num_responses": 1,
            "steer_from_end_position": False,
            "generate_orig_output": True,
            "generation_params": {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "temperature": args.temperature,
            },
            "vllm_enable": False,
        })

        # Override multipliers dynamically by patching hparams for the method
        # BaseVectorApplier reads apply hparams and then loads vectors
        applier = BaseVectorApplier(top_cfg)
        # Patch the multiplier on the loaded hparams
        for k, hp in applier.hparams_dict.items():
            if hasattr(hp, "multipliers"):
                # broadcast scalar to match layers length
                if isinstance(hp.layers, list) and hp.layers:
                    hp.multipliers = [mult for _ in hp.layers]
                else:
                    hp.multipliers = [mult]

        applier.apply_vectors()
        applier.generate({"dialz_test": test_entries})
        if applier.model is not None:
            applier.model.reset_all()

    print(f"Finished for multipliers: {args.multipliers}. Results under {base_output_dir}/mult_*/")


if __name__ == "__main__":
    main()


