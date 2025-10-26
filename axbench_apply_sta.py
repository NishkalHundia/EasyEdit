#!/usr/bin/env python3
"""Apply STA vectors trained on AxBench Concept500 to test prompts.

Loads test split for the specified concept_id and runs generation while
applying STA vectors. Supports multiple multipliers and trims.
"""

import os
from typing import Dict, List

from datasets import load_dataset
from omegaconf import OmegaConf

from steer.vector_appliers.vector_applier import BaseVectorApplier


def _build_test_inputs(concept_id: int, test_rows: List[Dict]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for row in test_rows:
        try:
            cid = int(row.get("concept_id", -1))
        except Exception:
            cid = -1
        if cid != concept_id:
            continue
        inp = row.get("input", "")
        if not inp:
            continue
        items.append({
            "input": inp,
            "reference_response": row.get("output", ""),
        })
    return items


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--trim", type=float, default=0.65)
    parser.add_argument("--mode", default="act_and_freq")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    print("Loading Concept500 (test) ...")
    ds_test = load_dataset("pyvene/axbench-concept500", split="test", verification_mode="no_checks")
    test_rows = [dict(r) for r in ds_test]
    items = _build_test_inputs(args.concept_id, test_rows)
    if not items:
        raise RuntimeError("No test items found for given concept_id")

    model_tag = args.model.split("/")[-1]
    dataset_label = f"concept_{args.concept_id}"
    vector_dir = f"vectors/axbench/{model_tag}/{dataset_label}/sta_vector"
    if not os.path.exists(vector_dir):
        raise FileNotFoundError(f"Vectors not found at {vector_dir}. Generate first.")

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
        "generation_output_dir": f"generation/axbench/{model_tag}/{dataset_label}/sta",
        "steer_from_end_position": True,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "vllm_enable": False,
    }

    print(f"Testing with multipliers: {args.multipliers}")
    apply_cfg = OmegaConf.create(apply_config)
    applier = BaseVectorApplier(apply_cfg)

    # Set mode/trim for applier hyperparams
    applier.hparams_dict["sta"].mode = args.mode
    applier.hparams_dict["sta"].trims = [args.trim for _ in args.layers]

    for mult in args.multipliers:
        applier.hparams_dict["sta"].multipliers = [mult for _ in args.layers]
        applier.apply_vectors()
        applier.generate({"test": items})
        applier.model.reset_all()

    print(f"\nDone. Results saved to {apply_config['generation_output_dir']}")


if __name__ == "__main__":
    main()
