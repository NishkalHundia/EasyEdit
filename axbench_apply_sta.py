#!/usr/bin/env python3
"""Apply STA vectors for a Concept500 concept on the test split."""

import os
from typing import List, Dict
from datasets import load_dataset
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier


def build_test_inputs(concept_id: int) -> List[Dict[str, str]]:
    # Stream only the test split. Collect all positives for this concept_id.
    test_stream = load_dataset(
        "pyvene/axbench-concept500",
        split="test",
        streaming=True,
        verification_mode="no_checks",
    )

    items: List[Dict[str, str]] = []
    for r in test_stream:
        if r.get("concept_id") != concept_id:
            continue
        if r.get("category") != "positive":
            continue
        inp = r.get("input", "")
        if not inp:
            continue
        expected = r.get("output", "")
        items.append({"input": inp, "reference_response": expected})
    return items


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--trims", nargs="+", type=float, default=[0.65])
    parser.add_argument("--mode", default="act_and_freq", choices=["act_and_freq","only_act","only_freq"])
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    vector_dir = f"vectors/axbench/{model_tag}/concept_{args.concept_id}/sta_vector"

    if not os.path.exists(vector_dir):
        print(f"Error: Vectors not found at {vector_dir}")
        print("Generate them first with:")
        print(f"python axbench_generate_sta.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))}")
        return

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
        "generation_output_dir": f"generation/axbench/{model_tag}/concept_{args.concept_id}/sta",
        "steer_from_end_position": True,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "vllm_enable": False,
        "mode": args.mode,
    }

    print(f"Testing with multipliers: {args.multipliers}, trims: {args.trims}, mode: {args.mode}")
    apply_cfg = OmegaConf.create(apply_config)
    vector_applier = BaseVectorApplier(apply_cfg)

    test_items = {"test": build_test_inputs(args.concept_id)}

    for mult in args.multipliers:
        for trim in args.trims:
            print(f"\n{'='*60}")
            print(f"Multiplier: {mult} | Trim: {trim}")
            print('='*60)
            vector_applier.hparams_dict["sta"].multipliers = [mult for _ in args.layers]
            vector_applier.hparams_dict["sta"].trims = [trim for _ in args.layers]
            vector_applier.hparams_dict["sta"].mode = args.mode
            vector_applier.apply_vectors()
            vector_applier.generate(test_items)
            vector_applier.model.reset_all()

    print(f"\nDone! Results saved to {apply_config['generation_output_dir']}")


if __name__ == "__main__":
    main()


