#!/usr/bin/env python3
"""Sanity-check application: generate on the same training prompts used to build vectors.

Loads the saved CAA/STA vectors for a Concept500 concept_id and generates on the
contrastive training prompts (inputs) to verify steering works on the training set.
"""

import os
from typing import List, Dict
from datasets import load_dataset
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier


def build_training_inputs(concept_id: int, split: str = "train") -> List[Dict[str, str]]:
    # Load only the requested split
    dsplit = load_dataset("pyvene/axbench-concept500", split=split)

    pos_rows = [r for r in dsplit if r.get("concept_id") == concept_id and r.get("category") == "positive"]
    if not pos_rows:
        raise ValueError(f"No positive rows found for concept_id={concept_id} in split='{split}'")
    genre = pos_rows[0].get("concept_genre")

    neg_by_input = {}
    for r in dsplit:
        if r.get("concept_genre") == genre and r.get("category") == "negative":
            key = r.get("input", "")
            if key and key not in neg_by_input:
                neg_by_input[key] = r

    items: List[Dict[str, str]] = []
    for r in pos_rows:
        inp = r.get("input", "")
        if not inp:
            continue
        if inp not in neg_by_input:
            continue
        # expected = positive output (reference), matching the vector training target
        items.append({"input": inp, "reference_response": r.get("output", "")})
    return items


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--method", choices=["caa","sta"], required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--trims", nargs="+", type=float, default=[0.65])
    parser.add_argument("--mode", default="act_and_freq", choices=["act_and_freq","only_act","only_freq"])
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    base_dir = f"vectors/axbench/{model_tag}/concept_{args.concept_id}"
    vector_dir = os.path.join(base_dir, "caa_vector" if args.method == "caa" else "sta_vector")

    if not os.path.exists(vector_dir):
        print(f"Error: Vectors not found at {vector_dir}")
        print("Generate them first with:")
        gen = "axbench_generate_caa.py" if args.method == "caa" else "axbench_generate_sta.py"
        print(f"python {gen} --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))}")
        return

    apply_cfg_dict = {
        "alg_name": args.method,
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "apply_steer_hparam_paths": [
            "hparams/Steer/caa_hparams/apply_caa.yaml" if args.method == "caa" else "hparams/Steer/sta_hparams/apply_sta.yaml"
        ],
        "steer_vector_load_dir": [vector_dir],
        "generation_data": ["sanity"],
        "generation_data_size": None,
        "generation_output_dir": f"generation/axbench/{model_tag}/concept_{args.concept_id}/{args.method}_sanity",
        "steer_from_end_position": True,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "vllm_enable": False,
        "mode": args.mode,
    }

    apply_cfg = OmegaConf.create(apply_cfg_dict)
    vector_applier = BaseVectorApplier(apply_cfg)

    sanity_items = {"sanity": build_training_inputs(args.concept_id, split="train")}

    if args.method == "caa":
        for mult in args.multipliers:
            print(f"\n{'='*60}")
            print(f"Sanity check CAA | Multiplier: {mult}")
            print('='*60)
            vector_applier.hparams_dict["caa"].multipliers = [mult for _ in args.layers]
            vector_applier.apply_vectors()
            vector_applier.generate(sanity_items)
            vector_applier.model.reset_all()
    else:
        for mult in args.multipliers:
            for trim in args.trims:
                print(f"\n{'='*60}")
                print(f"Sanity check STA | Multiplier: {mult} | Trim: {trim} | Mode: {args.mode}")
                print('='*60)
                vector_applier.hparams_dict["sta"].multipliers = [mult for _ in args.layers]
                vector_applier.hparams_dict["sta"].trims = [trim for _ in args.layers]
                vector_applier.hparams_dict["sta"].mode = args.mode
                vector_applier.apply_vectors()
                vector_applier.generate(sanity_items)
                vector_applier.model.reset_all()

    print(f"\nDone! Results saved to {apply_cfg_dict['generation_output_dir']}")


if __name__ == "__main__":
    main()


