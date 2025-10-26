import os
from typing import List, Dict
from datasets import load_dataset
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier


def build_test_inputs(concept_id: int) -> List[Dict[str, str]]:
    # Load train and test separately; use streaming for test to avoid preparing other splits
    train = load_dataset("pyvene/axbench-concept500", split="train", verification_mode="no_checks")
    test_stream = load_dataset(
        "pyvene/axbench-concept500",
        split="test",
        streaming=True,
        verification_mode="no_checks",
    )

    pos_rows = [r for r in train if r.get("concept_id") == concept_id and r.get("category") == "positive"]
    if not pos_rows:
        raise ValueError(f"No positive rows found in train for concept_id={concept_id}")
    genre = pos_rows[0].get("concept_genre")

    items: List[Dict[str, str]] = []
    for r in test_stream:
        if r.get("concept_genre") != genre:
            continue
        inp = r.get("input", "")
        if not inp:
            continue
        expected = r.get("output", "") if (r.get("category") == "positive" and r.get("concept_id") == concept_id) else ""
        items.append({"input": inp, "reference_response": expected})
    return items


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="", type=int, default=[20])
    parser.add_argument("--multipliers", nargs="", type=float, default=[1.0])
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    vector_dir = f"vectors/axbench/{model_tag}/concept_{args.concept_id}/caa_vector"

    if not os.path.exists(vector_dir):
        print(f"Error: Vectors not found at {vector_dir}")
        print("Generate them first with:")
        print(f"python axbench_generate_caa.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))}")
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
        "generation_output_dir": f"generation/axbench/{model_tag}/concept_{args.concept_id}/caa",
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
    vector_applier = BaseVectorApplier(apply_cfg)

    test_items = {"test": build_test_inputs(args.concept_id)}

    for mult in args.multipliers:
        print(f"\n{'='*60}")
        print(f"Multiplier: {mult}")
        print('='*60)
        vector_applier.hparams_dict["caa"].multipliers = [mult for _ in args.layers]
        vector_applier.apply_vectors()
        vector_applier.generate(test_items)
        vector_applier.model.reset_all()

    print(f"\nDone! Results saved to {apply_config['generation_output_dir']}")


if __name__ == "__main__":
    main()