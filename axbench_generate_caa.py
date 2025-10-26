#!/usr/bin/env python3
"""Generate CAA vectors from axbench-concept500 dataset."""

import json
import os
import random
from omegaconf import OmegaConf
from steer.vector_generators.vector_generators import BaseVectorGenerator
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_concept_data(concept_id, limit=None, seed=0):
    """Load axbench-concept500 dataset and create contrastive pairs for a concept."""
    print(f"Loading axbench-concept500 dataset for concept_id={concept_id}...")
    
    # Download train parquet files directly - bypass datasets library completely
    print("Downloading train parquet files...")
    train_files = []
    for i in range(8):  # train has 8 parquet files (train-00000 to train-00007)
        filename = f"data/train-0000{i}-of-00008.parquet"
        try:
            filepath = hf_hub_download(repo_id="pyvene/axbench-concept500", filename=filename, repo_type="dataset")
            train_files.append(filepath)
        except Exception as e:
            print(f"Could not download {filename}: {e}")
            break
    
    # Read all parquet files and concatenate
    print(f"Reading {len(train_files)} parquet files...")
    tables = [pq.read_table(f) for f in train_files]
    combined_table = tables[0]
    for table in tables[1:]:
        combined_table = pq.concat_tables([combined_table, table])
    
    # Convert to list of dicts
    dataset = combined_table.to_pylist()
    
    # Find positive examples for this concept
    positive_examples = [ex for ex in dataset if ex['concept_id'] == concept_id and ex['category'] == 'positive']
    
    if not positive_examples:
        raise ValueError(f"No positive examples found for concept_id={concept_id}")
    
    # Get genre for this concept
    concept_genre = positive_examples[0]['concept_genre']
    concept_name = positive_examples[0]['output_concept']
    print(f"Concept: {concept_name}, Genre: {concept_genre}")
    print(f"Found {len(positive_examples)} positive examples")
    
    # Get all negative examples with same genre
    negative_examples = [
        ex for ex in dataset 
        if ex['category'] == 'negative' and ex['concept_genre'] == concept_genre
    ]
    print(f"Found {len(negative_examples)} negative examples in genre '{concept_genre}'")
    
    # Create mapping of input prompts to negative examples
    negative_by_input = {}
    for neg_ex in negative_examples:
        input_text = neg_ex['input']
        if input_text not in negative_by_input:
            negative_by_input[input_text] = neg_ex
    
    # Create contrastive pairs by matching positive and negative examples with same input
    contrastive_pairs = []
    for pos_ex in positive_examples:
        input_text = pos_ex['input']
        if input_text in negative_by_input:
            contrastive_pairs.append({
                "question": input_text,
                "matching": pos_ex['output'],
                "not_matching": negative_by_input[input_text]['output']
            })
    
    print(f"Created {len(contrastive_pairs)} contrastive pairs")
    
    if not contrastive_pairs:
        raise ValueError(f"No contrastive pairs found for concept_id={concept_id}")
    
    # Random sample if limit specified
    if limit and len(contrastive_pairs) > limit:
        rng = random.Random(seed)
        contrastive_pairs = rng.sample(contrastive_pairs, k=limit)
        print(f"Sampled {len(contrastive_pairs)} pairs (limit={limit})")
    
    return contrastive_pairs, concept_name


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_id", required=True, type=int, help="Concept ID from axbench-concept500")
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--train_limit", type=int, default=None, help="Limit number of training pairs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    # Load training data
    train_data, concept_name = load_concept_data(args.concept_id, limit=args.train_limit, seed=args.seed)
    
    # Config for vector generation
    dataset_name = f"concept_{args.concept_id}"
    vector_base_dir = f"vectors/axbench/{args.model.split('/')[-1]}"
    vector_dir = f"{vector_base_dir}/{dataset_name}/caa_vector"
    
    gen_config = {
        "alg_name": "caa",
        "model_name_or_path": args.model,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "system_prompt": "",
        "layers": args.layers,
        "save_vectors": True,
        "steer_vector_output_dirs": [vector_base_dir],
        "steer_train_hparam_paths": ["hparams/Steer/caa_hparams/generate_caa.yaml"],
        "steer_train_dataset": [dataset_name],
    }
    
    # Generate vectors
    print("\nGenerating CAA vectors...")
    gen_cfg = OmegaConf.create(gen_config)
    vector_generator = BaseVectorGenerator(gen_cfg)
    vector_generator.generate_vectors({dataset_name: train_data})
    
    # Save training pairs for sanity check
    sanity_dir = f"vectors/axbench/{args.model.split('/')[-1]}/{dataset_name}"
    os.makedirs(sanity_dir, exist_ok=True)
    with open(os.path.join(sanity_dir, "train_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Vectors saved to: {vector_dir}")
    print(f"Training pairs saved to: {sanity_dir}/train_pairs.json")
    print(f"Concept: {concept_name} (ID: {args.concept_id})")
    print(f"{'='*60}")
    print(f"\nTo apply these vectors, run:")
    print(f"python axbench_apply_caa.py --concept_id {args.concept_id} --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0 2.0")
    print(f"\nTo run sanity check, run:")
    print(f"python axbench_sanity_check.py --concept_id {args.concept_id} --method caa --model {args.model} --layers {' '.join(map(str, args.layers))} --multipliers 1.0")


if __name__ == "__main__":
    main()


