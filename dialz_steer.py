import argparse
import json
import os
import random
from typing import Dict, List

from omegaconf import OmegaConf
import yaml

from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def _resolve_dialz_datasets_dir() -> str:
    candidates = [
        os.path.join(ROOT_DIR, "dialz_data", "datasets", "load"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Prefer new location even if not yet created
    return candidates[0]


DIALZ_DATASETS_DIR = _resolve_dialz_datasets_dir()


def load_dialz_dataset(dataset_name: str, limit: int | None = None) -> List[Dict]:
    dataset_path = os.path.join(DIALZ_DATASETS_DIR, f"{dataset_name}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize into the format expected by CAA/STA: question (optional), matching, not_matching
    normalized: List[Dict] = []
    for item in data:
        # Prefer explicit question if present; otherwise leave empty
        question = item.get("question", "")

        # Many dialz sets use positive/negative pairs
        positive = item.get("positive")
        negative = item.get("negative")

        # Fallbacks if different keys are present
        if positive is None and "matching" in item:
            positive = item.get("matching")
        if negative is None and "not_matching" in item:
            negative = item.get("not_matching")

        # If still missing, attempt to use any string fields heuristically
        if positive is None:
            for k, v in item.items():
                if isinstance(v, str) and v.strip():
                    positive = v
                    break
        if negative is None:
            # Try to choose a different field for negative
            for k, v in item.items():
                if isinstance(v, str) and v.strip() and v != positive:
                    negative = v
                    break

        if positive is None or negative is None:
            # Skip malformed rows
            continue

        normalized.append({
            "question": question,
            "matching": positive,
            "not_matching": negative,
        })

    if limit is not None and limit > 0:
        normalized = normalized[:limit]
    return normalized


def build_test_prompts(raw_items: List[Dict], num_prompts: int = 10) -> List[Dict]:
    # Create lightweight prompts to probe steering behavior based on examples
    prompts: List[str] = []

    def clean_text(text: str) -> str:
        return " ".join(text.strip().split())

    candidates: List[str] = []
    for item in raw_items:
        # Prefer question if any
        if isinstance(item.get("question"), str) and item.get("question").strip():
            candidates.append(clean_text(item["question"]))
        # Use positive/negative exemplars as seeds
        for key in ("positive", "negative", "matching", "not_matching", "prompt"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                candidates.append(clean_text(val))

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    if not unique_candidates:
        return []

    sampled = unique_candidates[: max(1, num_prompts)]

    # Convert to generation dataset format expected by BaseVectorApplier.generate
    for s in sampled:
        # Truncate very long exemplars
        if len(s) > 800:
            s = s[:800]
        # Wrap to encourage a short response while keeping style/content cues
        prompts.append(f"Respond succinctly in one or two sentences: {s}")

    return [{"input": p} for p in prompts]


def run(method: str,
        dataset_name: str,
        model_name_or_path: str,
        device: str,
        train_limit: int | None,
        num_test: int) -> None:
    method = method.lower()
    if method not in {"caa", "sta"}:
        raise ValueError("method must be one of {'caa','sta'}")

    # 1) Load dataset and normalize for training
    train_items = load_dialz_dataset(dataset_name, limit=train_limit)
    if not train_items:
        raise RuntimeError(f"No usable items found in dataset {dataset_name}")

    # 2) Configure vector generation (Gemma 2 9B, layer 20)
    generate_hparam_path = f"hparams/Steer/{method}_hparams/generate_{method}.yaml"

    # If STA, ensure canonical SAEs are downloaded into hugging_cache if missing
    if method == "sta":
        try:
            with open(generate_hparam_path, "r", encoding="utf-8") as f:
                gen_cfg = yaml.safe_load(f)
            sae_paths = gen_cfg.get("sae_paths", []) or []
        except Exception:
            sae_paths = []

        # infer repo name from path (e.g., hugging_cache/gemma-scope-9b-it-res/..)
        repo_name = None
        for p in sae_paths:
            parts = os.path.normpath(p).split(os.sep)
            if "hugging_cache" in parts:
                idx = parts.index("hugging_cache")
                if idx + 1 < len(parts):
                    repo_name = parts[idx + 1]
                    break
        if repo_name is None:
            repo_name = "gemma-scope-9b-it-res"

        base_cache_dir = os.path.join(ROOT_DIR, "hugging_cache", repo_name)
        # If any sae_path is missing, fetch the full repo snapshot so all layers/widths are present
        need_download = False
        for p in sae_paths:
            # resolve relative paths like ../hugging_cache/... from project root
            abs_p = p
            if not os.path.isabs(abs_p):
                abs_p = os.path.abspath(os.path.join(os.path.dirname(__file__), p))
            if not os.path.exists(abs_p):
                need_download = True
                break
        if need_download:
            try:
                from huggingface_hub import snapshot_download
                repo_id = f"google/{repo_name}"
                os.makedirs(os.path.dirname(base_cache_dir), exist_ok=True)
                # Download/update repo into hugging_cache/repo_name
                snapshot_download(repo_id=repo_id, local_dir=base_cache_dir, local_dir_use_symlinks=False, allow_patterns=None)
                print(f"Downloaded canonical SAEs to: {base_cache_dir}")
            except Exception as e:
                print(f"Warning: failed to download canonical SAEs automatically: {e}")
                print("Install huggingface_hub and ensure internet/HF auth if needed, or pre-populate hugging_cache.")
    top_generate_cfg = OmegaConf.create({
        "model_name_or_path": model_name_or_path,
        "torch_dtype": "bfloat16",
        "device": device,
        "seed": 0,
        "use_chat_template": False,
        "system_prompt": "",
        "steer_train_hparam_paths": [generate_hparam_path],
        "steer_train_dataset": [dataset_name],
        "save_vectors": True,
        "steer_vector_output_dirs": [f"vectors/dialz/gemma-2-9b"]
    })

    vector_generator = BaseVectorGenerator(top_generate_cfg)
    vectors = vector_generator.generate_vectors({dataset_name: train_items})

    # 3) Configure vector application
    vector_dir = os.path.join(
        "vectors",
        "dialz",
        "gemma-2-9b",
        dataset_name,
        f"{method}_vector",
    )

    apply_hparam_path = f"hparams/Steer/{method}_hparams/apply_{method}.yaml"

    generation_output_dir = os.path.join(
        "generation",
        "dialz",
        "gemma-2-9b",
        dataset_name,
        method,
    )

    apply_top_cfg = OmegaConf.create({
        "model_name_or_path": model_name_or_path,
        "torch_dtype": "bfloat16",
        "device": device,
        "seed": 0,
        "use_chat_template": False,
        "system_prompt": "",
        "apply_steer_hparam_paths": [apply_hparam_path],
        "steer_vector_load_dir": [vector_dir],
        "generation_data": [dataset_name],  # placeholder; not used when passing datasets explicitly
        "generation_data_size": None,
        "generation_output_dir": generation_output_dir,
        "num_responses": 1,
        "steer_from_end_position": False,
        "generate_orig_output": True,
        "generation_params": {
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.9,
        },
        "vllm_enable": False,
    })

    vector_applier = BaseVectorApplier(apply_top_cfg)
    vector_applier.apply_vectors()

    # 4) Build simple test prompts from raw examples and generate
    with open(os.path.join(DIALZ_DATASETS_DIR, f"{dataset_name}.json"), "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    test_entries = build_test_prompts(raw_items, num_prompts=num_test)
    if not test_entries:
        print("No test prompts could be constructed from dataset; skipping generation.")
        return

    test_dataset = {"dialz_test": test_entries}
    vector_applier.generate(test_dataset)

    print("\nSteering complete.")
    print(f"Vectors saved under: {os.path.join('vectors', 'dialz', 'gemma-2-9b', dataset_name)}")
    print(f"Generation outputs saved to: {generation_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Dialz steering runner for Gemma 2 9B (layer 20) using CAA/STA.")
    parser.add_argument("--method", choices=["caa", "sta"], required=True, help="Steering method to use")
    parser.add_argument("--dataset", required=True, help="Dataset name (JSON file basename) from dialz/dialz/datasets/load")
    parser.add_argument("--model", default="google/gemma-2-9b", help="Model path or HF id for Gemma 2 9B")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    parser.add_argument("--train_limit", type=int, default=128, help="Limit number of training pairs (0 = all)")
    parser.add_argument("--num_test", type=int, default=10, help="Number of test prompts to generate")

    args = parser.parse_args()
    train_limit = None if args.train_limit == 0 else args.train_limit

    run(
        method=args.method,
        dataset_name=args.dataset,
        model_name_or_path=args.model,
        device=args.device,
        train_limit=train_limit,
        num_test=args.num_test,
    )


if __name__ == "__main__":
    main()


