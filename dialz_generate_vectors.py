import argparse
import json
import os
from typing import Dict, List

from omegaconf import OmegaConf
import yaml

from steer.vector_generators.vector_generators import BaseVectorGenerator


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def _resolve_dialz_datasets_dir() -> str:
    candidates = [
        os.path.join(ROOT_DIR, "dialz_data", "datasets", "load"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


DIALZ_DATASETS_DIR = _resolve_dialz_datasets_dir()


def load_dialz_dataset(dataset_name: str, limit: int | None = None) -> List[Dict]:
    dataset_path = os.path.join(DIALZ_DATASETS_DIR, f"{dataset_name}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized: List[Dict] = []
    for item in data:
        question = item.get("question", "")
        positive = item.get("positive", item.get("matching"))
        negative = item.get("negative", item.get("not_matching"))
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


def ensure_sta_sae_available(generate_hparam_path: str, layers: List[int] | None, sae_width: str | None) -> List[str]:
    """Ensure required STA SAE(s) are available. Returns sae_paths to use.

    If sae_width is provided, only download that width for the selected layers and
    construct canonical sae_paths accordingly.
    """
    try:
        with open(generate_hparam_path, "r", encoding="utf-8") as f:
            gen_cfg = yaml.safe_load(f)
        sae_paths = gen_cfg.get("sae_paths", []) or []
    except Exception:
        sae_paths = []

    if sae_width is None and sae_paths:
        # Use existing YAML paths
        return sae_paths

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
    # Build target sae_paths if a specific width is requested or none existed
    target_layers = layers or [20]
    width_str = sae_width or "16k"
    # Canonical average dirs per width (based on project usage)
    canonical_avg = {
        "16k": "average_l0_91",
        "131k": "average_l0_24",
    }
    avg_dir = canonical_avg.get(width_str, "average_l0_91")
    target_paths_rel: List[str] = [
        os.path.join("hugging_cache", repo_name, f"layer_{L}", f"width_{width_str}", avg_dir)
        for L in target_layers
    ]

    # Check existence; if missing (or params.npz missing), download only those subfolders
    missing = []
    for p in target_paths_rel:
        abs_p = os.path.abspath(os.path.join(ROOT_DIR, p))
        params_file = os.path.join(abs_p, "params.npz")
        if not os.path.exists(params_file):
            missing.append(abs_p)
    if missing:
        try:
            from huggingface_hub import snapshot_download
            repo_id = f"google/{repo_name}"
            os.makedirs(os.path.dirname(base_cache_dir), exist_ok=True)
            allow_patterns = []
            for p in missing:
                rel = os.path.relpath(p, base_cache_dir).replace("\\", "/")
                # ensure pattern stays within repo folder
                if not rel.startswith(".."):  # inside
                    # we need params.npz and cfg.json (and possibly others) under this folder
                    allow_patterns.append(rel + "/params.npz")
                    allow_patterns.append(rel + "/cfg.json")
            snapshot_download(
                repo_id=repo_id,
                local_dir=base_cache_dir,
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )
            print(f"Downloaded SAEs: {allow_patterns}")
        except Exception as e:
            print(f"Warning: failed to download canonical SAEs automatically: {e}")
            print("Install huggingface_hub and ensure internet/HF auth if needed, or pre-populate hugging_cache.")

    return target_paths_rel if target_paths_rel else sae_paths


def main():
    parser = argparse.ArgumentParser(description="Generate steering vectors from dialz datasets.")
    parser.add_argument("--method", choices=["caa", "sta"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="google/gemma-2-9b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_limit", type=int, default=128)
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Override layers to generate vectors for (e.g., 20 21)")
    parser.add_argument("--sae_width", default=None, help="STA only: SAE width to use (e.g., 16k or 131k). Downloads only that width.")
    args = parser.parse_args()

    train_limit = None if args.train_limit == 0 else args.train_limit
    train_items = load_dialz_dataset(args.dataset, limit=train_limit)
    if not train_items:
        raise RuntimeError(f"No usable items found in dataset {args.dataset}")

    generate_hparam_path = f"hparams/Steer/{args.method}_hparams/generate_{args.method}.yaml"
    sae_paths_override: List[str] | None = None
    if args.method == "sta":
        sae_paths_override = ensure_sta_sae_available(generate_hparam_path, args.layers, args.sae_width)

    cfg_dict = {
        "model_name_or_path": args.model,
        "torch_dtype": "bfloat16",
        "device": args.device,
        "seed": 0,
        "use_chat_template": False,
        "system_prompt": "",
        "steer_train_hparam_paths": [generate_hparam_path],
        "steer_train_dataset": [args.dataset],
        "save_vectors": True,
        "steer_vector_output_dirs": [f"vectors/dialz/gemma-2-9b"],
    }
    if args.layers is not None:
        cfg_dict["layers"] = args.layers
    if sae_paths_override:
        # Use project-root relative paths like "hugging_cache/..."
        cfg_dict["sae_paths"] = [os.path.join(ROOT_DIR, p) for p in sae_paths_override]
    top_generate_cfg = OmegaConf.create(cfg_dict)

    vector_generator = BaseVectorGenerator(top_generate_cfg)
    vectors = vector_generator.generate_vectors({args.dataset: train_items})
    print("Generated vectors:", {k: list(v.keys()) for k, v in vectors.items()})
    print(f"Saved under vectors/dialz/gemma-2-9b/{args.dataset}/{args.method}_vector/")


if __name__ == "__main__":
    main()


