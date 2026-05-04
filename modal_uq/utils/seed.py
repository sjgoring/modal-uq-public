
import hashlib
import random

import numpy as np


def allocate_seed() -> int:
    """Allocate a fresh 32-bit seed from OS entropy."""
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def derive_seed(parent_seed: int, *labels) -> int:
    """Derive a stable child seed from a parent seed and one or more labels."""
    if parent_seed is None:
        return allocate_seed()

    payload = "::".join([str(int(parent_seed)), *[str(label) for label in labels]])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


def resolve_seed(seed: int | None, parent_seed: int | None = None, *labels) -> int:
    """Return an explicit seed if present, otherwise derive or allocate one."""
    if seed is not None:
        return int(seed)
    if parent_seed is None:
        return allocate_seed()
    return derive_seed(parent_seed, *labels)


def resolve_runtime_seeds(cfg: dict) -> dict:
    """Fill in missing runtime seeds in-place and return the updated config.
    
    Consumes top-level 'seeds' block if present, falling back to per-field seeds for backward compatibility.
    The top-level 'seeds' block should contain: experiment, dataset, model, pgt, uncertainty.
    """
    # Check for top-level seeds block
    seeds_block = cfg.get("seeds", {})
    
    experiment = cfg.setdefault("experiment", {})
    # Use top-level seed if present and no explicit experiment.seed
    if seeds_block.get("experiment") is not None and experiment.get("seed") is None:
        experiment["seed"] = seeds_block["experiment"]
    root_seed = resolve_seed(experiment.get("seed"))
    experiment["seed"] = root_seed

    dataset = cfg.setdefault("dataset", {})
    dataset_name = dataset.get("name")
    dataset_params = dataset.setdefault("params", {})
    # Use top-level seed if present and no explicit dataset.params.seed_master or split_seed
    dataset_parent_seed = root_seed
    if seeds_block.get("dataset") is not None and dataset_params.get("seed_master") is None:
        dataset_parent_seed = seeds_block["dataset"]
    
    # Only add split_seed for datasets that use _apply_split (faithful, forestfires)
    if dataset_name in {"faithful", "forestfires"}:
        dataset_params["split_seed"] = resolve_seed(
            dataset_params.get("split_seed"), dataset_parent_seed, "dataset", dataset_name or "dataset", "split_seed"
        )

    if dataset_name in {"synthetic", "synthetic_multimodal"}:
        seed_master = resolve_seed(
            dataset_params.get("seed_master"), dataset_parent_seed, "dataset", dataset_name, "seed_master"
        )
        dataset_params["seed_master"] = seed_master
        for child_key in ("seed_mode_locs", "seed_mode_assign", "seed_sample", "seed_noise"):
            dataset_params[child_key] = resolve_seed(
                dataset_params.get(child_key),
                seed_master,
                "dataset",
                dataset_name,
                child_key,
            )
    elif dataset_name == "synthetic_conditional":
        seed_master = resolve_seed(
            dataset_params.get("seed_master"), dataset_parent_seed, "dataset", dataset_name, "seed_master"
        )
        dataset_params["seed_master"] = seed_master
        for child_key in ("seed_mode_assign", "seed_sample", "seed_noise"):
            dataset_params[child_key] = resolve_seed(
                dataset_params.get(child_key),
                seed_master,
                "dataset",
                dataset_name,
                child_key,
            )
    elif dataset_name == "synthetic_constant_var":
        dataset_params["seed"] = resolve_seed(
            dataset_params.get("seed"), dataset_parent_seed, "dataset", dataset_name, "seed"
        )
    elif dataset_name == "moons_synthetic":
        dataset_params["random_state"] = resolve_seed(
            dataset_params.get("random_state"), dataset_parent_seed, "dataset", dataset_name, "random_state"
        )

    model = cfg.setdefault("model", {})
    model_name = model.get("name")
    model_params = model.setdefault("params", {})
    model_parent_seed = root_seed
    if seeds_block.get("model") is not None and model_params.get("seed") is None:
        model_parent_seed = seeds_block["model"]
    if model_name == "ensemble" or "seed" in model_params:
        model_params["seed"] = resolve_seed(model_params.get("seed"), model_parent_seed, "model", model_name or "model", "seed")
    if "random_state" in model_params:
        model_params["random_state"] = resolve_seed(
            model_params.get("random_state"), model_parent_seed, "model", model_name or "model", "random_state"
        )

    pgt = cfg.setdefault("pseudo_ground_truth", {})
    pgt_name = pgt.get("name")
    pgt_params = pgt.setdefault("params", {})
    pgt_parent_seed = root_seed
    if seeds_block.get("pgt") is not None and pgt_params.get("random_state") is None:
        pgt_parent_seed = seeds_block["pgt"]
    if pgt_name == "gmm" or "random_state" in pgt_params:
        pgt_params["random_state"] = resolve_seed(
            pgt_params.get("random_state"), pgt_parent_seed, "pseudo_ground_truth", pgt_name or "pgt", "random_state"
        )

    if experiment.get("type") == "ood":
        ood_cfg = experiment.setdefault("ood", {})
        ood_cfg["seed"] = resolve_seed(ood_cfg.get("seed"), root_seed, "experiment", "ood", "seed")

    uncertainty = cfg.get("uncertainty", {})
    unc_parent_seed = root_seed
    if seeds_block.get("uncertainty") is not None:
        unc_parent_seed = seeds_block["uncertainty"]
    for measure_index, measure_spec in enumerate(uncertainty.get("measures", [])):
        if not isinstance(measure_spec, dict):
            continue
        params = measure_spec.setdefault("params", {})
        measure_name = measure_spec.get("name", f"measure_{measure_index}")
        if "random_state" in params:
            params["random_state"] = resolve_seed(
                params.get("random_state"),
                unc_parent_seed,
                "uncertainty",
                measure_name,
                str(measure_index),
                "random_state",
            )
        if measure_name in {"alpha_volume", "integrated_volume"} or "mc_random_state" in params:
            params["mc_random_state"] = resolve_seed(
                params.get("mc_random_state"),
                unc_parent_seed,
                "uncertainty",
                measure_name,
                str(measure_index),
                "mc_random_state",
            )

    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
