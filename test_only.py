import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from models.model import TreeVAE
from train.validate_tree import compute_likelihood
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_data_tree, construct_tree_fromnpy
from utils.training_utils import Custom_Metrics, compute_leaves, get_dataset_labels, predict, validate_one_epoch
from utils.utils import cluster_acc, dendrogram_purity, leaf_purity, reset_random_seeds


def load_config(checkpoint_path):
    config_path = checkpoint_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf8") as stream:
        configs = yaml.load(stream, Loader=yaml.Loader)

    configs.setdefault("parser", {})
    configs["globals"]["save_model"] = False
    configs["globals"]["wandb_logging"] = "disabled"
    return configs


def load_treevae_checkpoint(checkpoint_path, configs, device):
    data_tree_path = checkpoint_path / "data_tree.npy"
    weights_path = checkpoint_path / "model_weights.pt"

    if not data_tree_path.exists():
        raise FileNotFoundError(
            f"Missing tree structure file: {data_tree_path}. "
            "A grown/pruned TreeVAE checkpoint needs data_tree.npy as well as model_weights.pt."
        )
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights file: {weights_path}")

    data_tree = np.load(data_tree_path, allow_pickle=True)
    model = TreeVAE(**make_root_only_training_config(configs["training"], data_tree))
    model = construct_tree_fromnpy(model, data_tree, configs)

    state_dict = torch.load(weights_path, map_location=device)
    state_dict = normalize_state_dict_keys(state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        print_load_diagnostics(model, state_dict)
        raise error

    model.to(device)
    model.eval()
    return model


def make_root_only_training_config(training_config, data_tree):
    model_config = dict(training_config)
    root_has_children = any(row[2] == 0 for row in data_tree)
    if root_has_children:
        model_config["initial_depth"] = 0
    return model_config


def normalize_state_dict_keys(state_dict):
    prefixes = ("_orig_mod.", "module.")
    normalized = {}
    changed = False

    for key, value in state_dict.items():
        normalized_key = key
        for prefix in prefixes:
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
                changed = True
        normalized[normalized_key] = value

    return normalized if changed else state_dict


def print_load_diagnostics(model, state_dict, max_items=30):
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state_dict)

    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatch = sorted(
        key for key in model_keys & checkpoint_keys
        if model_state[key].shape != state_dict[key].shape
    )

    print("\nCheckpoint load diagnostics")
    print(f"  model keys: {len(model_keys)}")
    print(f"  checkpoint keys: {len(checkpoint_keys)}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    print(f"  shape mismatches: {len(shape_mismatch)}")

    if missing:
        print("\nFirst missing keys:")
        for key in missing[:max_items]:
            print(" ", key, tuple(model_state[key].shape))
    if unexpected:
        print("\nFirst unexpected keys:")
        for key in unexpected[:max_items]:
            value = state_dict[key]
            print(" ", key, tuple(value.shape) if hasattr(value, "shape") else type(value))
    if shape_mismatch:
        print("\nFirst shape mismatches:")
        for key in shape_mismatch[:max_items]:
            print(" ", key, "model", tuple(model_state[key].shape), "checkpoint", tuple(state_dict[key].shape))


def evaluate_test_only(testset, model, device, configs):
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)
    y_test = get_dataset_labels(testset).numpy()

    metrics_calc_test = Custom_Metrics(device).to(device)
    validate_one_epoch(gen_test, model, metrics_calc_test, 0, device, test=True)

    node_leaves_test, prob_leaves_test = predict(gen_test, model, device, "node_leaves", "prob_leaves")
    y_test_pred = np.squeeze(np.argmax(prob_leaves_test, axis=-1)).numpy()

    acc, _ = cluster_acc(y_test, y_test_pred, return_index=True)
    nmi = normalized_mutual_info_score(y_test, y_test_pred)
    ari = adjusted_rand_score(y_test, y_test_pred)

    leaves = compute_leaves(model.tree, configs["training"]["n_ary"])
    ind_samples_of_leaves = [
        [leaves[i]["node"], np.where(y_test_pred == i)[0]]
        for i in range(len(leaves))
    ]
    dp = dendrogram_purity(model.tree, y_test, ind_samples_of_leaves, configs["training"]["n_ary"])
    lp = leaf_purity(model.tree, y_test, ind_samples_of_leaves, configs["training"]["n_ary"])

    data_tree = construct_data_tree(
        model,
        y_predicted=y_test_pred,
        y_true=y_test,
        n_leaves=len(node_leaves_test),
        data_name=configs["data"]["data_name"],
        n_ary=configs["training"]["n_ary"],
    )

    print(np.unique(y_test_pred, return_counts=True))
    print("Accuracy:", acc)
    print("Normalized Mutual Information:", nmi)
    print("Adjusted Rand Index:", ari)
    print("Dendrogram Purity:", dp)
    print("Leaf Purity:", lp)
    print("Digits", np.unique(y_test))

    return {
        "accuracy": acc,
        "normalized_mutual_information": nmi,
        "adjusted_rand_index": ari,
        "dendrogram_purity": dp,
        "leaf_purity": lp,
        "data_tree": data_tree,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved TreeVAE checkpoint on the test split only.")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to an experiment directory containing config.yaml, data_tree.npy, and model_weights.pt.",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader num_workers.")
    parser.add_argument(
        "--compute_ll",
        action="store_true",
        help="Also compute the saved config's test log-likelihood routine.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    configs = load_config(checkpoint_path)
    if args.num_workers is not None:
        configs["parser"]["num_workers"] = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Checkpoint path:", checkpoint_path)
    print("Device:", device)

    wandb.init(project="treevae", mode="disabled")
    reset_random_seeds(configs["globals"]["seed"])

    _, _, testset = get_data(configs)
    model = load_treevae_checkpoint(checkpoint_path, configs, device)
    evaluate_test_only(testset, model, device, configs)

    if args.compute_ll:
        compute_likelihood(testset, model, device, configs)

    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
