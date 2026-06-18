import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from matplotlib.patches import Patch
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from models.model import TreeVAE
from train.validate_tree import compute_likelihood
from utils.data_utils import get_data, get_gen
from utils.model_utils import construct_data_tree, construct_tree_fromnpy
from utils.taxonomy_class_utils import taxoclass
from utils.training_utils import Custom_Metrics, compute_leaves, get_dataset_labels, predict, validate_one_epoch
from utils.utils import cluster_acc, dendrogram_purity, leaf_purity, reset_random_seeds


def compute_tree_layout(data_tree):
    children_by_parent = {}
    for node_id, _, parent_id, _ in data_tree:
        if parent_id is not None:
            children_by_parent.setdefault(parent_id, []).append(node_id)

    for children in children_by_parent.values():
        children.sort()

    positions = {}
    next_leaf_x = 0

    def place_node(node_id, depth):
        nonlocal next_leaf_x
        children = children_by_parent.get(node_id, [])
        if not children:
            positions[node_id] = (next_leaf_x, -depth)
            next_leaf_x += 1
            return positions[node_id][0]

        child_x = [place_node(child_id, depth + 1) for child_id in children]
        positions[node_id] = (float(np.mean(child_x)), -depth)
        return positions[node_id][0]

    place_node(data_tree[0][0], 0)
    return positions, children_by_parent


def plot_tree_with_leaf_label_histograms(data_tree, leaf_label_counts, class_labels, save_path, use_taxonomy_colors=False):
    positions, children_by_parent = compute_tree_layout(data_tree)
    n_leaves = max(1, len(leaf_label_counts))
    n_classes = max(1, len(class_labels))
    max_depth = max(abs(y) for _, y in positions.values()) if positions else 1
    if use_taxonomy_colors:
        class_colors = taxoclass.colors_for_labels(class_labels)
    else:
        class_colors = "#2f6f9f"

    fig_width = max(10, n_leaves * 1.8)
    fig_height = max(6, (max_depth + 1) * 1.4 + min(3.0, n_classes * 0.18))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for parent_id, child_ids in children_by_parent.items():
        parent_x, parent_y = positions[parent_id]
        for child_id in child_ids:
            child_x, child_y = positions[child_id]
            ax.plot([parent_x, child_x], [parent_y, child_y], color="0.65", linewidth=1.5, zorder=1)

    leaf_rows = [row for row in data_tree if row[3] == 1]
    leaf_index_by_node = {node_id: leaf_idx for leaf_idx, (node_id, _, _, _) in enumerate(leaf_rows)}

    x_values = [x for x, _ in positions.values()]
    y_values = [y for _, y in positions.values()]
    ax.set_xlim(min(x_values) - 0.8, max(x_values) + 0.8)
    ax.set_ylim(min(y_values) - 2.0, max(y_values) + 0.5)

    for node_id, label, _, node_type in data_tree:
        x, y = positions[node_id]
        if node_type == 1:
            ax.scatter(x, y, s=220, color="#7bc96f", edgecolor="0.25", linewidth=1.0, zorder=3)
            ax.text(x, y + 0.18, f"Leaf {leaf_index_by_node[node_id]}", ha="center", va="bottom", fontsize=8)
        else:
            ax.scatter(x, y, s=320, color="#8db7e8", edgecolor="0.25", linewidth=1.0, zorder=3)
            ax.text(x, y, str(label), ha="center", va="center", fontsize=8)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    hist_width = min(0.18, 0.85 / max(1, n_leaves))
    hist_height = min(0.32, max(0.18, n_classes * 0.025))

    for leaf_idx, (node_id, _, _, _) in enumerate(leaf_rows):
        x, y = positions[node_id]
        axes_x = (x - x_min) / x_span - hist_width / 2
        axes_y = (y - y_min) / y_span - hist_height - 0.05
        inset = ax.inset_axes([axes_x, axes_y, hist_width, hist_height], transform=ax.transAxes)
        counts = leaf_label_counts[leaf_idx]
        y_pos = np.arange(n_classes)
        inset.barh(y_pos, counts, color=class_colors, height=0.75)
        inset.set_title(f"L{leaf_idx} n={int(counts.sum())}", fontsize=7, pad=1)
        inset.tick_params(axis="both", labelsize=6, length=2, pad=1)
        tick_step = max(1, int(np.ceil(n_classes / 12)))
        tick_positions = y_pos[::tick_step]
        inset.set_yticks(tick_positions)
        inset.set_yticklabels(class_labels[::tick_step])
        inset.set_xlim(0, max(1, int(counts.max())))
        inset.invert_yaxis()
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)

    if use_taxonomy_colors:
        legend_handles = [
            Patch(facecolor=taxoclass.colors["power"], label="power"),
            Patch(facecolor=taxoclass.colors["intermediate"], label="intermediate"),
            Patch(facecolor=taxoclass.colors["precision"], label="precision"),
        ]
        if any(taxoclass.category_for_label(label) == "unknown" for label in class_labels):
            legend_handles.append(Patch(facecolor=taxoclass.colors["unknown"], label="unknown"))
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    ax.set_axis_off()
    ax.set_title("Tree leaf class-label histograms", fontsize=13, pad=12)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tree_visualization(data_tree, y_test, y_test_pred, data_name):
    os.makedirs("assets/", exist_ok=True)
    class_labels = np.unique(y_test)
    leaf_label_counts = []
    for leaf_idx in range(sum(row[3] == 1 for row in data_tree)):
        indices = np.where(y_test_pred == leaf_idx)[0]
        counts = np.array([np.sum(y_test[indices] == label) for label in class_labels])
        leaf_label_counts.append(counts)

    save_path = Path("assets") / "test_tree_leaf_label_histograms.png"
    use_taxonomy_colors = data_name in {"hograspnet_full_toy", "hograspnet_uniform_toy"}
    plot_tree_with_leaf_label_histograms(
        data_tree,
        leaf_label_counts,
        class_labels,
        save_path,
        use_taxonomy_colors=use_taxonomy_colors,
    )
    return save_path


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
    tree_visualization_path = save_tree_visualization(data_tree, y_test, y_test_pred, configs["data"]["data_name"])

    print(np.unique(y_test_pred, return_counts=True))
    print("Accuracy:", acc)
    print("Normalized Mutual Information:", nmi)
    print("Adjusted Rand Index:", ari)
    print("Dendrogram Purity:", dp)
    print("Leaf Purity:", lp)
    print("Digits", np.unique(y_test))
    print("Tree visualization:", tree_visualization_path)

    return {
        "accuracy": acc,
        "normalized_mutual_information": nmi,
        "adjusted_rand_index": ari,
        "dendrogram_purity": dp,
        "leaf_purity": lp,
        "data_tree": data_tree,
        "tree_visualization": tree_visualization_path,
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
