"""
Checkpoint helpers for phase-boundary training resume.

This module intentionally supports epoch/phase boundary resume only. Batch-level
resume is out of scope because dataloader sampler state is not tracked.
"""
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from utils.model_utils import serialize_tree_topology


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint_last.pt"
RESUME_GRANULARITY = "epoch_or_phase_boundary"


class ResumePhase:
    INITIAL_TRAINING = "initial_training"
    GROW_LOOP_BOUNDARY = "grow_loop_boundary"
    INTERMEDIATE_FINETUNING = "intermediate_finetuning"
    SMALLTREE_TRAINING = "smalltree_training"
    ATTACH_DONE = "attach_done"
    PRUNE_PRECHECK_DONE = "prune_precheck_done"
    PRUNING = "pruning"
    FINAL_FINETUNING = "final_finetuning"
    DONE = "done"


SUPPORTED_PHASES = {
    ResumePhase.INITIAL_TRAINING,
    ResumePhase.GROW_LOOP_BOUNDARY,
    ResumePhase.INTERMEDIATE_FINETUNING,
    ResumePhase.SMALLTREE_TRAINING,
    ResumePhase.ATTACH_DONE,
    ResumePhase.PRUNE_PRECHECK_DONE,
    ResumePhase.PRUNING,
    ResumePhase.FINAL_FINETUNING,
}


KNOWN_PHASES = {
    ResumePhase.INITIAL_TRAINING,
    ResumePhase.GROW_LOOP_BOUNDARY,
    ResumePhase.INTERMEDIATE_FINETUNING,
    ResumePhase.SMALLTREE_TRAINING,
    ResumePhase.ATTACH_DONE,
    ResumePhase.PRUNE_PRECHECK_DONE,
    ResumePhase.PRUNING,
    ResumePhase.FINAL_FINETUNING,
    ResumePhase.DONE,
}


REQUIRED_CHECKPOINT_KEYS = {
    "schema_version",
    "resume_granularity",
    "phase",
    "phase_epoch",
    "global_step",
    "model_state_dict",
    "optimizer_state_dict",
    "lr_scheduler_state_dict",
    "alpha",
    "configs",
    "experiment_path",
    "wandb_run_id",
    "rng_state",
}


REQUIRED_SMALLTREE_CHECKPOINT_KEYS = {
    "small_model_state_dict",
    "small_optimizer_state_dict",
    "small_lr_scheduler_state_dict",
    "small_alpha",
    "smalltree_epoch",
    "selected_leaf_index",
    "selected_leaf_path",
    "n_effective_leaves",
    "growing_iterations",
}

REQUIRED_GROW_BOUNDARY_CHECKPOINT_KEYS = {
    "growing_iterations",
    "tree_topology",
}


REQUIRED_INTERMEDIATE_FINETUNING_CHECKPOINT_KEYS = {
    "growing_iterations",
    "intermediate_epoch",
    "tree_topology",
}


REQUIRED_ATTACH_DONE_CHECKPOINT_KEYS = {
    "growing_iterations",
    "tree_topology",
}


REQUIRED_PRUNE_PRECHECK_CHECKPOINT_KEYS = {
    "prune",
    "pruning_iterations",
    "tree_topology",
}


REQUIRED_PRUNING_CHECKPOINT_KEYS = {
    "pruning_iterations",
    "pruning_complete",
    "tree_topology",
}


REQUIRED_FINAL_FINETUNING_CHECKPOINT_KEYS = {
    "tree_topology",
}


MODEL_CONFIG_KEYS = (
    ("training", "n_ary"),
    ("training", "latent_dim"),
    ("training", "mlp_layers"),
    ("training", "initial_depth"),
    ("training", "activation"),
    ("training", "inp_shape"),
    ("training", "encoder"),
    ("training", "decoder"),
)


def get_rng_state() -> Dict[str, Any]:
    rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def restore_rng_state(rng_state: Dict[str, Any]) -> None:
    torch.set_rng_state(rng_state["torch"].detach().cpu())
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])
    if torch.cuda.is_available() and "cuda" in rng_state:
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in rng_state["cuda"]])


def build_checkpoint(
    *,
    phase: str,
    phase_epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    alpha: Any,
    configs: Dict[str, Any],
    experiment_path: Path,
    wandb_run_id: Optional[str],
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if phase not in KNOWN_PHASES:
        raise ValueError(f"Unknown checkpoint phase: {phase}")

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "resume_granularity": RESUME_GRANULARITY,
        "phase": phase,
        "phase_epoch": int(phase_epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "alpha": _serialize_alpha(alpha),
        "configs": configs,
        "experiment_path": str(experiment_path),
        "wandb_run_id": wandb_run_id,
        "rng_state": get_rng_state(),
    }
    if hasattr(model, "tree"):
        checkpoint["tree_topology"] = serialize_tree_topology(model.tree)
    if extra_state is not None:
        checkpoint.update(extra_state)
    validate_checkpoint_schema(checkpoint)
    return checkpoint


def save_checkpoint(checkpoint: Dict[str, Any], checkpoint_path: Path) -> None:
    validate_checkpoint_schema(checkpoint)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    validate_checkpoint_schema(checkpoint)
    return checkpoint


def validate_checkpoint_schema(checkpoint: Dict[str, Any]) -> None:
    missing_keys = REQUIRED_CHECKPOINT_KEYS - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing_keys)}")

    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema version: {checkpoint['schema_version']}"
        )

    if checkpoint["resume_granularity"] != RESUME_GRANULARITY:
        raise ValueError(
            f"Unsupported resume granularity: {checkpoint['resume_granularity']}"
        )

    if checkpoint["phase"] not in KNOWN_PHASES:
        raise ValueError(f"Unknown checkpoint phase: {checkpoint['phase']}")

    if checkpoint["phase"] == ResumePhase.SMALLTREE_TRAINING:
        missing_smalltree_keys = (
            REQUIRED_SMALLTREE_CHECKPOINT_KEYS | {"tree_topology"}
        ) - set(checkpoint.keys())
        if missing_smalltree_keys:
            raise ValueError(
                "SmallTree checkpoint is missing required keys: "
                f"{sorted(missing_smalltree_keys)}"
            )
    elif checkpoint["phase"] == ResumePhase.GROW_LOOP_BOUNDARY:
        _raise_if_missing_phase_keys(
            checkpoint,
            REQUIRED_GROW_BOUNDARY_CHECKPOINT_KEYS,
            "grow-loop-boundary",
        )
    elif checkpoint["phase"] == ResumePhase.INTERMEDIATE_FINETUNING:
        _raise_if_missing_phase_keys(
            checkpoint,
            REQUIRED_INTERMEDIATE_FINETUNING_CHECKPOINT_KEYS,
            "intermediate-finetuning",
        )
    elif checkpoint["phase"] == ResumePhase.ATTACH_DONE:
        _raise_if_missing_phase_keys(
            checkpoint,
            REQUIRED_ATTACH_DONE_CHECKPOINT_KEYS,
            "attach-done",
        )
    elif checkpoint["phase"] == ResumePhase.PRUNE_PRECHECK_DONE:
        _raise_if_missing_phase_keys(
            checkpoint,
            REQUIRED_PRUNE_PRECHECK_CHECKPOINT_KEYS,
            "prune-precheck",
        )
    elif checkpoint["phase"] == ResumePhase.PRUNING:
        _raise_if_missing_phase_keys(
            checkpoint,
            REQUIRED_PRUNING_CHECKPOINT_KEYS,
            "pruning",
        )
    elif checkpoint["phase"] == ResumePhase.FINAL_FINETUNING:
        _raise_if_missing_phase_keys(
            checkpoint,
            REQUIRED_FINAL_FINETUNING_CHECKPOINT_KEYS,
            "final-finetuning",
        )


def validate_resume_config(
    checkpoint_config: Dict[str, Any],
    current_config: Dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    mismatches = []
    for key_path in MODEL_CONFIG_KEYS:
        checkpoint_value = _get_nested(checkpoint_config, key_path)
        current_value = _get_nested(current_config, key_path)
        if checkpoint_value != current_value:
            mismatches.append(
                "{}: checkpoint={!r}, current={!r}".format(
                    ".".join(key_path),
                    checkpoint_value,
                    current_value,
                )
            )

    if not mismatches:
        return

    formatted = "\n".join(f"  - {mismatch}" for mismatch in mismatches)
    message = (
        "Resume config is incompatible with the checkpoint model shape:\n"
        f"{formatted}"
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def is_phase_supported_now(phase: str) -> bool:
    return phase in SUPPORTED_PHASES


def _get_nested(config: Dict[str, Any], key_path: tuple) -> Any:
    value = config
    for key in key_path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _raise_if_missing_phase_keys(
    checkpoint: Dict[str, Any],
    required_keys: set,
    phase_name: str,
) -> None:
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"{phase_name} checkpoint is missing required keys: {sorted(missing_keys)}"
        )


def _serialize_alpha(alpha: Any) -> Any:
    if torch.is_tensor(alpha):
        return alpha.detach().cpu()
    return alpha
