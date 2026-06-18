"""
Checkpoint helpers for phase-boundary training resume.

This module intentionally supports epoch/phase boundary resume only. Batch-level
resume is out of scope because dataloader sampler state is not tracked.
"""
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint_last.pt"
RESUME_GRANULARITY = "epoch_or_phase_boundary"


class ResumePhase:
    INITIAL_TRAINING = "initial_training"
    SMALLTREE_TRAINING = "smalltree_training"
    ATTACH_DONE = "attach_done"
    PRUNE_PRECHECK_DONE = "prune_precheck_done"
    PRUNING = "pruning"
    FINAL_FINETUNING = "final_finetuning"
    DONE = "done"


SUPPORTED_PHASES = {
    ResumePhase.INITIAL_TRAINING,
    ResumePhase.FINAL_FINETUNING,
}


KNOWN_PHASES = {
    ResumePhase.INITIAL_TRAINING,
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
    torch.set_rng_state(rng_state["torch"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])
    if torch.cuda.is_available() and "cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


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


def validate_resume_config(
    checkpoint_config: Dict[str, Any],
    current_config: Dict[str, Any],
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

    if mismatches:
        formatted = "\n".join(f"  - {mismatch}" for mismatch in mismatches)
        raise ValueError(
            "Resume config is incompatible with the checkpoint model shape:\n"
            f"{formatted}"
        )


def is_phase_supported_now(phase: str) -> bool:
    return phase in SUPPORTED_PHASES


def _get_nested(config: Dict[str, Any], key_path: tuple) -> Any:
    value = config
    for key in key_path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _serialize_alpha(alpha: Any) -> Any:
    if torch.is_tensor(alpha):
        return alpha.detach().cpu()
    return alpha
