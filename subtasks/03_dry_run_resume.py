"""
Dry-run checkpoint/resume validation with a tiny synthetic point-cloud dataset.

This intentionally avoids external datasets. It verifies that:
1. one initial-training epoch creates checkpoint_last.pt,
2. resuming from that checkpoint starts at the next epoch,
3. the resumed model can run test evaluation.
"""
from copy import deepcopy
from pathlib import Path
import tempfile

import torch
import wandb
from torch.utils.data import TensorDataset

import os, sys
sys.path.append(os.getcwd())

from train.train_tree import run_tree
from utils.checkpoint_utils import CHECKPOINT_FILENAME, ResumePhase, load_checkpoint
from utils.data_utils import get_gen
from utils.training_utils import Custom_Metrics, validate_one_epoch
from utils.utils import reset_random_seeds


def make_configs(experiment_path, num_epochs, resume_from=None):
    return {
        "run_name": "dry_run_resume",
        "data": {
            "data_name": "dry_run",
            "num_clusters_data": 2,
        },
        "training": {
            "n_ary": 2,
            "num_epochs": num_epochs,
            "num_epochs_smalltree": 0,
            "num_epochs_intermediate_fulltrain": 0,
            "num_epochs_finetuning": 0,
            "batch_size": 4,
            "lr": 0.001,
            "weight_decay": 0.0,
            "decay_lr": 0.1,
            "decay_stepsize": 100,
            "decay_kl": 0.01,
            "kl_start": 0.0,
            "inp_shape": 48,
            "latent_dim": [8, 8, 8],
            "mlp_layers": [32, 16, 16],
            "initial_depth": 1,
            "activation": "chamfer",
            "encoder": {
                "architecture": "pointnet",
                "num_points": 16,
            },
            "decoder": {
                "architecture": "pointnet",
            },
            "grow": False,
            "prune": False,
            "num_clusters_tree": 2,
            "augment": False,
            "augmentation_method": ["simple"],
            "aug_decisions_weight": 1,
            "compute_ll": False,
        },
        "parser": {
            "num_workers": 2,
        },
        "globals": {
            "wandb_logging": "disabled",
            "eager_mode": True,
            "seed": 123,
            "save_model": False,
            "resume_from": str(resume_from) if resume_from is not None else None,
            "wandb_run_id": None,
            "results_dir": Path(experiment_path).parent,
        },
    }


def make_dataset():
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(8, 3, 16, generator=generator)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    return TensorDataset(x, y)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = make_dataset()
    trainset_eval = make_dataset()
    testset = make_dataset()

    with tempfile.TemporaryDirectory(prefix="treevae_resume_dry_run_") as tmpdir:
        experiment_path = Path(tmpdir) / "experiment"
        experiment_path.mkdir(parents=True)

        wandb.init(project="treevae", mode="disabled")
        configs = make_configs(experiment_path, num_epochs=1)
        reset_random_seeds(configs["globals"]["seed"])
        run_tree(trainset, trainset_eval, testset, device, configs, experiment_path=experiment_path)

        checkpoint_path = experiment_path / CHECKPOINT_FILENAME
        checkpoint = load_checkpoint(checkpoint_path, device)
        assert checkpoint["phase"] == ResumePhase.INITIAL_TRAINING
        assert checkpoint["phase_epoch"] == 0

        resume_configs = make_configs(experiment_path, num_epochs=2, resume_from=checkpoint_path)
        resume_configs["globals"]["wandb_run_id"] = configs["globals"]["wandb_run_id"]
        reset_random_seeds(resume_configs["globals"]["seed"])
        model = run_tree(
            trainset,
            trainset_eval,
            testset,
            device,
            resume_configs,
            resume_checkpoint=checkpoint,
            experiment_path=experiment_path,
        )

        resumed_checkpoint = load_checkpoint(checkpoint_path, device)
        assert resumed_checkpoint["phase"] == ResumePhase.INITIAL_TRAINING
        assert resumed_checkpoint["phase_epoch"] == 1

        eval_configs = deepcopy(resume_configs)
        gen_test = get_gen(testset, eval_configs, validation=True, shuffle=False)
        metrics_calc = Custom_Metrics(device).to(device)
        validate_one_epoch(gen_test, model, metrics_calc, 0, device, test=True, configs=eval_configs)
        wandb.finish(quiet=True)

    print("Dry-run resume check passed.")


if __name__ == "__main__":
    main()
