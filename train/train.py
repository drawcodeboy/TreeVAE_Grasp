"""
Run training and validation functions of TreeVAE.
"""
import time
from pathlib import Path
import wandb
import uuid
import os
import torch

from utils.data_utils import get_data
from utils.utils import reset_random_seeds
from utils.checkpoint_utils import load_checkpoint
from train.train_tree import run_tree
from train.validate_tree import val_tree


def run_experiment(configs):
	"""
	Run the experiments for TreeVAE as defined in the config setting. This method will set up the device, the correct
	experimental paths, initialize Wandb for tracking, generate the dataset, train and grow the TreeVAE model, and
	finally it will validate the result. All final results and validations will be stored in Wandb, while the most
	important ones will be also printed out in the terminal. If specified, the model will also be saved for further
	exploration using the Jupyter Notebook: tree_exploration.ipynb.

	Parameters
	----------
	configs: dict
		The config setting for training and validating TreeVAE defined in configs or in the command line.
	"""
	# Setting device on GPU if available, else CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Additional info when using cuda
	if device.type == 'cuda':
		print("Using", torch.cuda.get_device_name(0))
	else:
		print("No GPU available")

	# Set paths
	project_dir = Path(__file__).absolute().parent
	resume_checkpoint = None
	resume_from = configs['globals'].get('resume_from')
	if resume_from:
		resume_checkpoint = load_checkpoint(Path(resume_from), device)
		experiment_path = Path(resume_checkpoint['experiment_path']) 
		ex_name = experiment_path.name
		if not experiment_path.exists():
			raise FileNotFoundError(f"Resume experiment path does not exist: {experiment_path}")
	else:
		timestr = time.strftime("%Y%m%d-%H%M%S")
		ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
		# yml 파일에 result_dir이라는 attribute는 없긴 한데, 이거는 prepare_config()함수에서 정리됨.
		experiment_path = configs['globals']['results_dir'] / configs['data']['data_name'] / ex_name
		experiment_path.mkdir(parents=True)
	os.makedirs(os.path.join(project_dir, '../models/logs', ex_name), exist_ok=True)
	print("Experiment path: ", experiment_path)

	# Wandb
	os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
	os.environ["WANDB_SILENT"] = "true"

	# ADD YOUR WANDB ENTITY
	wandb_kwargs = {
		"project": "treevae",
		"config": configs,
		"group": configs['run_name'],
		"mode": configs['globals']['wandb_logging'],
	}
	if resume_checkpoint is not None and resume_checkpoint.get('wandb_run_id') is not None:
		wandb_kwargs["id"] = resume_checkpoint['wandb_run_id']
		wandb_kwargs["resume"] = "allow"
	wandb.init(**wandb_kwargs)
	configs['globals']['wandb_run_id'] = wandb.run.id if wandb.run is not None else None

	if configs['globals']['wandb_logging'] not in ['online', 'offline', 'disabled']:
		ValueError('wandb needs to be set to online, offline or disabled.')

	# Reproducibility
	reset_random_seeds(configs['globals']['seed'])

	# Generate a new dataset each run
	trainset, trainset_eval, testset = get_data(configs)

	# Run the full training of treeVAE model, including the growing of the tree
	model = run_tree(trainset, trainset_eval, testset, device, configs)

	# Save model
	if configs['globals']['save_model']:
		print("\nSaving weights at ", experiment_path)
		torch.save(model.state_dict(), experiment_path / 'model_weights.pt')

	# Evaluation of TreeVAE
	print("\n" * 2)
	print("Evaluation")
	print("\n" * 2)
	val_tree(trainset_eval, testset, model, device, experiment_path, configs)
	wandb.finish(quiet=True)
	return
