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
from utils.checkpoint_utils import ResumePhase, load_checkpoint, validate_resume_config
from train.train_tree import run_tree
from train.validate_tree import val_tree
from models.CoMA import mesh_operations

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
		validate_resume_config(
			resume_checkpoint['configs'],
			configs,
			strict=resume_checkpoint['phase'] != ResumePhase.INITIAL_TRAINING,
		)
		experiment_path = Path(resume_checkpoint['experiment_path']) 
		ex_name = experiment_path.name
		if not experiment_path.exists():
			raise FileNotFoundError(f"Resume experiment path does not exist: {experiment_path}")
	else:
		timestr = time.strftime("%Y%m%d-%H%M%S")
		ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
		# yml 파일에 result_dir이라는 attribute는 없긴 한데, 이거는 prepare_config()함수에서 정리됨.
		experiment_path = configs['globals']['results_dir'] / configs['run_name'] / ex_name
		experiment_path.mkdir(parents=True)
	os.makedirs(os.path.join(project_dir, '../models/logs', ex_name), exist_ok=True)
	print("Experiment path: ", experiment_path)

	# Wandb
	os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, '../wandb', '.cache', 'wandb')
	os.environ["WANDB_SILENT"] = "true"
	if configs['globals']['wandb_logging'] not in ['online', 'offline', 'disabled']:
		raise ValueError('wandb needs to be set to online, offline or disabled.')

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
	wandb.define_metric('wandb_epoch_step')
	epoch_metric_names = (
		'loss_value',
		'rec_loss',
		'kl_decisions',
		'kl_root',
		'kl_nodes',
		'aug_decisions',
		'perc_samples',
		'nmi',
		'accuracy',
	)
	for metric_name in epoch_metric_names:
		wandb.define_metric(
			f'train/{metric_name}',
			step_metric='wandb_epoch_step',
		)
		wandb.define_metric(
			f'validation/{metric_name}',
			step_metric='wandb_epoch_step',
		)
	wandb.define_metric(
		'train/alpha',
		step_metric='wandb_epoch_step',
	)
	configs['globals']['wandb_run_id'] = wandb.run.id if wandb.run is not None else None

	# Reproducibility
	reset_random_seeds(configs['globals']['seed'])

	# Generate a new dataset each run
	trainset, trainset_eval, testset = get_data(configs)

	# Here, Mesh
	if configs['training']['modal'] == 'mesh':  # MANO for CoMA
		default_mano_path = Path(
			'/workspace/dwkwon/HOGraspNet/thirdparty/'
			'mano_v1_2/models/MANO_RIGHT.pkl'
		)
		mano_model_path = configs['training'].get(
			'mano_model_path',
			default_mano_path,
		)
		template_mesh = mesh_operations.load_mano_template(mano_model_path)
		M, A, D, U = mesh_operations.generate_transform_matrices(
			template_mesh,
			configs['training']['downsampling_factors'],
		)

		D_t = [mesh_operations.scipy_to_torch_sparse(d).to(device) for d in D]
		U_t = [mesh_operations.scipy_to_torch_sparse(u).to(device) for u in U]
		A_t = [mesh_operations.scipy_to_torch_sparse(a).to(device) for a in A]
		num_nodes_mesh = [len(M[i].v) for i in range(len(M))]

		configs['training']['encoder']['D_t'] = D_t
		configs['training']['encoder']['U_t'] = U_t
		configs['training']['encoder']['A_t'] = A_t
		configs['training']['encoder']['num_nodes_mesh'] = num_nodes_mesh

	# Run the full training of treeVAE model, including the growing of the tree
	model = run_tree(
		trainset,
		trainset_eval,
		testset,
		device,
		configs,
		resume_checkpoint=resume_checkpoint,
		experiment_path=experiment_path,
	)

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
