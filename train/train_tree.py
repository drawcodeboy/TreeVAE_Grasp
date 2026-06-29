"""
Training function of TreeVAE and SmallTreeVAE.
"""
import wandb
import numpy as np
import gc
import torch
import torch.optim as optim

from utils.training_utils import train_one_epoch, validate_one_epoch, AnnealKLCallback, Custom_Metrics, \
	get_ind_small_tree, compute_growing_leaf, compute_pruning_leaf, get_optimizer, predict, get_dataset_labels, \
	compute_leaves
from utils.data_utils import get_gen
from utils.checkpoint_utils import (
	CHECKPOINT_FILENAME,
	ResumePhase,
	build_checkpoint,
	restore_rng_state,
	save_checkpoint,
)
from utils.model_utils import (
	return_list_tree,
	construct_data_tree,
	find_node_by_path,
	get_node_path,
	restore_tree_from_topology,
)
from models.model import TreeVAE
from models.model_smalltree import SmallTreeVAE


def run_tree(trainset, trainset_eval, testset, device, configs, resume_checkpoint=None, experiment_path=None):
	"""
	Run the TreeVAE model as defined in the config setting. The method will first train a TreeVAE model with initial
	depth defined in config (initial_depth). After training TreeVAE for epochs=num_epochs, if grow=True then it will
	start the iterative growing schedule. At each step, a SmallTreeVAE will be trained for num_epochs_smalltree and
	attached to the selected leaf of TreeVAE. The resulting TreeVAE will then grow at each step and will be finetuned
	throughout the growing procedure for num_epochs_intermediate_fulltrain and at the end of the growing procedure for
	num_epochs_finetuning.

	Parameters
	----------
	trainset: torch.utils.data.Dataset
		The train dataset
	trainset_eval: torch.utils.data.Dataset
		The validation dataset
	testset: torch.utils.data.Dataset
		The test dataset
	device: torch.device
		The device in which to validate the model
	configs: dict
		The config setting for training and validating TreeVAE defined in configs or in the command line

	Returns
	------
	models.model.TreeVAE
		The trained TreeVAE model
	"""

	graph_mode = not configs['globals']['eager_mode']
	gen_train = get_gen(trainset, configs, validation=False, shuffle=True)
	gen_train_eval = get_gen(trainset_eval, configs, validation=True, shuffle=False)
	gen_test = get_gen(testset, configs, validation=True, shuffle=False)
	_ = gc.collect()

	# Define model & optimizer
	model = TreeVAE(**configs['training'])
	model.to(device)
	if resume_checkpoint is not None and "tree_topology" in resume_checkpoint:
		model = restore_tree_from_topology(model, resume_checkpoint["tree_topology"], configs)
		model.to(device)

	# print(graph_mode); import sys; sys.exit()
	# if graph_mode:
		# model = torch.compile(model)

	optimizer = get_optimizer(model, configs)

	# Initialize schedulers
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'],
											 gamma=configs['training']['decay_lr'])
	alpha_scheduler = AnnealKLCallback(model, decay=configs['training']['decay_kl'],
									   start=configs['training']['kl_start'])

	################################# Obtain RESUME information #################################

	start_epoch = 0
	global_step = 0
	resume_phase = resume_checkpoint['phase'] if resume_checkpoint is not None else None
	if resume_checkpoint is not None:
		if resume_phase not in {
			ResumePhase.INITIAL_TRAINING,
			ResumePhase.GROW_LOOP_BOUNDARY,
			ResumePhase.INTERMEDIATE_FINETUNING,
			ResumePhase.SMALLTREE_TRAINING,
			ResumePhase.ATTACH_DONE,
			ResumePhase.PRUNE_PRECHECK_DONE,
			ResumePhase.PRUNING,
			ResumePhase.FINAL_FINETUNING,
		}:
			raise NotImplementedError(
				f"Resume from phase {resume_phase!r} is not implemented yet. "
				f"Currently supported here: {ResumePhase.INITIAL_TRAINING!r}, "
				f"{ResumePhase.GROW_LOOP_BOUNDARY!r}, "
				f"{ResumePhase.INTERMEDIATE_FINETUNING!r}, "
				f"{ResumePhase.SMALLTREE_TRAINING!r}, "
				f"{ResumePhase.ATTACH_DONE!r}, "
				f"{ResumePhase.PRUNE_PRECHECK_DONE!r}, "
				f"{ResumePhase.PRUNING!r}, "
				f"{ResumePhase.FINAL_FINETUNING!r}."
			)
	if resume_phase == ResumePhase.INITIAL_TRAINING:
		model.load_state_dict(resume_checkpoint['model_state_dict'])
		optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
		lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler_state_dict'])
		model.alpha = resume_checkpoint['alpha'].to(device)
		restore_rng_state(resume_checkpoint['rng_state'])
		start_epoch = resume_checkpoint['phase_epoch'] + 1
		global_step = resume_checkpoint['global_step']
		print(f"Resuming initial training from epoch {start_epoch}")
	elif resume_phase == ResumePhase.FINAL_FINETUNING:
		start_epoch = configs['training']['num_epochs']
		global_step = resume_checkpoint['global_step']
		print("Skipping initial training and tree construction for final finetuning resume")
	elif resume_phase == ResumePhase.ATTACH_DONE:
		model.load_state_dict(resume_checkpoint['model_state_dict'])
		model.alpha = resume_checkpoint['alpha'].to(device)
		restore_rng_state(resume_checkpoint['rng_state'])
		start_epoch = configs['training']['num_epochs']
		global_step = resume_checkpoint['global_step']
		print("Resuming from attach_done checkpoint")
	elif resume_phase in {
		ResumePhase.GROW_LOOP_BOUNDARY,
		ResumePhase.INTERMEDIATE_FINETUNING,
		ResumePhase.SMALLTREE_TRAINING,
	}:
		model.load_state_dict(resume_checkpoint['model_state_dict'])
		model.alpha = resume_checkpoint['alpha'].to(device)
		restore_rng_state(resume_checkpoint['rng_state'])
		start_epoch = configs['training']['num_epochs']
		global_step = resume_checkpoint['global_step']
		print(f"Resuming from {resume_phase} checkpoint")
	elif resume_phase in {ResumePhase.PRUNE_PRECHECK_DONE, ResumePhase.PRUNING}:
		model.load_state_dict(resume_checkpoint['model_state_dict'])
		model.alpha = resume_checkpoint['alpha'].to(device)
		restore_rng_state(resume_checkpoint['rng_state'])
		start_epoch = configs['training']['num_epochs']
		global_step = resume_checkpoint['global_step']
		print(f"Resuming from {resume_phase} checkpoint")

	# Initialize Metrics
	metrics_calc_train = Custom_Metrics(device).to(device)
	metrics_calc_val = Custom_Metrics(device).to(device)

	################################# TRAINING TREEVAE with depth defined in config #################################
	
	# Training the initial tree
	for epoch in range(start_epoch, configs['training']['num_epochs']):  # loop over the dataset multiple times
		train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device, configs=configs)
		validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device, configs=configs)
		lr_scheduler.step()
		alpha_scheduler.on_epoch_end(epoch)
		global_step += len(gen_train)
		if experiment_path is not None:
			checkpoint = build_checkpoint(
				phase=ResumePhase.INITIAL_TRAINING,
				phase_epoch=epoch,
				global_step=global_step,
				model=model,
				optimizer=optimizer,
				lr_scheduler=lr_scheduler,
				alpha=model.alpha,
				configs=configs,
				experiment_path=experiment_path,
				wandb_run_id=configs['globals'].get('wandb_run_id'),
			)
			save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)
		_ = gc.collect()

	################################# GROWING THE TREE #################################

	# Start the growing loop of the tree
	# Compute metrics and set node.expand False for the nodes that should not grow
	# This loop goes layer-wise
	grow = configs['training']['grow']
	if resume_phase in {ResumePhase.PRUNE_PRECHECK_DONE, ResumePhase.PRUNING, ResumePhase.FINAL_FINETUNING}:
		grow = False
	initial_depth = configs['training']['initial_depth']
	max_depth = len(configs['training']['mlp_layers']) - 1
	if initial_depth >= max_depth:
		grow = False
	growing_iterations = 0
	if resume_phase == ResumePhase.ATTACH_DONE:
		growing_iterations = resume_checkpoint.get('growing_iterations', resume_checkpoint['phase_epoch']) + 1
		print(f"Continuing grow loop from iteration {growing_iterations}")
	elif resume_phase in {
		ResumePhase.GROW_LOOP_BOUNDARY,
		ResumePhase.INTERMEDIATE_FINETUNING,
		ResumePhase.SMALLTREE_TRAINING,
	}:
		growing_iterations = resume_checkpoint.get('growing_iterations', resume_checkpoint['phase_epoch'])
		print(f"Continuing grow loop from iteration {growing_iterations}")


	while grow and growing_iterations < 150:
		resuming_smalltree = (
			resume_phase == ResumePhase.SMALLTREE_TRAINING
			and growing_iterations == resume_checkpoint['phase_epoch']
		)
		resuming_intermediate = (
			resume_phase == ResumePhase.INTERMEDIATE_FINETUNING
			and growing_iterations == resume_checkpoint['phase_epoch']
		)

		# full model finetuning during growing after every 3 splits
		# Intermediate full-tree fine tuning
		# Small Tree를 학습시키고 계속 붙임에 따라 '국소적으로'는 각 leaf를 잘 쪼갠 subtree가 생기겠지만,
		# 전체 TreeVAE 관점에서는 parent router, 기존 latent path, 새 child router/decoder가 한 objective로
		# 같이 맞춰진 상태는 아니다. 그래서, 전체 TreeVAE를 3번에 한 번씩 학습한다.
		if configs['training']['num_epochs_intermediate_fulltrain']>0 and not resuming_smalltree:
			if resuming_intermediate or (growing_iterations != 0 and growing_iterations % 3 == 0):
				# Initialize optimizer and schedulers
				optimizer = get_optimizer(model, configs)
				lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'],
														 gamma=configs['training']['decay_lr'])
				alpha_scheduler = AnnealKLCallback(model, decay=configs['training']['decay_kl'],
												   start=configs['training']['kl_start'])
				intermediate_start_epoch = 0
				if resuming_intermediate:
					optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
					lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler_state_dict'])
					model.alpha = resume_checkpoint['alpha'].to(device)
					intermediate_start_epoch = resume_checkpoint['intermediate_epoch'] + 1
					print(f"Resuming intermediate finetuning from epoch {intermediate_start_epoch}")

				# Training the initial split
				print('\nTree intermediate finetuning\n')
				for epoch in range(intermediate_start_epoch, configs['training']['num_epochs_intermediate_fulltrain']):
					train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device, configs=configs)
					validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device, configs=configs)
					lr_scheduler.step()
					alpha_scheduler.on_epoch_end(epoch)
					global_step += len(gen_train)
					if experiment_path is not None:
						checkpoint = build_checkpoint(
							phase=ResumePhase.INTERMEDIATE_FINETUNING,
							phase_epoch=growing_iterations,
							global_step=global_step,
							model=model,
							optimizer=optimizer,
							lr_scheduler=lr_scheduler,
							alpha=model.alpha,
							configs=configs,
							experiment_path=experiment_path,
							wandb_run_id=configs['globals'].get('wandb_run_id'),
							extra_state={
								"growing_iterations": growing_iterations,
								"intermediate_epoch": epoch,
							},
						)
						save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)
					_ = gc.collect()
				if resuming_intermediate:
					resume_phase = None

		if resuming_smalltree:
			selected_leaf_path = resume_checkpoint['selected_leaf_path']
			selected_leaf_node = find_node_by_path(model.tree, selected_leaf_path)
			leaves = compute_leaves(model.tree, configs['training']['n_ary'])
			ind_leaf = None
			for leaf_idx, candidate_leaf in enumerate(leaves):
				if candidate_leaf['node'] is selected_leaf_node:
					ind_leaf = leaf_idx
					leaf = candidate_leaf
					break
			if ind_leaf is None:
				raise ValueError(
					f"Could not find selected leaf path {selected_leaf_path} in restored tree."
				)
			if ind_leaf != resume_checkpoint['selected_leaf_index']:
				print(
					"Selected leaf index changed after topology restore: "
					f"checkpoint={resume_checkpoint['selected_leaf_index']}, restored={ind_leaf}"
				)
			n_effective_leaves = resume_checkpoint['n_effective_leaves']
			node_leaves_train = predict(gen_train_eval, model, device, 'node_leaves', configs=configs)
			node_leaves_test = predict(gen_test, model, device, 'node_leaves', configs=configs)
			print('\nResuming smalltree training: Leaf %d at depth %d\n' % (ind_leaf, leaf['depth']))
		else:
			# extract information of leaves
			node_leaves_train = predict(gen_train_eval, model, device, 'node_leaves', configs=configs)
			node_leaves_test = predict(gen_test, model, device, 'node_leaves', configs=configs)

			# compute which leaf to grow and split
			ind_leaf, leaf, n_effective_leaves = compute_growing_leaf(gen_train_eval, model, node_leaves_train, max_depth,
																	  configs['training']['batch_size'],
																	  max_leaves=configs['training']['num_clusters_tree'])
			if ind_leaf == None:
				break
			else:
				print('\nGrowing tree: Leaf %d at depth %d\n' % (ind_leaf, leaf['depth']))
		depth, node = leaf['depth'], leaf['node']
		selected_leaf_path = get_node_path(model.tree, node)

		# get subset of data that has high prob. of falling in subtree
		ind_train = get_ind_small_tree(node_leaves_train[ind_leaf], n_effective_leaves)
		ind_test = get_ind_small_tree(node_leaves_test[ind_leaf], n_effective_leaves)
		gen_train_small = get_gen(trainset, configs, shuffle=True, smalltree=True, smalltree_ind=ind_train)
		gen_test_small = get_gen(testset, configs, shuffle=False, validation=True, smalltree=True,
								 smalltree_ind=ind_test)

		# preparation for the smalltree training
		# initialize the smalltree
		small_model = SmallTreeVAE(depth=depth+1, **configs['training'])
		small_model.to(device)
		# if graph_mode:
		# 	small_model = torch.compile(small_model)

		# Optimizer for smalltree
		optimizer = get_optimizer(small_model, configs)

		# Initialize schedulers
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'],
												 gamma=configs['training']['decay_lr'])
		alpha_scheduler = AnnealKLCallback(small_model, decay=configs['training']['decay_kl'],
										   start=configs['training']['kl_start'])
		smalltree_start_epoch = 0
		if resuming_smalltree:
			small_model.load_state_dict(resume_checkpoint['small_model_state_dict'])
			optimizer.load_state_dict(resume_checkpoint['small_optimizer_state_dict'])
			lr_scheduler.load_state_dict(resume_checkpoint['small_lr_scheduler_state_dict'])
			small_alpha = resume_checkpoint['small_alpha']
			small_model.alpha = small_alpha.to(device) if torch.is_tensor(small_alpha) else torch.tensor(small_alpha, device=device)
			smalltree_start_epoch = resume_checkpoint['smalltree_epoch'] + 1
			print(f"Resuming smalltree training from epoch {smalltree_start_epoch}")

		# Training the smalltree subsplit
		for epoch in range(smalltree_start_epoch, configs['training']['num_epochs_smalltree']):
			train_one_epoch(gen_train_small, model, optimizer, metrics_calc_train, epoch, device, train_small_tree=True,
							small_model=small_model, ind_leaf=ind_leaf, configs=configs)
			validate_one_epoch(gen_test_small, model, metrics_calc_val, epoch, device, train_small_tree=True,
							   small_model=small_model, ind_leaf=ind_leaf, configs=configs)
			lr_scheduler.step()
			alpha_scheduler.on_epoch_end(epoch)
			global_step += len(gen_train_small)
			if experiment_path is not None:
				checkpoint = build_checkpoint(
					phase=ResumePhase.SMALLTREE_TRAINING,
					phase_epoch=growing_iterations,
					global_step=global_step,
					model=model,
					optimizer=optimizer,
					lr_scheduler=lr_scheduler,
					alpha=model.alpha,
					configs=configs,
					experiment_path=experiment_path,
					wandb_run_id=configs['globals'].get('wandb_run_id'),
					extra_state={
						"growing_iterations": growing_iterations,
						"small_model_state_dict": small_model.state_dict(),
						"small_optimizer_state_dict": optimizer.state_dict(),
						"small_lr_scheduler_state_dict": lr_scheduler.state_dict(),
						"small_alpha": small_model.alpha.detach().cpu() if torch.is_tensor(small_model.alpha) else small_model.alpha,
						"smalltree_epoch": epoch,
						"selected_leaf_index": ind_leaf,
						"selected_leaf_path": selected_leaf_path,
						"n_effective_leaves": n_effective_leaves,
					},
				)
				save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)
			_ = gc.collect()

		# attach smalltree to full tree by assigning decisions and adding new children nodes to full tree
		model.attach_smalltree(node, small_model)
		if resuming_smalltree:
			resume_phase = None
		if experiment_path is not None:
			attach_optimizer = get_optimizer(model, configs)
			attach_lr_scheduler = optim.lr_scheduler.StepLR(
				attach_optimizer,
				step_size=configs['training']['decay_stepsize'],
				gamma=configs['training']['decay_lr'],
			)
			checkpoint = build_checkpoint(
				phase=ResumePhase.ATTACH_DONE,
				phase_epoch=growing_iterations,
				global_step=global_step,
				model=model,
				optimizer=attach_optimizer,
				lr_scheduler=attach_lr_scheduler,
				alpha=model.alpha,
				configs=configs,
				experiment_path=experiment_path,
				wandb_run_id=configs['globals'].get('wandb_run_id'),
				extra_state={"growing_iterations": growing_iterations},
			)
			save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)

		# Check if reached the max number of effective leaves before finetuning unnecessarily
		# split 할 때마다 leaf의 증가량은 n_ary - 1이다.
		# n_effective_leaves: 
		# configs['training']['num_clusters_tree']: 트리가 가질 수 있는 최대 leaf 노드의 수

		# 다음 split을 붙인 뒤 effective leaf 수가 목표 cluster 수에 도달했거나, 더 grow하면 목표 수를 넘길 것 같으면 grow loop를 끝내자
		# predict, compute_growing_leaf가 무엇인지 파악이 필요하네 3번째 return이 effective leaf nodes인 거 같은데
		leaf_increment = configs['training']['n_ary'] - 1
		if n_effective_leaves + leaf_increment >= configs['training']['num_clusters_tree']:
			# node_leaves_train : leaf 별 정보를 담은 list, 각 원소는 dict -> {'prob': sample-wise probability of reaching the node, 'z_sample': sampled leaf embedding}
			node_leaves_train = predict(gen_train_eval, model, device, 'node_leaves', configs=configs)
			_, _, max_growth = compute_growing_leaf(gen_train_eval, model, node_leaves_train, max_depth,
													configs['training']['batch_size'],
													max_leaves=configs['training']['num_clusters_tree'], check_max=True)
			if max_growth is True:
				break

		growing_iterations += 1
		if experiment_path is not None:
			grow_optimizer = get_optimizer(model, configs)
			grow_lr_scheduler = optim.lr_scheduler.StepLR(
				grow_optimizer,
				step_size=configs['training']['decay_stepsize'],
				gamma=configs['training']['decay_lr'],
			)
			checkpoint = build_checkpoint(
				phase=ResumePhase.GROW_LOOP_BOUNDARY,
				phase_epoch=growing_iterations,
				global_step=global_step,
				model=model,
				optimizer=grow_optimizer,
				lr_scheduler=grow_lr_scheduler,
				alpha=model.alpha,
				configs=configs,
				experiment_path=experiment_path,
				wandb_run_id=configs['globals'].get('wandb_run_id'),
				extra_state={"growing_iterations": growing_iterations},
			)
			save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)

	#-----------------------------------------------------------------------------------------

	# The growing loop of the tree is concluded!
	# check whether we need to prune the final tree and log pre-pruning dendrogram
	prune = configs['training']['prune']
	pruning_iterations = 0
	if resume_phase == ResumePhase.FINAL_FINETUNING:
		prune = False
	if resume_phase == ResumePhase.PRUNE_PRECHECK_DONE:
		prune = resume_checkpoint.get('prune', configs['training']['prune'])
		pruning_iterations = resume_checkpoint.get('pruning_iterations', 0)
		print("Skipping pruning precheck; it was already completed")
	elif resume_phase == ResumePhase.PRUNING:
		prune = not resume_checkpoint.get('pruning_complete', False)
		pruning_iterations = resume_checkpoint.get('pruning_iterations', 0)
		if prune:
			print(f"Continuing pruning loop from iteration {pruning_iterations}")
		else:
			print("Pruning was already completed in checkpoint")
	elif prune:
		node_leaves_test, prob_leaves_test = predict(gen_test, model, device, 'node_leaves', 'prob_leaves', configs=configs)
		if len(node_leaves_test)<2:
			prune = False
		else:
			print('\nStarting pruning!\n')
			yy = np.squeeze(np.argmax(prob_leaves_test, axis=-1))
			y_test = get_dataset_labels(testset)
			data_tree = construct_data_tree(model, y_predicted=yy, y_true=y_test, n_leaves=len(node_leaves_test),
											data_name=configs['data']['data_name'], n_ary=configs['training']['n_ary'])

			# Pruning 하기 전 tree 구조를 기록
			table = wandb.Table(columns=["node_id", "node_name", "parent", "size"], data=data_tree)
			fields = {"node_name": "node_name", "node_id": "node_id", "parent": "parent", "size": "size"}
			dendro = wandb.plot_table(vega_spec_name="stacey/flat_tree", data_table=table, fields=fields)
			wandb.log({"dendogram_pre_pruned": dendro})
		if experiment_path is not None:
			prune_optimizer = get_optimizer(model, configs)
			prune_lr_scheduler = optim.lr_scheduler.StepLR(
				prune_optimizer,
				step_size=configs['training']['decay_stepsize'],
				gamma=configs['training']['decay_lr'],
			)
			checkpoint = build_checkpoint(
				phase=ResumePhase.PRUNE_PRECHECK_DONE,
				phase_epoch=0,
				global_step=global_step,
				model=model,
				optimizer=prune_optimizer,
				lr_scheduler=prune_lr_scheduler,
				alpha=model.alpha,
				configs=configs,
				experiment_path=experiment_path,
				wandb_run_id=configs['globals'].get('wandb_run_id'),
				extra_state={
					"prune": prune,
					"pruning_iterations": pruning_iterations,
				},
			)
			save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)

	# prune the tree
	while prune:
		# check pruning conditions
		node_leaves_train = predict(gen_train_eval, model, device, 'node_leaves', configs=configs)
		ind_leaf, leaf = compute_pruning_leaf(model, node_leaves_train)

		if ind_leaf == None:
			print('\nPruning finished!\n')
			if experiment_path is not None:
				prune_optimizer = get_optimizer(model, configs)
				prune_lr_scheduler = optim.lr_scheduler.StepLR(
					prune_optimizer,
					step_size=configs['training']['decay_stepsize'],
					gamma=configs['training']['decay_lr'],
				)
				checkpoint = build_checkpoint(
					phase=ResumePhase.PRUNING,
					phase_epoch=pruning_iterations,
					global_step=global_step,
					model=model,
					optimizer=prune_optimizer,
					lr_scheduler=prune_lr_scheduler,
					alpha=model.alpha,
					configs=configs,
					experiment_path=experiment_path,
					wandb_run_id=configs['globals'].get('wandb_run_id'),
					extra_state={
						"pruning_iterations": pruning_iterations,
						"pruning_complete": True,
					},
				)
				save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)
			break
		else:
			# prune leaves and internal nodes without children
			print(f'\nPruning leaf {ind_leaf}!\n')
			current_node = leaf['node']

			if model.n_ary == 2:
				while all(child is None for child in [current_node.left, current_node.right]):
					if current_node.parent is not None:
						parent = current_node.parent
					# root does not get pruned
					else:
						break
					parent.prune_child(current_node)
					current_node = parent
			elif model.n_ary > 2:
				while not current_node.has_children():
					if current_node.parent is not None:
						parent = current_node.parent
					# root does not get pruned
					else:
						break
					parent.prune_child(current_node)
					current_node = parent

			# reinitialize model
			transformations, routers, denses, decoders, routers_q = return_list_tree(model.tree)
			model.decisions_q = routers_q
			model.transformations = transformations
			model.decisions = routers
			model.denses = denses
			model.decoders = decoders
			model.depth = model.compute_depth()
			pruning_iterations += 1
			if experiment_path is not None:
				prune_optimizer = get_optimizer(model, configs)
				prune_lr_scheduler = optim.lr_scheduler.StepLR(
					prune_optimizer,
					step_size=configs['training']['decay_stepsize'],
					gamma=configs['training']['decay_lr'],
				)
				checkpoint = build_checkpoint(
					phase=ResumePhase.PRUNING,
					phase_epoch=pruning_iterations,
					global_step=global_step,
					model=model,
					optimizer=prune_optimizer,
					lr_scheduler=prune_lr_scheduler,
					alpha=model.alpha,
					configs=configs,
					experiment_path=experiment_path,
					wandb_run_id=configs['globals'].get('wandb_run_id'),
					extra_state={
						"pruning_iterations": pruning_iterations,
						"pruned_leaf_index": ind_leaf,
						"pruning_complete": False,
					},
				)
				save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)
	_ = gc.collect()

	################################# FULL MODEL FINETUNING #################################


	print('\n*****************model depth %d******************\n' % (model.depth))
	print('\n*****************model finetuning******************\n')

	# Initialize optimizer and schedulers
	optimizer = get_optimizer(model, configs)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['training']['decay_stepsize'], gamma=configs['training']['decay_lr'])
	alpha_scheduler = AnnealKLCallback(model, decay=max(0.01,1/max(1,configs['training']['num_epochs_finetuning']-1)), start=configs['training']['kl_start'])
	final_start_epoch = 0
	if resume_phase == ResumePhase.FINAL_FINETUNING:
		try:
			model.load_state_dict(resume_checkpoint['model_state_dict'])
		except RuntimeError as error:
			raise RuntimeError(
				"Final finetuning resume failed because the checkpoint model structure "
				"does not match the restored TreeVAE structure. Check that the checkpoint "
				"contains the matching tree_topology and was created with compatible model settings."
			) from error
		optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
		lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler_state_dict'])
		model.alpha = resume_checkpoint['alpha'].to(device)
		restore_rng_state(resume_checkpoint['rng_state'])
		final_start_epoch = resume_checkpoint['phase_epoch'] + 1
		print(f"Resuming final finetuning from epoch {final_start_epoch}")
	# finetune the full tree
	print('\nTree final finetuning\n')
	for epoch in range(final_start_epoch, configs['training']['num_epochs_finetuning']):  # loop over the dataset multiple times
		train_one_epoch(gen_train, model, optimizer, metrics_calc_train, epoch, device, configs=configs)
		validate_one_epoch(gen_test, model, metrics_calc_val, epoch, device, configs=configs)
		lr_scheduler.step()
		alpha_scheduler.on_epoch_end(epoch)
		global_step += len(gen_train)
		if experiment_path is not None:
			checkpoint = build_checkpoint(
				phase=ResumePhase.FINAL_FINETUNING,
				phase_epoch=epoch,
				global_step=global_step,
				model=model,
				optimizer=optimizer,
				lr_scheduler=lr_scheduler,
				alpha=model.alpha,
				configs=configs,
				experiment_path=experiment_path,
				wandb_run_id=configs['globals'].get('wandb_run_id'),
			)
			save_checkpoint(checkpoint, experiment_path / CHECKPOINT_FILENAME)
		_ = gc.collect()

	return model
