"""
General utility functions.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.special import comb
import torch
import os
import random
from pathlib import Path
import yaml


def cluster_acc(y_true, y_pred, return_index=False):
	"""
	Calculate clustering accuracy.
	# Arguments
		y: true labels, numpy.array with shape `(n_samples,)`
		y_pred: predicted labels, numpy.array with shape `(n_samples,)`
	# Return
		accuracy, in [0,1]
	"""
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
	w = np.zeros((int(D), (D)), dtype=np.int64)
	for i in range(y_pred.size):
		w[int(y_pred[i]), int(y_true[i])] += 1
	ind = np.array(linear_assignment(w.max() - w))
	if return_index:
		assert all(ind[0] == range(len(ind[0])))  # Assert rows don't change order
		cluster_acc = sum(w[ind[0], ind[1]]) * 1.0 / y_pred.size
		return cluster_acc, ind[1]
	else:
		return sum([w[ind[0,i], ind[1,i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size


def reset_random_seeds(seed):
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	# No determinism as nn.Upsample has no deterministic implementation
	#torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


def merge_yaml_args(configs, args):
	arg_dict = args.__dict__
	# yml 파일에 parser가 있는 경우도 있는데, num_workers 같은 걸 여기서 할당
	# 이렇게 새롭게 dictionary를 선언해버리면, parser 정보가 날아가는 거 아닌가?
	configs['parser'] = dict()

	# yml 파일에서 이중 dictionary를 쓰면 안 되겠네. 애초에 안 되는 건가?
	# 
	for key, value in arg_dict.items():
		flag = True
		# Replace/Create values in config if they are defined by arg in parser.
		if arg_dict[key] is not None:
			for key_config in configs.keys():
				# If value of config is dict itself, then search key-value pairs inside this dict for matching the arg
				if type(configs[key_config]) is dict:
					for key2, value2 in configs[key_config].items():
						if key == key2:
							configs[key_config][key2] = value
							flag = False
				# If value of config is not a dict, check whether key matches to the arg
				else:
					if key == key_config:
						configs[key_config] = value
						flag = False
				# Break out of loop if key got replaced
				if flag == False:
					break
			# If arg does not match any keys of config, define a new key
			else:
				print("Could not find this key in config, therefore adding it:", key)
				configs['parser'][key] = arg_dict[key]
	return configs


def prepare_config(args, project_dir):
	# Load config
	data_name = args.config_name +'.yml'
	config_path = project_dir / 'configs' / data_name

	with config_path.open(mode='r') as yamlfile:
		configs = yaml.safe_load(yamlfile)

	# Override config if args in parser
	# YAML에 있던 config에 argparse로 받은 인자 override
	configs = merge_yaml_args(configs, args)

	if isinstance(configs['training']['latent_dim'], str):
		a = configs['training']['latent_dim'].split(",")
		configs['training']['latent_dim'] = [int(i) for i in a]
	if isinstance(configs['training']['mlp_layers'], str):
		a = configs['training']['mlp_layers'].split(",")
		configs['training']['mlp_layers'] = [int(i) for i in a]
	
	a = configs['training']['augmentation_method'].split(",")
	configs['training']['augmentation_method'] = [str(i) for i in a]

	configs['globals']['results_dir'] = os.path.join(project_dir, 'models/experiments')
	configs['globals']['results_dir'] = Path(configs['globals']['results_dir']).absolute()

	# Prepare for passing x' and x'' through model by setting batch size to an even number
	if configs['training']['augment'] is True and configs['training']['augmentation_method'] != ['simple'] and configs['training']['batch_size'] % 2 != 0:
		configs['training']['batch_size'] += 1

	# for n-ary tree
	if 'n_ary' not in configs['training']:
		configs['training']['n_ary'] = 2

	configs['training']['n_ary'] = int(configs['training']['n_ary'])

	if configs['training']['n_ary'] < 2:
		raise ValueError("n_ary must be at least 2")

	# for n-ary tree: Check that num_clusters_tree is at least n_ary
	# tree가 가져야 할 leaf node의 최대 개수는 n_ary보다 커야 트리가 제대로 성장할 수 있기 때문
	n_ary = configs['training']['n_ary']
	num_clusters_tree = configs['training']['num_clusters_tree']

	if num_clusters_tree < n_ary:
		raise ValueError("num_clusters_tree must be at least n_ary")
		
	return configs

def count_values_in_sequence(seq):
	from collections import defaultdict
	res = defaultdict(lambda : 0)
	for key in seq:
		res[key] += 1
	return dict(res)


def dendrogram_purity(tree_root, ground_truth, ind_samples_of_leaves, n_ary=2):
	total_per_label_frequencies = count_values_in_sequence(ground_truth)
	total_per_label_pairs_count = {k: comb(v, 2, True) for k, v in total_per_label_frequencies.items()}
	total_n_of_pairs = sum(total_per_label_pairs_count.values())
	one_div_total_n_of_pairs = 1. / total_n_of_pairs
	purity = 0.

	def calculate_purity(node, level):
		nonlocal purity
		if node.decoder:
			# Match node to leaf samples
			ind_leaf = np.where([node == ind_samples_of_leaves[ind_leaf][0] for ind_leaf in range(len(ind_samples_of_leaves))])[0].item()
			ind_samples_of_leaf = ind_samples_of_leaves[ind_leaf][1]
			node_total_dp_count = len(ind_samples_of_leaf)
			# Count how many samples of given leaf fall into which ground-truth class (-> For treevae make use of ground_truth(to which class a sample belongs)&yy(into which leaf a sample falls))
			node_per_label_frequencies = count_values_in_sequence(
				[ground_truth[id] for id in ind_samples_of_leaf])
			# From above, deduct how many pairs will fall into same leaf
			node_per_label_pairs_count = {k: comb(v, 2, True) for k, v in node_per_label_frequencies.items()}
		
		elif node.router is None and node.decoder is None:
			# We are in an internal node with pruned leaves and thus only have one child. Therefore no prunity calculation here!
			if n_ary == 2:
				node_left, node_right = node.left, node.right
				child = node_left if node_left is not None else node_right
			else:
				child = node.single_child()
			node_per_label_frequencies, node_total_dp_count = calculate_purity(child, level + 1)	
			return node_per_label_frequencies, node_total_dp_count
		
		else:  
			# it is an inner splitting node
			if n_ary == 2:
				left_child_per_label_freq, left_child_total_dp_count = calculate_purity(node.left, level + 1)
				right_child_per_label_freq, right_child_total_dp_count = calculate_purity(node.right, level + 1)
				node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
				# Count how many samples of given internal node fall into which ground-truth class (=sum of their children's values)
				node_per_label_frequencies = {k: left_child_per_label_freq.get(k, 0) + right_child_per_label_freq.get(k, 0) \
											for k in set(left_child_per_label_freq) | set(right_child_per_label_freq)}
				
				# Class-wisedly count how many pairs of samples of a class will have this node as least common ancestor (=mult. of their children's values, bcs this is all possible pairs coming from different sides)
				node_per_label_pairs_count = {k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k) \
											for k in set(left_child_per_label_freq) & set(right_child_per_label_freq)}
			else:
				child_results = [calculate_purity(child, level + 1) for child in node.active_children()]
				child_per_label_freqs = [child_per_label_freq for child_per_label_freq, _ in child_results]
				node_total_dp_count = sum(child_total_dp_count for _, child_total_dp_count in child_results)

				# Count how many samples of given internal node fall into which ground-truth class (=sum of their children's values)
				node_per_label_frequencies = {}
				for child_per_label_freq in child_per_label_freqs:
					for label, label_freq in child_per_label_freq.items():
						node_per_label_frequencies[label] = node_per_label_frequencies.get(label, 0) + label_freq
				
				# For n-ary splits, pairs whose LCA is this node can come from any pair of distinct children.
				node_per_label_pairs_count = {}
				for i in range(len(child_per_label_freqs)):
					for j in range(i + 1, len(child_per_label_freqs)):
						for label in set(child_per_label_freqs[i]) & set(child_per_label_freqs[j]):
							node_per_label_pairs_count[label] = node_per_label_pairs_count.get(label, 0) + \
								child_per_label_freqs[i][label] * child_per_label_freqs[j][label]

		# Given the class-wise number of pairs with given node as least common ancestor node, calculate their purity
		for label, pair_count in node_per_label_pairs_count.items():
			label_freq = node_per_label_frequencies[label]
			label_pairs = node_per_label_pairs_count[label]
			purity += one_div_total_n_of_pairs * label_freq / node_total_dp_count * label_pairs # (1/n_all_pairs) * purity(=n_samples_of_this_class_in_node/n_samples) * n_class_pairs_with_this_node_being_least_common_ancestor(this last term represents sum over pairs with this node being least common ancestor)
		return node_per_label_frequencies, node_total_dp_count

	calculate_purity(tree_root, 0)
	return purity


def leaf_purity(tree_root, ground_truth, ind_samples_of_leaves, n_ary=2):
	values = [] # purity rate per leaf
	weights = [] # n_samples per leaf
	# For each leaf calculate the maximum over classes for in-leaf purity (i.e. majority class / n_samples_in_leaf)
	def get_leaf_purities(node):
		nonlocal values
		nonlocal weights
		if node.decoder:
			ind_leaf = np.where([node == ind_samples_of_leaves[ind_leaf][0] for ind_leaf in range(len(ind_samples_of_leaves))])[0].item()
			ind_samples_of_leaf = ind_samples_of_leaves[ind_leaf][1]
			node_total_dp_count = len(ind_samples_of_leaf)
			node_per_label_counts = count_values_in_sequence(
				[ground_truth[id] for id in ind_samples_of_leaf])
			if node_total_dp_count > 0:
				purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
			else:
				purity_rate = 1.0
			values.append(purity_rate)
			weights.append(node_total_dp_count)
		elif node.router is None and node.decoder is None:
			# We are in an internal node with pruned leaves and thus only have one child.
			if n_ary == 2:
				node_left, node_right = node.left, node.right
				child = node_left if node_left is not None else node_right
			else:
				child = node.single_child()
			get_leaf_purities(child)	
		else:
			if n_ary == 2:
				get_leaf_purities(node.left)
				get_leaf_purities(node.right)
			else:
				for child in node.active_children():
					get_leaf_purities(child)

	get_leaf_purities(tree_root)
	assert len(values) == len(ind_samples_of_leaves), "Didn't iterate through all leaves"
	# Return mean leaf_purity
	return np.average(values, weights=weights)

def display_image(image):
    assert image.dim() == 3 
    if image.size()[0] == 1:
        return torch.clamp(image.squeeze(0),0,1)
    elif image.size()[0] == 3:
        return torch.clamp(image.permute(1, 2, 0),0,1)
    elif image.size()[-1] == 3:
        return torch.clamp(image,0,1)
    else:
        raise NotImplementedError
