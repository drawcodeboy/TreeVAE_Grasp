"""
Utility functions for model.
"""
import numpy as np
import torch.nn as nn

def compute_posterior(mu_q, mu_p, sigma_q, sigma_p):
	epsilon = 1e-7 
	z_sigma_q = 1 / (1 / (sigma_q + epsilon) + 1 / (sigma_p + epsilon))
	z_mu_q = (mu_q / (sigma_q + epsilon) +
			  mu_p / (sigma_p + epsilon)) * z_sigma_q
	return z_mu_q, z_sigma_q


def construct_tree(transformations, routers, routers_q, denses, decoders, n_ary=2):
	"""
		Construct the tree by passing a list of transformations and routers from root to leaves visiting nodes
		layer-wise from left to right

		:param transformations: list of transformations to attach to the nodes of the tree
		:param routers: list of decisions to attach to the nodes of the tree
		:param denses: list of dense network that from d of the bottom up compute node-specific q
		:param decoders: list of decoders to attach to the nodes, they should be set to None except the leaves
		:return: the root of the tree
		"""
	if len(transformations) != len(routers) and len(transformations) != len(denses) \
			and len(transformations) != len(decoders):
		raise ValueError('Len transformation is different than len routers in constructing the tree.')
	root = Node(transformation=transformations[0], router=routers[0], routers_q=routers_q[0], dense=denses[0], decoder=decoders[0], n_ary=n_ary)
	for i in range(1, len(transformations)):
		# 0은 root임 root의 child를 insert하는 과정이라 보면 됨.
		root.insert(transformation=transformations[i], router=routers[i], routers_q=routers_q[i], dense=denses[i], decoder=decoders[i])
	return root


class Node:
	def __init__(self, transformation, router, routers_q, dense, decoder=None, expand=True, n_ary=2):
		self.left = None
		self.right = None
		self.parent = None
		self.transformation = transformation
		self.dense = dense
		self.router = router
		self.routers_q = routers_q
		self.decoder = decoder
		self.expand = expand
		self.n_ary = n_ary # Added by Dawoon Kwon
		self.children = [None for _ in range(n_ary)] # Added by Dawoon Kwon

	def child_slots(self):
		if self.n_ary == 2:
			return [self.left, self.right]
		return self.children

	def active_children(self):
		return [child for child in self.child_slots() if child is not None]

	def active_child_indices(self):
		# None이 아닌 child의 index만 담은 list를 return
		return [idx for idx, child in enumerate(self.child_slots()) if child is not None]

	def single_child(self):
		active_children = self.active_children()
		if len(active_children) != 1:
			raise ValueError("Expected exactly one active child.")
		return active_children[0]

	def has_children(self):
		return any(child is not None for child in self.child_slots())

	def insert(self, transformation=None, router=None, routers_q=None, dense=None, decoder=None):
		'''
		이 함수는 기존 트리에 노드를 하나 삽입하는 역할을 수행한다.
		left 노드가 없으면, left 노드를 하나 삽입하고 함수를 종료
		left 노드가 있으면, right 노드를 하나 삽입하고 함수를 종료
		left, right 노드가 둘 다 있으면, left 노드의 child 노드를 탐색하여 어떻게든 하나의 노드를 삽입하고 함수를 종료

		'''
		queue = []
		node = self
		queue.append(node)
		while len(queue) > 0:
			node = queue.pop(0)
			if node.expand and node.n_ary == 2:
				if node.left is None:
					node.left = Node(transformation, router, routers_q, dense, decoder, n_ary=node.n_ary)
					node.left.parent = node
					return
				elif node.right is None:
					node.right = Node(transformation, router, routers_q, dense, decoder, n_ary=node.n_ary)
					node.right.parent = node
					return
				else:
					queue.append(node.left)
					queue.append(node.right)
			elif node.expand and node.n_ary >= 3:
				for idx in range(len(node.children)):
					if node.children[idx] is None:
						node.children[idx] = Node(transformation, router, routers_q, dense, decoder, n_ary=node.n_ary)
						node.children[idx].parent = node
						return
				for idx in range(len(node.children)):
					queue.append(node.children[idx])

		print('\nAttention node has not been inserted!\n')
		return

	def prune_child(self, child):
		if self.n_ary == 2:
			if child is self.left:
				self.left = None
				self.router = None
				self.routers_q = None
				child.parent = None
			elif child is self.right:
				self.right = None
				self.router = None
				self.routers_q = None
				child.parent = None
			else:
				raise ValueError("This is not my child! (Node is not a child of this parent.)")
		elif self.n_ary >= 3:
			for idx, node_child in enumerate(self.children):
				if child is node_child: # prune하려는 child node를 찾았다면
					self.children[idx] = None
					child.parent = None

					# 활성화 되어있는 child의 indices 받아오기
					active_child_indices = self.active_child_indices()
					if len(active_child_indices) <= 1:
						# Pruning했을 때, child node가 하나 남아있다면, 더 이상 child 쪽 branch가 뻗어있어야 할 
						# 이유가 없기 때문에 router 자체를 제거해버린다.
						self.router = None
						self.routers_q = None
					else:
						import types
						import torch

						def mask_pruned_child_probabilities(router):
							if router is None:
								return
							if not hasattr(router, "_treevae_base_forward"):
								router._treevae_base_forward = router.forward

							router._treevae_active_child_indices = active_child_indices

							def masked_forward(router_self, inputs, *args, **kwargs):
								# 기존 router의 forward method를 사용한다. 
								output = router_self._treevae_base_forward(inputs, *args, **kwargs)
								if isinstance(output, tuple):
									probabilities, *extra_outputs = output
								else:
									probabilities, extra_outputs = output, None

								# Probability normalization
								active_indices = router_self._treevae_active_child_indices
								masked_probabilities = torch.zeros_like(probabilities)
								active_probabilities = probabilities[:, active_indices]
								normalizer = active_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-7)
								masked_probabilities[:, active_indices] = active_probabilities / normalizer

								if extra_outputs is not None:
									return (masked_probabilities, *extra_outputs)
								return masked_probabilities

							router.forward = types.MethodType(masked_forward, router)

						mask_pruned_child_probabilities(self.router)
						mask_pruned_child_probabilities(self.routers_q)
					return
			raise ValueError("This is not my child! (Node is not a child of this parent.)")

def return_list_tree(root):
	list_nodes = [root]
	denses = []
	transformations = []
	routers = []
	routers_q = []
	decoders = []
	while len(list_nodes) != 0:
		current_node = list_nodes.pop(0)
		denses.append(current_node.dense)
		transformations.append(current_node.transformation)
		routers.append(current_node.router)
		routers_q.append(current_node.routers_q)
		decoders.append(current_node.decoder)
		if current_node.router is not None:
			list_nodes.extend(current_node.active_children())
		elif current_node.router is None and current_node.decoder is None:
			# We are in an internal node with pruned leaves and thus only have one child
			list_nodes.append(current_node.single_child())
	return nn.ModuleList(transformations), nn.ModuleList(routers), nn.ModuleList(denses), nn.ModuleList(decoders), nn.ModuleList(routers_q)


def construct_tree_fromnpy(model, data_tree, configs):
	from models.model_smalltree import SmallTreeVAE
	nodes = {0: {'node': model.tree, 'depth': 0}}

	for i in range(1, len(data_tree)-1):
		node_left = data_tree[i]
		node_right = data_tree[i + 1]
		id_node_left = node_left[0]
		id_node_right = node_right[0]

		if node_left[2] == node_right[2]:
			id_parent = node_left[2]

			parent = nodes[id_parent]
			node = parent['node']
			depth = parent['depth']

			new_depth = depth + 1

			small_model = SmallTreeVAE(new_depth+1, **configs['training'])

			node.router = small_model.decision
			node.routers_q = small_model.decision_q

			node.decoder = None
			n = []
			for j in range(2):
				dense = small_model.denses[j]
				transformation = small_model.transformations[j]
				decoder = small_model.decoders[j]
				n.append(Node(transformation, None, None, dense, decoder))

			node.left = n[0]
			node.right = n[1]

			nodes[id_node_left] = {'node': node.left, 'depth': new_depth}
			nodes[id_node_right] = {'node': node.right, 'depth': new_depth}
		elif data_tree[i][2] != data_tree[i - 1][2]: # Internal node w/ 1 child only
			id_parent = node_left[2]

			parent = nodes[id_parent]
			node = parent['node']
			depth = parent['depth']

			new_depth = depth + 1

			small_model = SmallTreeVAE(new_depth+1, **configs['training'])

			node.router = None
			node.routers_q = None

			node.decoder = None
			n = []
			for j in range(1):
				dense = small_model.denses[j]
				transformation = small_model.transformations[j]
				decoder = small_model.decoders[j]
				n.append(Node(transformation, None, None, dense, decoder))

			node.left = n[0]
			nodes[id_node_left] = {'node': node.left, 'depth': new_depth}

	transformations, routers, denses, decoders, routers_q = return_list_tree(model.tree)
	model.decisions_q = routers_q
	model.transformations = transformations
	model.decisions = routers
	model.denses = denses
	model.decoders = decoders
	model.depth = model.compute_depth()
	return model


def construct_data_tree(model, y_predicted, y_true, n_leaves, data_name):
	list_nodes = [{'node':model.tree, 'id': 0, 'parent':None}]
	data = []
	i = 0
	labels = [i for i in range(n_leaves)]
	while len(list_nodes) != 0:
		current_node = list_nodes.pop(0)
		if current_node['node'].router is not None:
			data.append([current_node['id'], str(current_node['id']), current_node['parent'], 10])
			node_left, node_right = current_node['node'].left, current_node['node'].right
			i += 1
			list_nodes.append({'node':node_left, 'id': i, 'parent': current_node['id']})
			i += 1
			list_nodes.append({'node':node_right, 'id': i, 'parent': current_node['id']})
		elif current_node['node'].router is None and current_node['node'].decoder is None:
			# We are in an internal node with pruned leaves and will only add the non-pruned leaves
			data.append([current_node['id'], str(current_node['id']), current_node['parent'], 10])
			node_left, node_right = current_node['node'].left, current_node['node'].right
			child = node_left if node_left is not None else node_right
			i += 1
			list_nodes.append({'node': child, 'id': i, 'parent': current_node['id']})
		else:
			y_leaf = labels.pop(0)
			ind = np.where(y_predicted == y_leaf)[0]
			digits, counts = np.unique(y_true[ind], return_counts=True)
			tot = len(ind)
			if tot == 0:
				name = 'no digits'
			else:
				counts = np.round(counts / np.sum(counts), 2)
				ind = np.where(counts > 0.1)[0]
				name = ' '
				for j in ind:
					if data_name == 'fmnist':
						items = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
								 'Bag', 'Boot']
						name = name + str(items[digits[j]]) + ': ' + str(counts[j]) + ' '
					elif data_name == 'cifar10':
						items = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
								 'truck']
						name = name + str(items[digits[j]]) + ': ' + str(counts[j]) + ' '
					elif data_name == 'news20':
						items = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
								 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale','rec.autos',
								 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
								 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
								 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
								 'talk.religion.misc']
						name = name + str(items[digits[j]]) + ': ' + str(counts[j]) + ' '
					elif data_name == 'omniglot':
						from utils.data_utils import get_selected_omniglot_alphabets
						items = get_selected_omniglot_alphabets()
						if np.unique(y_true).shape[0]>len(items):
							items=np.arange(50)
						
						name = name + items[digits[j]] + ': ' + str(counts[j]) + ' '
					else:
						name = name + str(digits[j]) + ': ' + str(counts[j]) + ' '
				name = name + 'tot ' + str(tot)
			data.append([current_node['id'], name, current_node['parent'], 1])
	return data
