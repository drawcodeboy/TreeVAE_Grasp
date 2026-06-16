import numpy as np
import torch
import torch.distributions as td
from matplotlib import pyplot as plt
from utils.model_utils import construct_tree, compute_posterior
import re
import networkx as nx
from sklearn.decomposition import PCA



def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''
        Encodes the hierarchy for the tree layout in a graph.
        From https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3 
        If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})


def plot_tree_graph(data):

    # get a '/n' before every 'tot' in each second entry of data
    data = data.copy()
    for d in data:
        if d[3] == 1:
            #d[1] = d[1].replace('tot', '\ntot')
            pattern = r'(\w+:\s\d+\.\d+|\d+:\s\d+\.\d+|\w+\s\d+|\d+\s\d+|\w+:\s\d+|\d+:\s\d+|\w+:\s\d+\s\w+|\d+:\s\d+\s\w+|\w+\s\d+\s\w+|\d+\s\d+\s\w+|\w+:\s\d+\.\d+\s\w+|\d+:\s\d+\.\d+\s\w+)'

            # Split the string using the regular expression pattern
            result = re.findall(pattern, d[1])

            # Join the resulting list to format it as desired
            d[1] = '\n'.join(result)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in data:
        node_id, label, parent_id, node_type = node
        G.add_node(node_id, label=label, node_type=node_type)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

    # Get positions of graph nodes
    pos = hierarchy_pos(G, 0, levels=None, width=1, height=1)

    # get the labels of the nodes
    labels = nx.get_node_attributes(G, 'label')

    # Initialize node color and size lists
    node_colors = []
    node_sizes = []

    # Iterate through nodes to set colors and sizes
    for node_id, node_data in G.nodes(data=True):
        if G.out_degree(node_id) == 0:  # Leaf nodes have out-degree 0
            node_colors.append('lightgreen')  
            node_sizes.append(4000) 

        else:
            node_colors.append('lightblue')  
            node_sizes.append(1000) 

    # Draw the graph with different node properties
    tree = plt.figure(figsize=(10, 5))
    nx.draw(G, pos=pos, labels=labels, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=7)

    plt.show()



def get_node_embeddings(model, x):
    assert model.training == False
    epsilon = 1e-7
    device = x.device

    # compute deterministic bottom up
    d = x
    encoders = []

    for i in range(0, len(model.hidden_layers)):
        d, _, _ = model.bottom_up[i](d)
        # store the bottom-up layers for the top-down computation
        encoders.append(d)

    # Create a list to store node information
    node_info_list = []

    # Create a list of nodes of the tree that need to be processed
    list_nodes = [{'node': model.tree, 'depth': 0, 'prob': torch.ones(x.size(0), device=device), 'z_parent_sample': None}]

    while len(list_nodes) != 0:
        # Store info regarding the current node
        current_node = list_nodes.pop(0)
        node, depth_level, prob = current_node['node'], current_node['depth'], current_node['prob']
        z_parent_sample = current_node['z_parent_sample']

        # Access deterministic bottom-up mu and sigma hat (computed above)
        d = encoders[-(1 + depth_level)]
        z_mu_q_hat, z_sigma_q_hat = node.dense(d)

        if depth_level == 0:
            z_mu_q, z_sigma_q = z_mu_q_hat, z_sigma_q_hat
        else:
            # The generative mu and sigma are the output of the top-down network given the sampled parent
            _, z_mu_p, z_sigma_p = node.transformation(z_parent_sample)
            z_mu_q, z_sigma_q = compute_posterior(z_mu_q_hat, z_mu_p, z_sigma_q_hat, z_sigma_p)

        # Compute sample z using mu_q and sigma_q
        z = td.Independent(td.Normal(z_mu_q, torch.sqrt(z_sigma_q + epsilon)), 1)
        z_sample = z.rsample()

        # Store information in the list
        node_info = {'prob': prob, 'z_sample': z_sample}
        node_info_list.append(node_info)

        if node.router is not None:
            if model.n_ary == 2:
                prob_child_left_q = node.routers_q(d).squeeze()

                # We are not in a leaf, so we have to add the left and right child to the list
                prob_node_left, prob_node_right = prob * prob_child_left_q, prob * (1 - prob_child_left_q)

                node_left, node_right = node.left, node.right
                list_nodes.append(
                    {'node': node_left, 'depth': depth_level + 1, 'prob': prob_node_left, 'z_parent_sample': z_sample})
                list_nodes.append({'node': node_right, 'depth': depth_level + 1, 'prob': prob_node_right,
                                'z_parent_sample': z_sample})
            else:
                prob_child_q = node.routers_q(d)
                active_indices = node.active_child_indices()
                children = node.active_children()

                prob_child_q = prob_child_q[:, active_indices]
                prob_child_q = prob_child_q / (prob_child_q.sum(dim=1, keepdim=True) + epsilon)

                prob_nodes = prob.unsqueeze(1) * prob_child_q
                for idx, child in enumerate(children):
                    list_nodes.append({
                        'node': child, 'depth': depth_level + 1, 'prob': prob_nodes[:, idx], 'z_parent_sample': z_sample
                    })

        elif node.decoder is None and node.has_children():
            # We are in an internal node with pruned leaves and thus only have one child
            child = node.single_child()
            list_nodes.append(
                {'node': child, 'depth': depth_level + 1, 'prob': prob, 'z_parent_sample': z_sample})

    return node_info_list



# Create a function to draw scatter plots as nodes
def draw_scatter_node(node_id, node_embeddings, colors, ax, pca = True):

    # if list is empty --> node has been pruned
    if node_embeddings[node_id]['z_sample'] == []:
        # return empty plot
        ax.set_title(f"Node {node_id}")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    z_sample = node_embeddings[node_id]['z_sample']
    weights = node_embeddings[node_id]['prob']

    if pca:
        pca_fit = PCA(n_components=2)
        z_sample = pca_fit.fit_transform(z_sample)


    ax.scatter(z_sample[:, 0], z_sample[:, 1], c=colors, cmap='tab10', alpha=weights, s = 0.25)
    ax.set_title(f"Node {node_id}")
    # no ticks
    ax.set_xticks([])
    ax.set_yticks([])


def get_depth(node_id, data):
    # Initialize the depth to 0
    depth = 0
    
    # Find the node in the data list
    node = next(node for node in data if node[0] == node_id)
    
    # Recursively calculate the depth
    if node[2] is not None:
        depth = 1 + get_depth(node[2], data)
    
    return depth


def get_scatter_axes_position(node_id, pos, node_width=0.1, node_height=0.1):
    x, y = pos[node_id]
    x = np.clip(x - node_width / 2, 0, 1 - node_width)
    y = np.clip(y + 0.9, 0, 1 - node_height)
    return x, y, node_width, node_height


# Create the tree graph with scatter plots as nodes
def draw_tree_with_scatter_plots(data, node_embeddings, label_list, pca = True):

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in data:
        node_id, label, parent_id, node_type = node
        G.add_node(node_id, label=label, node_type=node_type)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

    # Get positions of graph nodes
    pos = hierarchy_pos(G, 0, levels=None, width=1, height=1)

    # get the labels of the nodes
    labels = nx.get_node_attributes(G, 'label')


    fig, ax = plt.subplots(figsize=(20, 10))
    node_positions = {}

    for node_id, node_data in G.nodes(data=True):
        # Create a subplot for each node, centered on the node
        x, y, width, height = get_scatter_axes_position(node_id, pos)
        sub_ax = fig.add_axes([x, y, width, height])
        draw_scatter_node(node_id, node_embeddings, label_list, sub_ax, pca)
        node_positions[node_id] = (x + width / 2, y + height / 2)

    # Draw the lines between above nodes.
    for node in data:
        node_id, label, parent_id, node_type = node

        # draw the connection lines
        if parent_id is not None:
            x, y = node_positions[node_id]
            x_parent, y_parent = node_positions[parent_id]
            ax.plot([x_parent, x], [y_parent, y], color='black', alpha=0.5)


    # Set the limits of the plot
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.axis('off')

    plt.show()

