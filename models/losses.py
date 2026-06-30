"""
Loss functions for the reconstruction term of the ELBO.
"""
import torch
import torch.nn.functional as F

def loss_reconstruction_binary(x, x_decoded_mean, weights):
    x = torch.flatten(x, start_dim=1)
    x_decoded_mean = [torch.flatten(decoded_leaf, start_dim=1) for decoded_leaf in x_decoded_mean]
    loss = torch.sum(
        torch.stack([weights[i] *
                        F.binary_cross_entropy(input = x_decoded_mean[i], target = x, reduction='none').sum(dim=-1)
                        for i in range(len(x_decoded_mean))], dim=-1), dim=-1)
    return loss

def loss_reconstruction_mse(x, x_decoded_mean, weights):
    x = torch.flatten(x, start_dim=1)
    x_decoded_mean = [torch.flatten(decoded_leaf, start_dim=1) for decoded_leaf in x_decoded_mean]
    loss = torch.sum(
        torch.stack([weights[i] *
                        F.mse_loss(input = x_decoded_mean[i], target = x, reduction='none').sum(dim=-1)
                        for i in range(len(x_decoded_mean))], dim=-1), dim=-1)
    return loss

def loss_reconstruction_mae(x, x_decoded_mean, weights):
    x = torch.flatten(x, start_dim=1)
    x_decoded_mean = [torch.flatten(decoded_leaf, start_dim=1) for decoded_leaf in x_decoded_mean]
    # recon loss가 너무 높아서 임시 방편 -> sum to mean
    loss = torch.sum(
        torch.stack([weights[i] * F.l1_loss(input = x_decoded_mean[i], target = x, reduction='none').mean(dim=-1)
                        for i in range(len(x_decoded_mean))], dim=-1), dim=-1)
    return loss

def loss_reconstruction_cov_mse_eval(x, x_decoded_mean, weights):
    # NOTE Only use for evaluation purposes, as the clamping stops gradient flow
    # NOTE WE ASSUME IDENTITY MATRIX BECAUSE WE ASSUME THIS IMPLICITLY WHEN ONLY OPTIMIZING MSE
    scale = torch.diag(torch.ones_like(x_decoded_mean[0])) 
    logpx = torch.zeros_like(weights[0])
    for i in range(len(x_decoded_mean)):
        x_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.clamp(x_decoded_mean[i],0,1), covariance_matrix=scale)   
        logpx = logpx + weights[i] * x_dist.log_prob(x)
    return logpx

"""
Point cloud reconstruction losses for TreeVAE.
"""

def _to_pointcloud(x):
    if x.dim() == 2:
        batch_size = x.size(0)
        if x.size(1) % 3 != 0:
            raise ValueError('Point cloud tensor second dimension must be divisible by 3.')
        return x.view(batch_size, -1, 3)
    elif x.dim() == 3:
        if x.size(1) == 3:
            return x.transpose(1, 2)
        elif x.size(2) == 3:
            return x
    raise ValueError('Unsupported point cloud tensor shape %s.' % (tuple(x.shape),))


def _chamfer_distance(x, y):
    x = _to_pointcloud(x)
    y = _to_pointcloud(y)

    # x: [B, N, 3], y: [B, M, 3]
    x_expanded = x.unsqueeze(2)  # [B, N, 1, 3]
    y_expanded = y.unsqueeze(1)  # [B, 1, M, 3]
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)

    x_to_y = torch.min(dist, dim=2)[0]
    y_to_x = torch.min(dist, dim=1)[0]

    # mean over points to normalize by cloud size
    return x_to_y.mean(dim=1) + y_to_x.mean(dim=1)


def loss_reconstruction_chamfer(x, x_decoded_mean, weights):
    """
    Chamfer distance reconstruction loss for point clouds.

    Parameters
    ----------
    x : Tensor
        Ground-truth point cloud, shape [B, N, 3] or [B, 3, N] or flattened [B, N*3].
    x_decoded_mean : list of Tensor
        List of reconstructed point clouds, each with the same shape convention as x.
    weights : list of Tensor
        Per-leaf weights of shape [B] or [B, 1].

    Returns
    -------
    Tensor
        Per-sample loss of shape [B].
    """
    losses = []
    for i in range(len(x_decoded_mean)):
        decoded = x_decoded_mean[i]
        cd = _chamfer_distance(x, decoded)
        w = weights[i]
        if w.dim() == 2 and w.size(1) == 1:
            w = w.squeeze(-1)
        losses.append(w * cd)

    loss = torch.stack(losses, dim=-1).sum(dim=-1)
    return loss
