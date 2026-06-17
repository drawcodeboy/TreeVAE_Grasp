import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
from einops import rearrange

#-----------------[PointNet]-----------------

def actvn(x):
    return F.leaky_relu(x, negative_slope=0.3)


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    '''
        Attributes
        ----------
        encoded_size : int
            point cloud의 latent vector dimension size
        num_points : int 
            point cloud 점 개수, encoder만 쓸 거면 필요 없는 attribute인데, decoder가 몇 개의 점으로 복원해야 할 지 모르기 때문
        channel : int
            point cloud channel, (x, y, z) default로 가져감
        input_transform : bool
            input pc의 T-Net 사용 유무
        feature_transform : bool
            feature pc의 T-Net 사용 유무
    '''
    def __init__(self, encoded_size, num_points, channel=3, input_transform=True, feature_transform=True):
        super(PointNetEncoder, self).__init__()
        self.num_points = num_points
        self.channel = channel
        self.stn = STN3d(channel) if input_transform else nn.Identity()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, encoded_size)
        self.bn5 = nn.BatchNorm1d(encoded_size)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() == 2:
            x = x.view(batch_size, self.num_points, self.channel).transpose(1, 2)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if self.channel > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if self.channel > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = actvn(self.bn5(self.fc2(x)))
        return x, trans, trans_feat

    '''
    def feature_transform_regularizer(self, trans):
        d = trans.size(1)
        I = torch.eye(d, device=trans.device).unsqueeze(0)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss
    '''

class PointNetDecoder(nn.Module):
    def __init__(self, input_shape, num_points):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(input_shape, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024, bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        # Model capacity 너무 약하지 않을까 point cloud의 공간적인 특성이 전혀 보완되지 않겠는데 
        self.fc3 = nn.Linear(1024, self.num_points * 3, bias=True)

    def forward(self, inputs):
        x = actvn(self.bn1(self.fc1(inputs)))
        x = actvn(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = rearrange(x, 'B (c N) -> B c N', c=3)
        return x
    
#-----------------[FoldingNet]-----------------
# Refer this repository: https://github.com/qinglew/FoldingNet

def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.

    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods
    
    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx

class GraphLayer(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """
    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        # KNN
        knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0].permute(0, 2, 1)  # (B, N, C)
        
        # Feature Map
        x = F.relu(self.bn(self.conv(x)))
        return x

class FoldingNetEncoder(nn.Module):
    """
    Graph based encoder.
    """
    def __init__(self, encoded_size=512):
        super(FoldingNetEncoder, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, encoded_size, 1)
        self.conv5 = nn.Conv1d(encoded_size, encoded_size, 1)
        self.bn4 = nn.BatchNorm1d(encoded_size)
        self.bn5 = nn.BatchNorm1d(encoded_size)

    def forward(self, x):
        b, c, n = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn(x, k=16)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        x = torch.cat([x, covariances], dim=1)  # (B, 12, N)

        # three layer MLP (12xN -> 64XN)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # two consecutive graph layers (64XN -> 1024xN)
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        # 왜 아래 두 라인은 논문에서 보는 거랑 달리 순서가 바뀌어있지
        # x = self.bn4(self.conv4(x))
        # x = torch.max(x, dim=-1)[0]

        # 논문에 맞게 수정
        # global max-pooling (1024XN -> 1024)
        x = torch.max(x, dim=-1, keepdim=True)[0] # (B, 1024, 1)
        
        # 2 layer perceptron
        x = self.bn4(self.conv4(x)) # (B, encoded_size, 1)
        x = self.bn5(self.conv5(x)) # (B, encoded_size, 1)
        x = x.squeeze(-1) # (B, encoded_size)
        return x, None, None # The reason why return the 'None' is PointNet implementation.
    
class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x

class FoldingNetDecoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, in_channel=512, sqrt_m=45):
        super(FoldingNetDecoder, self).__init__()

        # Sample the grids in 2D space
        xx = np.linspace(-0.3, 0.3, sqrt_m, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, sqrt_m, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)   # (2, sqrt_m, sqrt_m)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, sqrt_m, sqrt_m) -> (2, sqrt_m * sqrt_m)
        
        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(in_channel + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(in_channel + 3, [512, 512, 3])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, sqrt_m * sqrt_m)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, sqrt_m * sqrt_m)
        
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, sqrt_m * sqrt_m)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        
        return recon2

def get_encoder_pc(architecture, encoded_size, num_points):
    if architecture == 'pointnet':
        encoder = PointNetEncoder(encoded_size, num_points, channel=3, input_transform=True, feature_transform=True)
    elif architecture == 'foldingnet':
        encoder = FoldingNetEncoder(encoded_size)
    else:
        raise ValueError('The encoder architecture is mispecified.')
    return encoder

def get_decoder_pc(architecture, input_shape, num_points=45*45):
    if architecture == 'pointnet':
        # PointNet decoder의 num_points를 안 받는 이유는 혹여나 FoldingNet과 비교 필요시
        # fair comparison을 위해 개수를 맞춤
        decoder = PointNetDecoder(input_shape, num_points)
    elif architecture == 'foldingnet':
        decoder = FoldingNetDecoder(in_channel=input_shape, sqrt_m=45)
    else:
        raise ValueError('The decoder architecture is mispecified.')
    return decoder