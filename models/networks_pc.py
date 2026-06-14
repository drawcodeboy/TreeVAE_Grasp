import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys


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
        return x
    
def get_encoder_pc(architecture, encoded_size, num_points):
    if architecture == 'pointnet':
        encoder = PointNetEncoder(encoded_size, num_points, channel=3, input_transform=True, feature_transform=True)
    else:
        raise ValueError('The encoder architecture is mispecified.')
    return encoder

def get_decoder_pc(architecture, input_shape, num_points):
    if architecture == 'pointnet':
        decoder = PointNetDecoder(input_shape, num_points)
    else:
        raise ValueError('The decoder architecture is mispecified.')
    return decoder