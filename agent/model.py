import numpy as np
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ChannelMaxPool(nn.Module):
    def forward(self, x):
        (
            N,
            C,
        ) = x.shape[:2]
        return torch.max(x.reshape(N, C, -1), 2)[0]


class ConcatCoords(nn.Module):
    # assume X is of shape N, C, H, W
    # TODO: generalize to arbitrary dimensions
    def forward(self, x):
        N, C, H, W = x.shape
        coords = torch.empty(N, 2, H, W).double().to(x.device)
        # x coordinate
        x_coord = (
            2 * (torch.arange(H, dtype=torch.double).reshape(-1, 1).repeat(1, W) / W)
            - 1
        )
        # y coordinate
        y_coord = (
            2 * (torch.arange(W, dtype=torch.double).reshape(1, -1).repeat(H, 1) / H)
            - 1
        )

        coords[:, 0, :, :] = x_coord
        coords[:, 1, :, :] = y_coord
        x = torch.cat([x, coords], 1)
        return x


class MHDPA(nn.Module):
    def __init__(self, entity_dim, qkv_dim, n_heads):
        super().__init__()
        self.entity_dim = entity_dim
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads
        self.d = self.qkv_dim // n_heads
        assert (
            self.d * n_heads == self.qkv_dim
        ), "Number of heads must evenly divide QKV dimension"

        self.Wq = nn.Linear(entity_dim, qkv_dim)
        self.Wv = nn.Linear(entity_dim, qkv_dim)
        self.Wk = nn.Linear(entity_dim, qkv_dim)

        self.attention_weights = None

    def forward(self, q, k, v):
        N = q.size(0)

        # linear projections, layer normalizations, reshaping to heads x d
        q = (
            nn.functional.layer_norm(self.Wq(q), [self.qkv_dim])
            .view(N, -1, self.n_heads, self.d)
            .transpose(1, 2)
        )
        k = (
            nn.functional.layer_norm(self.Wk(k), [self.qkv_dim])
            .view(N, -1, self.n_heads, self.d)
            .transpose(1, 2)
        )
        v = (
            nn.functional.layer_norm(self.Wv(v), [self.qkv_dim])
            .view(N, -1, self.n_heads, self.d)
            .transpose(1, 2)
        )
        interactions, self.attention_weights = self.attention(q, k, v)
        x = interactions.transpose(1, 2).contiguous().view(N, -1, self.n_heads * self.d)
        return x

    def attention(self, q, k, v):
        d = q.size(-1)
        saliencies = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d)
        weights = torch.nn.functional.softmax(saliencies, dim=-1)
        return torch.matmul(weights, v), weights


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        return x


class RelationalBlock(nn.Module):
    def __init__(self, n_entities, entity_dim, qkv_dim, n_heads):
        super().__init__()
        self.mhdpa = MHDPA(entity_dim, qkv_dim, n_heads)
        self.mlps = nn.ModuleList(
            [MLP(entity_dim, entity_dim) for i in range(n_entities)]
        )
        self.n_entities = n_entities

    def forward(self, x):
        # x: N x C x H * W
        # -> x: H * W x N x C
        y = x.permute(2, 0, 1)
        y = self.mhdpa(y, y, y)
        # -> x: N, H * W, C
        y = y.permute(1, 0, 2)
        splits = torch.split(y, 1, dim=1)
        outs = []
        for i, split in enumerate(splits):
            outs.append(self.mlps[i](split))
        y = torch.cat(outs, dim=1)
        y = y.transpose(1, 2)
        y = y + x
        y = nn.functional.layer_norm(y, [y.size(-1)])
        return y


class ResidualBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(*[BasicBlock(26, 26) for _ in range(n)])

    def forward(self, x):
        return self.net(x)


class RelationalPolicy(nn.Module):
    def __init__(self, n_relational, channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 6, (2, 2), 1),
            nn.ReLU(),
            nn.Conv2d(6, 24, (2, 2), 1),
            nn.ReLU(),
            ConcatCoords(),
        )
        self.rb = nn.Sequential(
            *[RelationalBlock(36, 26, 26, 2) for _ in range(n_relational)]
        )
        self.linear = nn.Sequential(
            ChannelMaxPool(),
            nn.Linear(26, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.conv(x)
        N, C = x.shape[:2]
        x = x.reshape(N, C, -1)
        x = self.rb(x)
        x = self.linear(x)
        return x


class BaselinePolicy(nn.Module):
    def __init__(self, n_residual):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, (2, 2), 1),
            nn.ReLU(),
            nn.Conv2d(6, 24, (2, 2), 1),
            nn.ReLU(),
            ConcatCoords(),
            ResidualBlock(n_residual),
            ChannelMaxPool(),
            nn.Linear(26, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.net(x)