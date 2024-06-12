import torch
import torch.nn as nn
from utils.registry import ARCH_REGISTRY
import copy

class LinerBNAct(nn.Sequential):
    def __init__(self, dim_i, dim_o):
        super(LinerBNAct, self).__init__(
            nn.Linear(dim_i, dim_o),
            nn.BatchNorm1d(dim_o),
            nn.GELU()
        )


class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, padding=None, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.LeakyReLU(0.1, True)
        )


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel // 2)

        self.conv2 = nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel // 2, kernel_size=3, bias=False,
                               groups=in_channel // 2, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channel // 2)

        self.conv3 = nn.Conv2d(in_channels=in_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class MyBaseEncoder(nn.Module):
    def __init__(self):
        super(MyBaseEncoder, self).__init__()
        self.stage1 = nn.Sequential(
            ConvAct(3, 16, 3),
            ConvAct(16, 16, 3),
            ConvAct(16, 32, 3, 2),
        )

        self.stage2 = nn.Sequential(
            ConvAct(32, 32, 3),
            ConvAct(32, 64, 3, 2),
        )

        self.stage3 = nn.Sequential(
            ConvAct(64, 128, 3),
            ConvAct(128, 128, 3, 2),
        )

        self.stage4 = nn.Sequential(
            ConvAct(128, 256, 3),
            ConvAct(256, 256, 3, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp1 = LinerBNAct(256, 128)
        self.mlp2 = LinerBNAct(128, 128)

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1000)
        )

    def forward(self, x):
        fea1 = self.stage1(x)
        fea2 = self.stage2(fea1)
        fea3 = self.stage3(fea2)
        fea4 = self.stage4(fea3)
        fea4 = self.pool(fea4).squeeze(-1).squeeze(-1)
        out1 = self.mlp1(fea4)
        out2 = self.mlp2(out1)
        predict = self.classifier(torch.flatten(fea4, 1))
        return [out1, out2], [fea1, fea2, fea3], predict


@ARCH_REGISTRY.register()
class ConStyle_v2(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    # base_encoder: resnet-50来自于： import torchvision.models as models
    def __init__(self, Train=True, dim=128, K=65760, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(ConStyle_v2, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.Train = Train
        self.encoder_q = MyBaseEncoder()
        
        if self.Train:
            self.encoder_k = MyBaseEncoder()

            for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(dim, K))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, q):
        # gather keys before updating queue

        batch_size = q.shape[0]

        ptr1 = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        q1 = copy.deepcopy(self.queue[:, ptr1: ptr1 + batch_size].detach())
        ptr2 = (ptr1 + batch_size) % self.K
        q2 = copy.deepcopy(self.queue[:, ptr2: ptr2 + batch_size].detach())
        self.queue[:, ptr1:ptr1 + batch_size] = q.T
        ptr1 = (ptr1 + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr1
        return [q2.T, q1.T]

    def forward(self, im_q, im_k=None):
        if self.Train:
            return self.train_forward(im_q, im_k)
        else:
            return self.encoder_q(im_q)

    def train_forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, feas, predict_q = self.encoder_q(im_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
            k, _, predict_k = self.encoder_k(im_k)

        l_pos = torch.einsum("nc,nc->n", [q[1], k[1]]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q[1], self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        q_from_queue = self._dequeue_and_enqueue(q[1])
        return q, k[1], q_from_queue, logits, labels, feas, [predict_q, predict_k]





