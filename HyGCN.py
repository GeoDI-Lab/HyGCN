import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv
import math
import numpy as np



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HyperGCNconv(nn.Module):
    def __init__(self, in_c, out_c):
        super(HyperGCNconv, self).__init__()
        self.hgcn = HypergraphConv(in_c, out_c)

    def forward(self, x, HE, HEW):
        # X - [B*N, F], hyperedge_index- [B*N, M]
        y = self.hgcn(x=x, hyperedge_index=HE, hyperedge_weight=HEW)
        return y


class GCNconv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GCNconv, self).__init__()
        self.gcn = GCNConv(in_c, out_c)

    def forward(self, x, HE, HEW):
        # X - [B*N, F], hyperedge_index- [B*N, M]
        y = self.gcn(x=x, edge_index=HE, edge_weight=HEW)
        return y


class HGCN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(HGCN, self).__init__()
        self.hgcn = HyperGCNconv(in_feature, out_feature)

    def forward(self, x, HE, HEWI):
        batch_size, num_vertices, features, time_stamps = x.shape
        gx = []
        for t in range(time_stamps):
            y = x[:, :, :, t]
            y = y.squeeze(-1).reshape(batch_size * num_vertices, features)
            out = self.hgcn(y, HE, HEWI)
            gx.append(out.unsqueeze(0))
        gx = torch.cat(gx, dim=0).reshape(time_stamps, batch_size, num_vertices, -1).permute(1, 2, 3, 0)
        gx = F.relu(gx)
        return gx


class GCN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(GCN, self).__init__()
        self.hgcn = GCNconv(in_feature, out_feature)

    def forward(self, x, E, EW):
        batch_size, num_vertices, features, time_stamps = x.shape
        gx = []
        for t in range(time_stamps):
            y = x[:, :, :, t]
            y = y.squeeze(-1).reshape(batch_size * num_vertices, features)
            out = self.hgcn(y, E, EW)
            gx.append(out.unsqueeze(0))
        gx = torch.cat(gx, dim=0).reshape(time_stamps, batch_size, num_vertices, -1).permute(1, 2, 3, 0)
        gx = F.relu(gx)
        return gx


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * Residual Connection *
    #        |                                |
    #        |    |--->--- CasualConv2d ----- + -------|
    # -------|----|                                   ⊙ ------>
    #             |--->--- CasualConv2d --- Sigmoid ---|
    #

    # param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, n_vertex, d=1, act_func='gtu'):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = nn.Conv2d(c_in, c_out, 1)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1),
                                            enable_padding=True, dilation=d)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1),
                                            enable_padding=True, dilation=d)
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x_in = self.align(x)
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                # GLU was first purposed in
                # *Language Modeling with Gated Convolutional Networks*.
                # URL: https://arxiv.org/abs/1612.08083
                # Input tensor X is split by a certain dimension into tensor X_a and X_b.
                # In PyTorch, GLU is defined as X_a ⊙ Sigmoid(X_b).
                # URL: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # (x_p + x_in) ⊙ Sigmoid(x_q)
                x = torch.mul((x_p + x_in), self.sigmoid(x_q))

            else:
                # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
                # x = torch.mul(self.tanh(x_p), self.sigmoid(x_q)) + x_in
                gate = self.sigmoid(x_p)
                x = (1 - gate) * x_in + gate * x_q

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == 'leaky_relu':
            x = self.leaky_relu(x_causal_conv + x_in)

        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)

        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')

        return x.permute(0, 3, 1, 2)


class STBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_vertices, kernel=3):
        super(STBlock, self).__init__()
        self.tcl1 = TemporalConvLayer(Kt=kernel, c_in=in_channel, c_out=out_channel, n_vertex=num_vertices)
        self.tcl2 = TemporalConvLayer(Kt=kernel, c_in=2 * out_channel, c_out=out_channel, n_vertex=num_vertices)
        self.gcn = GCN(out_channel, out_channel)
        self.hgcn = HGCN(out_channel, out_channel)
        self.se = SELayer(2 * out_channel)
        self.ln = nn.LayerNorm(out_channel)

    def forward(self, x, HE1, HEW1, HE2, HEW2):
        tx1 = self.tcl1(x)
        sx1 = self.gcn(tx1, HE1, HEW1)
        sx2 = self.hgcn(tx1, HE2, HEW2)
        pre = torch.cat((sx1, sx2), dim=2)
        pre = self.se(pre.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        output = self.tcl2(pre) + x
        return self.ln(F.gelu(output.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)


class GCNADP(nn.Module):
    def __init__(self, DEVICE, num_of_vertices):
        super(GCNADP, self).__init__()
        self.device = DEVICE
        self.nodevec1 = nn.Parameter(torch.randn(num_of_vertices, 40), requires_grad=True).to(DEVICE)
        self.nodevec2 = nn.Parameter(torch.randn(num_of_vertices, 40), requires_grad=True).to(DEVICE)

    def forward(self, x):
        B, N, _, _ = x.shape
        DE = torch.tanh(2 * self.nodevec1)
        EE = torch.tanh(2 * self.nodevec2).transpose(1, 0)
        adj = F.relu(torch.tanh(2 * torch.matmul(DE, EE)))

        mask = torch.zeros(adj.size(0), adj.size(1)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(20, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adp = adj * mask

        adp = adp.repeat(B, 1, 1)
        B, N, _ = adp.shape
        E = []
        for i in range(B):
            edge_index = adp[i, :, :].nonzero(as_tuple=False).t()
            edge_index[0] += i * N
            edge_index[1] += i * N
            E.append(edge_index)
        E = torch.cat(E, dim=1)
        HEW = []
        for i in range(B):
            edge_weight = adp[i, :, :].view(-1)
            HEW.append(edge_weight)
        HEW = torch.cat(HEW, dim=0)
        HEW = HEW[torch.nonzero(HEW)].squeeze(-1)
        return E, HEW


class HGCNADP(nn.Module):
    def __init__(self, DEVICE, num_of_vertices):
        super(HGCNADP, self).__init__()
        self.device = DEVICE
        self.nodevec = nn.Parameter(torch.randn(num_of_vertices, 40), requires_grad=True).to(DEVICE)
        self.edgevec = nn.Parameter(torch.randn(math.ceil(0.4 * num_of_vertices), 40), requires_grad=True).to(DEVICE)

    def forward(self, x):
        B, N, _, _ = x.shape
        DE = torch.tanh(2 * self.nodevec)
        EE = torch.tanh(2 * self.edgevec).transpose(1, 0)
        adj = F.relu(torch.tanh(2 * torch.matmul(DE, EE)))

        mask = torch.zeros(adj.size(0), adj.size(1)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(20, 1)
        mask.scatter_(1, t1, s1.fill_(1))

        adj = adj * mask
        adj = adj.repeat(B, 1, 1)
        HE = []
        HEW = []
        B, N, M = adj.shape
        for i in range(B):
            edge_index = adj[i, :, :].nonzero(as_tuple=False).t()
            edge_index[0] += i * N
            edge_index[1] += i * M
            HE.append(edge_index)
        HE = torch.cat(HE, dim=1)
        for i in range(B):
            edge_weight = adj[i, :, :].view(-1)
            HEW.append(edge_weight)
        HEW = torch.cat(HEW, dim=0)
        HEW = HEW[torch.nonzero(HEW)].squeeze(-1)
        return HE, HEW


class Encoder(nn.Module):
    def __init__(self, DEIVICE, in_channel, out_channel, num_vertices):
        super(Encoder, self).__init__()
        self.start_conv = nn.Conv2d(in_channel, out_channel, 1)
        self.st1 = STBlock(out_channel, out_channel, num_vertices)
        self.st2 = STBlock(out_channel, out_channel, num_vertices)
        self.conv1 = nn.Conv2d(out_channel, out_channel, 1)
        self.adp = GCNADP(DEIVICE, num_vertices)
        self.adph = HGCNADP(DEIVICE, num_vertices)

    def forward(self, x):
        x = self.start_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        HE1, HEW1 = self.adp(x)
        HE2, HEW2 = self.adph(x)
        stx1 = self.st1(x, HE1, HEW1, HE2, HEW2)
        stx2 = self.st2(stx1, HE1, HEW1, HE2, HEW2) + self.conv1(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return stx2


class HyGCN(nn.Module):
    def __init__(self, DEVICE, in_channels, nb_time_filter, num_of_vertices, num_for_predict, len_input=3):
        super(HyGCN, self).__init__()
        self.encoder = Encoder(DEVICE, in_channels, nb_time_filter, num_of_vertices)
        self.output = nn.Sequential(
            nn.Linear(nb_time_filter * len_input, nb_time_filter * num_for_predict),
            nn.ReLU(),
            nn.Linear(nb_time_filter * num_for_predict, 1 * num_for_predict)
        )
        self.p = num_for_predict

        self.to(DEVICE)

    def forward(self, x):
        en_x = self.encoder(x)
        B, N, F, T = en_x.shape
        output = self.output(en_x.reshape(B, N, F * T))
        return output[:, :, :self.p]


def make_model(DEVICE, input_size, hidden_size, num_of_vertices, num_of_timesteps):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    # 所有参数使用xavier uniform初始化
    model = HyGCN(DEVICE, input_size, hidden_size, num_of_vertices, num_of_timesteps)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model


if __name__ == "__main__":
    x = torch.FloatTensor(2, 2085, 1, 3).to('cuda:0')
    model = make_model('cuda:0', 1, 64, 2085, 1).to('cuda:0')
    output = model(x)
    print(output.shape)


