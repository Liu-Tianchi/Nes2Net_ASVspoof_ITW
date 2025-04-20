import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from s3prl import hub
import math
import sys
from .WavLM import *


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, used in ECAPA_TDNN.
    """
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here
        alpha = torch.tanh(self.linear1(x_in)) 
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)

"""
class SSLModel(nn.Module):
    def __init__(self, device, args):
        super(SSLModel, self).__init__()
        self.model = getattr(hub, "wavlm_large")()
        self.device = device
        self.out_dim = 1024
        self.agg = args.agg
        self.n_layer = 25
        if self.agg == 'SEA':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc_att_merge = nn.Sequential(
                nn.Linear(self.n_layer, int(self.n_layer//3), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(self.n_layer//3), self.n_layer, bias=False),
                nn.Sigmoid()
            )
        elif self.agg == 'WeightedSum':
            self.weight_hidd = nn.Parameter(torch.ones(self.n_layer))
        elif self.agg == 'AttM':
            self.n_feat = self.out_dim
            self.W = nn.Parameter(torch.randn(self.n_feat, 1))
            self.W1 = nn.Parameter(torch.randn(self.n_layer, int(self.n_layer//2)))
            self.W2 = nn.Parameter(torch.randn(int(self.n_layer//2), self.n_layer))
            self.hidden = int(self.n_layer*self.n_feat/4)
            self.linear_proj = nn.Linear(self.n_layer*self.n_feat, self.n_feat)
            self.SWISH = nn.SiLU()
        else:
            raise ValueError

    def _weighted_sum(self, x):
        feature = x['hidden_states']
        layer_num = len(feature)
        stacked_feature = torch.stack(feature, dim=0)
        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weight_hidd[:layer_num], dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)
        return weighted_feature

    def _SE_merge(self, x):
        ''' 
            SE Aggregation (SEA)
            A. Guragain, T. Liu, Z. Pan, H. B. Sailor and Q. Wang
            Speech Foundation Model Ensembles for the Controlled Singing Voice Deepfake Detection (CTRSVDD) Challenge 2024
            2024 IEEE Spoken Language Technology Workshop (SLT)
        '''
        feature = x['hidden_states']
        stacked_feature = torch.stack(feature, dim=1)
        b, c, _, _ = stacked_feature.size()
        y = self.avg_pool(stacked_feature).view(b, c)
        y = self.fc_att_merge(y).view(b, c, 1, 1)
        stacked_feature = stacked_feature * y.expand_as(stacked_feature)
        weighted_feature = torch.sum(stacked_feature, dim=1)
        return weighted_feature

    def _Att_merge(self, x):
        ''' 
            Attentive Merging (AttM)
            Z. Pan, T. Liu, H. B. Sailor and Q. Wang
            Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection
            2024 INTERSPEECH
        '''
        x = x['hidden_states']
        x = torch.stack(x, dim=1)
        x_input = x
        x = torch.mean(x, dim=2, keepdim=True) 
        x = self.SWISH(torch.matmul(x, self.W)) 
        x = self.SWISH(torch.matmul(x.view(-1, self.n_layer), self.W1))
        x = torch.sigmoid((torch.matmul(x, self.W2))) 
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x, x_input) 
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2), -1)
        weighted_feature = self.linear_proj(x)
        return weighted_feature

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        emb = self.model(input_tmp)
        if self.agg == 'SEA':
            return self._SE_merge(emb)
        elif self.agg == 'WeightedSum':
            return self._weighted_sum(emb)
        elif self.agg == 'AttM':
            return self._Att_merge(emb)
        else:
            raise ValueError
"""

class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = 'pretrained_models/WavLM-Large.pt'   # Change the pre-trained XLSR model path. 
        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])

    def loadParameters(self, param):
        self_state = self.model.state_dict()
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name
            
            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue
            self_state[name].copy_(param)

    def extract_feat(self, input_data):
        emb, layer_results =  self.model.extract_features(input_data)
        return emb
            
class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        bottleneck = channels // SE_ratio
        if bottleneck < 2:
            bottleneck = 2
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module): 

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        weighted_sum = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(kernel_size,1), dilation=(dilation, 1), padding=(num_pad, 0)))
            bns.append(nn.BatchNorm2d(width))
            initial_value = torch.ones(1, 1, 1, i+2) * (1 / (i+2))
            weighted_sum.append(nn.Parameter(initial_value, requires_grad=True))
        self.weighted_sum = nn.ParameterList(weighted_sum)
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes, SE_ratio)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out).unsqueeze(-1) #bz c T 1
        n_frames = out.shape[2]

        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        for i in range(self.nums):
          sp = torch.cat((sp, spx[i]), -1)

          sp = self.bns[i](self.relu(self.convs[i](sp)))
          sp_s = sp * self.weighted_sum[i]
          sp_s = torch.sum(sp_s, dim=-1, keepdim=False)

          if i==0:
            out = sp_s
          else:
            out = torch.cat((out, sp_s), 1)
        out = torch.cat((out, spx[self.nums].squeeze(-1)),1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class Nested_Res2Net_TDNN(nn.Module):

    def __init__(self, Nes_ratio=[8, 8], input_channel=1024, dilation=2, pool_func='mean', SE_ratio=8):

        super(Nested_Res2Net_TDNN, self).__init__()
        self.Nes_ratio = Nes_ratio[0]
        assert input_channel%self.Nes_ratio == 0
        C = input_channel//self.Nes_ratio
        self.C = C
        Build_in_Res2Nets = []
        bns = []
        for i in range(self.Nes_ratio-1):
            Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio[0]))
            bns.append(nn.BatchNorm1d(C))
        self.Build_in_Res2Nets = nn.ModuleList(Build_in_Res2Nets)
        self.bns = nn.ModuleList(bns)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.pool_func = pool_func
        if pool_func=='mean':
            self.fc = nn.Linear(1024, 2)
        elif pool_func=='ASTP':
            self.pooling = ASTP(in_dim=input_channel, bottleneck_dim=128, global_context_att=False)
            self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        spx = torch.split(x, self.C, 1)
        for i in range(self.Nes_ratio-1):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.Build_in_Res2Nets[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[-1]),1)
        out = self.bn(out)
        out = self.relu(out)
        if self.pool_func == 'mean':
            out = torch.mean(out, dim=-1)
        elif self.pool_func == 'ASTP':
            out = self.pooling(out)
        out = self.fc(out)
        return out

class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        print('args', args)
        self.Nested_Res2Net_TDNN = Nested_Res2Net_TDNN(
            Nes_ratio=args['Nes_ratio'], 
            input_channel=1024, 
            dilation=args['dilation'], 
            pool_func=args['pool_func'],
            SE_ratio=args['SE_ratio']
            )


    def forward(self, x, SSL_freeze=False):

        # Pre-trained WavLM model fine-tuning
        if SSL_freeze:
            with torch.no_grad():
                x_ssl_feat = self.ssl_model.extract_feat(x) # bz T 1024
        else:
            x_ssl_feat = self.ssl_model.extract_feat(x) # bz T 1024
        x_ssl_feat = x_ssl_feat.permute(0,2,1) # bz 1024 T 
        output = self.Nested_Res2Net_TDNN(x_ssl_feat)
     

        return output

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", type=str, default='SEA', choices=['SEA', 'AttM', 'WeightedSum'], help="the aggregation method for SSL")
    parser.add_argument("--dilation", type=int, default=1, help="dilation")
    parser.add_argument("--pool_func", type=str, default='mean', choices=['mean', 'ASTP'], help="pooling function, choose from mean and ASTP")
    parser.add_argument("--SE_ratio", type=int, nargs='+', default=[1], help="SE downsampling ratio in the bottleneck")
    parser.add_argument("--Nes_ratio", type=int, nargs='+', default=[8, 8], help="Nes_ratio, from outer to inner")
    
    args = parser.parse_args()
    model = Model(args=args, device='cuda')
    st = time.time()
    x = torch.rand((4, 32000)).to('cuda')
    model = model.to('cuda')
    for i in range(100):
        y = model(x)
    print('time cost 100 times', time.time()-st)
    print(y)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)
    trainable_params = sum(p.numel() for p in model.ssl_model.parameters() if p.requires_grad)
    print(trainable_params)
    trainable_params = sum(p.numel() for p in model.Nested_Res2Net_TDNN.parameters() if p.requires_grad)
    print(trainable_params)

    from fvcore.nn import FlopCountAnalysis
    x = torch.rand((1,1024,200)).to('cuda')
    flops = FlopCountAnalysis(model.Nested_Res2Net_TDNN, x)
    print(f"FLOPs: {flops.total()} ({flops.total() / 1e9} GFLOPs)")

    from ptflops import get_model_complexity_info
    input_shape = (1024, 200)
    macs, params = get_model_complexity_info(model.Nested_Res2Net_TDNN, input_shape, as_strings=True, print_per_layer_stat=True)
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")