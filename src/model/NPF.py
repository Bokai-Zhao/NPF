import timm
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from .adapter_modules import *
from .ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_


class pEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(pEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1) 


class CNN(nn.Module):
    """
    CNN for extracting spatial prior
    """
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4

class UNIAdapter(nn.Module):
    """
    UNIAdapter
    """
    def __init__(self,uni_freeze=True,conv_inplane=64, n_points=4, deform_num_heads=8,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0,norm_layer=None):
        super().__init__()
        self.uni = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        if uni_freeze:
            self.uni.eval()
            for param in self.uni.parameters():
                param.requires_grad = False

        self.embed_dim = self.uni.embed_dim
        self.interaction_indexes = interaction_indexes

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # self.nor = nn.BatchNorm1d(self.embed_dim)
        self.cnn = CNN(inplanes=conv_inplane, embed_dim=self.embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=self.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=0,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=False)
            for i in range(len(interaction_indexes))
        ])

        self.cnn.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self.uni.load_state_dict(
            torch.load(
                "../weights/UNI/pytorch_model.bin", 
                map_location="cpu"), 
                strict=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        device = x.device
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        c1, c2, c3, c4 = self.cnn(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        x = self.uni.patch_embed(x)
        x = self.uni._pos_embed(x)
        x = self.uni.patch_drop(x)
        x = self.uni.norm_pre(x)

        bs, n, dim = x.shape
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x_without_cls = x[:, 1:, :]
            x_without_cls = layer.injector(query=x_without_cls, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
            x = x + torch.cat([torch.zeros(bs,1,dim).to(device),x_without_cls],dim=1)
            for idx, blk in enumerate(self.uni.blocks[indexes[0]:indexes[-1] + 1]):
                x = blk(x)
            x_without_cls = x[:, 1:, :]
            c = layer.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x_without_cls, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=14, W=14)
            

        c4 = c[:, c2.size(1) + c3.size(1):, :]

        feature = torch.mean(c4, dim=1)
        cls_token = x[:, 0]

        return feature

class NPF(nn.Module):
    def __init__(self, num, indim=1024, N_freqs=6, in_channels=2,logscale=True):
        super(NPF, self).__init__()
        self.embedding_xy = pEmbedding(in_channels, N_freqs, logscale)
        self.xy_mlp_1 = nn.Sequential(
            nn.Linear(2+N_freqs*4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2+N_freqs*4),
        )
        self.xy_mlp_2 = nn.Sequential(
            nn.Linear(2+N_freqs*4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2+N_freqs*4),
        )
        self.xy_mlp_3 = nn.Sequential(
            nn.Linear(2+N_freqs*4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, indim),
        )

        self.predict = nn.Sequential(
            nn.Linear(1024+indim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num)
        )
        self.xy_mlp_1.apply(self._init_weights)
        self.xy_mlp_2.apply(self._init_weights)
        self.xy_mlp_3.apply(self._init_weights)

        self.imagemodel = UNIAdapter(interaction_indexes=[[0,6],[6,12],[12,18],[18,24]])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image,pos):
        pos = self.embedding_xy(pos)
        xy_embedding_1 = self.xy_mlp_1(pos)
        xy_embedding_1 += pos
        xy_embedding_2 = self.xy_mlp_2(xy_embedding_1)
        xy_embedding_2 += xy_embedding_1
        xy_embedding = self.xy_mlp_3(xy_embedding_2)

        image_embedding = self.imagemodel(image)
        feature_embedding = torch.cat([xy_embedding,image_embedding],-1)
        return self.predict(feature_embedding)

    