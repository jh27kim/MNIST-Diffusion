import os 
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class InstanceContextUnet(nn.Module):
    def __init__(
        self, in_channels, instance_size,
        n_feat = 256, n_classes=10, n_regions=4, 
    ):
        super(InstanceContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.n_regions = n_regions
        self.instance_size = instance_size

        self.init_conv = ResidualConvBlock(
            in_channels, n_feat, is_res=True
        )

        self.down1 = UnetDown(n_feat, n_feat)
        down_inst_size_1 = self.instance_size // 2
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        down_inst_size_2 = down_inst_size_1 // 2

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(down_inst_size_2), nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)
        
        self.regionembed1 = EmbedFC(n_regions, 2*n_feat)
        self.regionembed2 = EmbedFC(n_regions, 1*n_feat)
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                8 * n_feat, 2 * n_feat, down_inst_size_2, down_inst_size_2
            ),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, box, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        
        x = self.init_conv(x)
        down1 = self.down1(x) # 256, 128, 14, 14
        down2 = self.down2(down1) # 256, 256, 7, 7
        hiddenvec = self.to_vec(down2) # 256, 256, 1, 1

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float) # 256, 10
        box = nn.functional.one_hot(box, num_classes=self.n_regions).type(torch.float) # 256, 10
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1) # 256, 256, 1, 1
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1) # 256, 256, 1, 1
        bemd1 = self.regionembed1(box).view(-1, self.n_feat * 2, 1, 1) # 256, 256, 1, 1
        
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1) # 256, 128, 1, 1
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1) # 256, 128, 1, 1
        bemd2 = self.regionembed2(box).view(-1, self.n_feat, 1, 1) # 256, 128, 1, 1

        # could concatenate the context embedding here instead of adaGN
        hiddenvec = torch.cat((hiddenvec, temb1, cemb1, bemd1), 1) # 256, 1024, 1, 1
        
        up1 = self.up0(hiddenvec) # 256, 256, 4, 4
        up2 = self.up1(up1, down2) # 256, 128, 8, 8
        up3 = self.up2(up2, down1) # 256, 128, 28, 28
        out = self.out(torch.cat((up3, x), 1)) # 256, 1, 28, 28

        # up1 = self.up0(hiddenvec) # 256, 256, 7, 7
        # up2 = self.up1(cemb1*up1 + temb1 + bemd1, down2) # 256, 128, 14, 14
        # up3 = self.up2(cemb2*up2 + temb2 + bemd2, down1) # 256, 128, 28, 28
        # out = self.out(torch.cat((up3, x), 1)) # 256, 1, 28, 28
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class InstanceDDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(InstanceDDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        print("alphas_cumprod individual assigned")
        self.alphas_cumprod = torch.cumprod(self.alpha_t, dim=0) 
        
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c, box):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, box, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            noise = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * noise
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
    
    
    def unproject(self, xts, box, eff_size=(1, 28, 28)):
        res = []
        device = xts.device
        n_boxs, n_regions = box.shape
        size = xts.shape[1:]
        eff_n_sample = xts.shape[0] // n_regions
        assert xts.shape[0] & n_regions == 0, "n_sample must be divisible by n_regions"
        
        eff_xts = xts.view(eff_n_sample, n_regions, *size)
        for _i in range(len(eff_xts)):
            zts = torch.zeros((1, *eff_size)).to(device)
            weight = torch.zeros(1, *eff_size).to(device)
            
            for _j in range(n_regions):
                idx = _i*n_regions + _j
                x1, y1, x2, y2 = box[idx]
                
                zts[:, :, x1:x2, y1:y2] += eff_xts[_i][_j:_j+1]
                weight[:, :, x1:x2, y1:y2] += 1
                
            res.append(zts / weight)
        
        res = torch.cat(res, dim=0)
        
        return res
    
    
    def project(self, zts, box):
        x_i = []
        n_regions, _ = box.shape
        
        n_samples = zts.shape[0]
        per_instance = n_samples // n_regions
        
        for _i in range(n_samples):
            for _j in range(per_instance):
                idx = _i*per_instance + _j
                x1, y1, x2, y2 = box[idx]
                x_i.append(zts[_i:_i+1, :, x1:x2, y1:y2])
            
        x_i = torch.cat(x_i, dim=0)
        
        return x_i
    
    def inst_sample(
        self, eff_n_sample, inst_n_sample, size, device, guide_w = 0.0,
        box_cls=None, box=None, n_regions=4, eff_size=(1, 28, 28),
    ):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        
        zts = torch.randn(eff_n_sample, *eff_size).to(device) # B 1 28 28
        xts = self.project(
            zts=zts,
            box=box,
        ) # SB 1 16 16 
        
        n_cls = 10
        c_i = torch.arange(0, n_cls).to(device) # n_cls
        c_i = torch.repeat_interleave(c_i, n_regions) # n_cls * n_regions [0000 1111 2222 3333 ... 9999]
        c_i = c_i.repeat(int(eff_n_sample/n_cls)) # SB

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        box_i = box_cls.repeat(2)
        
        context_mask = context_mask.repeat(2)
        context_mask[inst_n_sample:] = 1. # makes second half of batch context free

        z_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(inst_n_sample,1,1,1)

            # double batch
            xts = xts.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            # split predictions and compute weighting
            eps = self.nn_model(
                xts, c_i, t_is, context_mask=context_mask,
                box=box_i,
            )
            eps1 = eps[:inst_n_sample]
            eps2 = eps[inst_n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            
            delta_t = self.unproject(
                eps, box
            )
            
            noise = torch.randn(eff_n_sample, *eff_size).to(device) if i > 1 else 0
            
            zts = (
                self.oneover_sqrta[i] * (zts - delta_t * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * noise
            )
            
            xts = self.project(
                zts=zts,
                box=box,
            )
            
            if i%20==0 or i==self.n_T or i<8:
                z_i_store.append(zts.detach().cpu().numpy())
        
        z_i_store = np.array(z_i_store)
        return zts, z_i_store
    
    
    def ddim_sample(
        self, eff_n_sample, inst_n_sample, size, device, guide_w = 0.0,
        box_cls=None, box=None, n_regions=4, eff_size=(1, 28, 28), sampling="ddim",
    ):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        
        zts = torch.randn(eff_n_sample, *eff_size).to(device) # B 1 28 28
        xts = self.project(
            zts=zts,
            box=box,
        ) # SB 1 16 16 
        
        n_cls = 10
        c_i = torch.arange(0, n_cls).to(device) # n_cls
        c_i = torch.repeat_interleave(c_i, n_regions) # n_cls * n_regions [0000 1111 2222 3333 ... 9999]
        c_i = c_i.repeat(int(eff_n_sample/n_cls)) # SB

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        box_i = box_cls.repeat(2)
        
        context_mask = context_mask.repeat(2)
        context_mask[inst_n_sample:] = 1. # makes second half of batch context free

        z_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(inst_n_sample,1,1,1)

            # double batch
            xts = xts.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            # split predictions and compute weighting
            eps = self.nn_model(
                xts, c_i, t_is, context_mask=context_mask,
                box=box_i,
            )
            eps1 = eps[:inst_n_sample]
            eps2 = eps[inst_n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            
            delta_t = self.unproject(
                eps, box
            )
            
            if sampling == "ddpm":
                noise = torch.randn(eff_n_sample, *eff_size).to(device) if i > 1 else 0
            
                zts = (
                    self.oneover_sqrta[i] * (zts - delta_t * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * noise
                )
                
            elif sampling == "ddim":
                current_alpha = self.alphas_cumprod[i]
                prev_alpha = self.alphas_cumprod[i-1] if i-1 >= 0 else 1.0
                    
                pred_z0 = (zts - torch.sqrt(1-current_alpha) * delta_t) / torch.sqrt(current_alpha)
                
                zts = torch.sqrt(prev_alpha) * pred_z0 + torch.sqrt(1-prev_alpha) * delta_t
            
            else:
                raise NotImplementedError(f"{sampling} method not implemented")
            
            xts = self.project(
                zts=zts,
                box=box,
            )
            
            if i%20==0 or i==self.n_T or i<8:
                z_i_store.append(zts.detach().cpu().numpy())
        
        z_i_store = np.array(z_i_store)
        return zts, z_i_store