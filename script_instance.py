from datetime import datetime 
import os 
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import sys 
sys.path.append("/home/jh27kim/code/current_projects/MNISTDiffusion")

from conditional_mnist.model.instance_model import InstanceDDPM
from conditional_mnist.model.instance_model import InstanceContextUnet

MNIST_IMG_SIZE = 28


def train_mnist():

    # hardcoding these here
    n_epoch = 20
    batch_size = 256
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    
    now = datetime.now()
    save_dir = f'./output/instance_concat_{now.strftime("%Y%m%d_%H%M%S")}/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    
    # Instance space denoising 
    instance_size = 16 # 14
    img_per_side = 2 # 3
    n_regions = img_per_side**2
    assert (instance_size * img_per_side - MNIST_IMG_SIZE) % (img_per_side-1) == 0, "instance size and img_per_side not compatible"
    stride = instance_size - (instance_size * img_per_side - MNIST_IMG_SIZE) // (img_per_side-1)
    
    region_list = []
    for i in range(0, 28, stride):
        for j in range(0, 28, stride):
            if i + instance_size <= MNIST_IMG_SIZE and j + instance_size <= MNIST_IMG_SIZE:
                region_list.append(
                    torch.tensor(
                        [i, j, i+instance_size, j+instance_size]
                    ).unsqueeze(0)
                )
            else:
                break
    region_list = torch.cat(region_list, dim=0).to(device)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ddpm = InstanceDDPM(
        nn_model=InstanceContextUnet(
            in_channels=1, 
            instance_size=instance_size,
            n_feat=n_feat, 
            n_classes=n_classes, 
            n_regions=n_regions,
        ), 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1
    )
    
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            
            box_cls = torch.randint(0, n_regions, (x.shape[0],))
            box = region_list[box_cls]
            
            instance_x = []
            for _i, _box in enumerate(box):
                x1, y1, x2, y2 = _box
                instance_x.append(x[_i:_i+1, :, x1:x2, y1:y2])
                
            instance_x = torch.cat(instance_x, dim=0)
            
            instance_x = instance_x.to(device)
            c = c.to(device)
            box_cls = box_cls.to(device)
            
            loss = ddpm(instance_x, c, box_cls)
            loss.backward()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
            
            # print("*"*50)
            # print("BREAK FOR DEBUGGING ")
            # print("BREAK FOR DEBUGGING ")
            # print("*"*50)
            # break
            
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            eff_n_sample = 4*n_classes # 4 instance spaces for 1 canonical space
            inst_n_sample = 4*n_classes * n_regions
            for w_i, w in enumerate(ws_test):
                box_cls = torch.arange(0, n_regions).repeat(eff_n_sample).to(device)
                box = region_list[box_cls].to(device)
            
                x_gen, x_gen_store = ddpm.inst_sample(
                    eff_n_sample,  inst_n_sample, 
                    (1, instance_size, instance_size), 
                    device, guide_w=w, 
                    box_cls=box_cls, box=box,
                    eff_size=(1, MNIST_IMG_SIZE, MNIST_IMG_SIZE),
                )

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(eff_n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(eff_n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(eff_n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
                    
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()

