from PIL import Image
import argparse
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import os
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

MNIST_IMG_SIZE = 28
MAX_BATCH = 50
MAX_DISPLAY_IMGS = 100

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def pil_to_torch(pil_img):
    _np_img = np.array(pil_img).astype(np.float32) / 255.0
    _torch_img = torch.from_numpy(_np_img).permute(2, 0, 1).unsqueeze(0)
    return _torch_img

def torch_to_pil(tensor):
    if tensor.dim() == 4:
        _b, *_ = tensor.shape
        if _b == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor[0, ...]
    
    tensor = tensor.permute(1, 2, 0)
    
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = (np_tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(np_tensor)
    return pil_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--guidance_scale", type=int, default=2)
    parser.add_argument("--n_sample", type=int, default=100)
    
    return parser.parse_args()


def load_model(model_name: str, n_feat: int, n_classes: int, n_T: int, device: str,
               instance_size=None, img_per_side=None, n_regions=None):
    
    if model_name == "canonical":
        from conditional_mnist.model.canonical_model import CanonicalDDPM
        from conditional_mnist.model.canonical_model import CanonicalContextUnet
        
        ddpm = CanonicalDDPM(
            nn_model=CanonicalContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), 
            betas=(1e-4, 0.02), 
            n_T=n_T, 
            device=device, 
            drop_prob=0.1
        )

        ckpt_path = "/home/jh27kim/code/current_projects/MNISTDiffusion/conditional_mnist/ckpt/canonical_concatmodel_19.pth"
    
    elif model_name == "instance":
        from conditional_mnist.model.instance_model import InstanceDDPM
        from conditional_mnist.model.instance_model import InstanceContextUnet
        
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

        ckpt_path = "/home/jh27kim/code/current_projects/MNISTDiffusion/conditional_mnist/ckpt/instance_concatmodel_19.pth"
        
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    

    ddpm.load_state_dict(torch.load(ckpt_path))
    print("Loaded checkpoint")

    ddpm.to(device)
    ddpm.eval()
    print("Loaded model")
    
    return ddpm

@torch.no_grad()
def sample(
    ddpm, model, device, 
    guide_w, output_dir, 
    n_classes=10, instance_size=None,
    n_regions=None, region_list=None,
    n_sample=100,
):  
    
    assert n_sample % n_classes == 0, "n_sample must be divisible by n_classes"
    
    # n_sample = per_class * n_classes
    image_grid_list = []
    
    for mini_batch in range(n_sample // MAX_BATCH):
        print(f"[*] Sampling from {model} --- {(mini_batch+1) * 50} / {n_sample}")
        if model == "canonical":
            x_gen, x_gen_store = ddpm.cond_sample(
                n_sample=MAX_BATCH, 
                size=(1, 28, 28), 
                device=device, 
                guide_w=guide_w,
            )
            
        elif model == "instance":
            assert n_regions is not None, "n_regions must be provided"
            assert region_list is not None, "region_list must be provided"
            
            inst_n_sample = MAX_BATCH * n_regions

            box_cls = torch.arange(0, n_regions).repeat(MAX_BATCH).to(device)
            box = region_list[box_cls].to(device)
            
            print("inst_n_sample", inst_n_sample)
            print("box_cls", box_cls.shape)
            print("box", box.shape)

            with torch.no_grad():
                x_gen, x_gen_store = ddpm.inst_sample(
                    MAX_BATCH,  inst_n_sample, 
                    (1, instance_size, instance_size), 
                    device, guide_w=2.0, 
                    box_cls=box_cls, box=box,
                    n_regions=n_regions,
                    eff_size=(1, MNIST_IMG_SIZE, MNIST_IMG_SIZE),
                )
                
        x_gen = (x_gen).clamp(0, 1)
        x_gen = torch.cat([x_gen] * 3, dim=1)
        print("x_gen", x_gen.shape)
        image_grid_list += [torch_to_pil(x) for x in x_gen]
                
    c_i = torch.arange(0,10).to(device)
    c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
    
    print("c_i", c_i.shape)
    print(len(image_grid_list))
    
    # x_gen = (x_gen *-1 + 1).clamp(0, 1)
    img_save_dir = os.path.join(output_dir, "samples")
    os.makedirs(img_save_dir, exist_ok=True)
    for _i, img in enumerate(image_grid_list):
        img.save(f"{img_save_dir}/sample_{args.model}_guidance_{guide_w}_cls_{c_i[_i].item()}_{_i}.png")
    
    num_display = min(MAX_DISPLAY_IMGS, len(image_grid_list))
    res = image_grid(
            image_grid_list[:num_display], 
            rows=num_display // n_classes,
            cols=n_classes,
        )
            
    concat_dir = os.path.join(output_dir, f"sample_{args.model}_guidance_{guide_w}.png")
    res.save(concat_dir)
    
    print("[*] Saved at", output_dir)


def main(args):
    guide_w = args.guidance_scale 
    
    from datetime import datetime 
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    args.output_dir = os.path.join(args.output_dir, f"{args.model}_{now}")
    if os.path.exists(args.output_dir):
        print(f"{args.output_dir} already exists. Exitting...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    instance_size = 16 # 14
    img_per_side = 2 # 3
    n_regions = img_per_side**2
    
    ddpm = load_model(
        model_name=args.model, 
        n_feat=n_feat, 
        n_classes=n_classes, 
        n_T=n_T,
        device=device,
        instance_size=instance_size,
        img_per_side=img_per_side,
        n_regions=n_regions,
    )
    
    region_list = None
    if args.model == "instance":
        MNIST_IMG_SIZE = 28
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
        
    
    
    sample(
        n_sample=args.n_sample,
        ddpm=ddpm, 
        model=args.model,
        device=device, 
        guide_w=guide_w,
        output_dir=args.output_dir,
        n_classes=n_classes,
        instance_size=instance_size,
        n_regions=n_regions,
        region_list=region_list,
    )
    

if __name__ == "__main__":
    args = parse_args()
    main(args)