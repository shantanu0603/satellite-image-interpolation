import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.to_device()
        
    def eval(self):
        self.flownet.eval()

    def to_device(self):
        self.flownet.to(device)

    def inference(self, img0, img1, timestep=0.5):
        # Ensure inputs have correct number of channels (3 each)
        assert img0.shape[1] == 3 and img1.shape[1] == 3, "Input images must have 3 channels each"
        
        # Ensure input dimensions are divisible by 32 (required for the multi-scale architecture)
        h, w = img0.shape[2], img0.shape[3]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        
        # Pad images
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8, 4, 2, 1]
        flow = None  # Initialize flow as None
        
        for scale in scale_list:
            if scale != 1:
                img_ = F.interpolate(imgs, scale_factor=1./scale, mode="bilinear", align_corners=False)
            else:
                img_ = imgs
            
            flow_comp = self.flownet(img_)
            if scale != 1:
                flow_comp = F.interpolate(flow_comp, scale_factor=scale, mode="bilinear", align_corners=False) * scale
            
            # Initialize or add flow
            if flow is None:
                flow = flow_comp
            else:
                # Ensure flow_comp is interpolated to match flow size
                if flow.shape != flow_comp.shape:
                    flow_comp = F.interpolate(flow_comp, size=(flow.shape[2], flow.shape[3]), 
                                            mode="bilinear", align_corners=False)
                flow = flow + flow_comp
        
        middle = timestep
        flow_0_1 = flow[:, :2]
        flow_1_0 = flow[:, 2:]
        
        flow_t_0 = -(1-middle) * middle * flow_0_1 + (middle**2) * flow_1_0
        flow_t_1 = ((1-middle)**2) * flow_0_1 - middle * (1-middle) * flow_1_0
        
        # Ensure output size matches input size
        if flow_t_0.shape[2:] != (h, w):
            flow_t_0 = F.interpolate(flow_t_0, size=(h, w), mode="bilinear", align_corners=False)
            flow_t_1 = F.interpolate(flow_t_1, size=(h, w), mode="bilinear", align_corners=False)
        
        return flow_t_0, flow_t_1

    def warp(self, img, flow):
        B, C, H, W = img.size()
        # Create meshgrid
        xx = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(device)
        
        # Normalize flow values to [-1, 1]
        flow = flow.clone()
        flow[:, 0] = flow[:, 0] / ((W-1.0) / 2.0)
        flow[:, 1] = flow[:, 1] / ((H-1.0) / 2.0)
        
        # Add flow to grid
        grid_ = (grid + flow).permute(0, 2, 3, 1)
        output = F.grid_sample(img, grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, 7, 1, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 4, 3, 1, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        return x 