#used to see if there is enough motion between frames to add a new keyframe to the depth video.
# We can tweak this for object tracking if we sort of extend the vectors of motion to collide with the car's frame / path of motion
#filter and pass the output to MPC for object avoidance

import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock
import numpy as np

class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image): #enables to make use of the mixed precision training and cuda faster
        """ context features """
        #from the droid net model
        #net is feature map from context network.
        net, inp = self.cnet(image).split([128,128], dim=2) #cnet is the context network, split into net and inp parts
        return net.tanh().squeeze(0), inp.relu().squeeze(0)
    
    @torch.cuda.amp.autocast(enabled=True)
    def context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)
    
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)
    
    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """
        #to get the id with the quaternion 1,0,0,0 and translation 0,0,0. 
        Id = lietorch.SE3.Identity(1,).data.squeeze()
        # CNN feature height and width with 1/8 resolution
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        #add the batch dimension for cnn batch 
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        # remove the mean rgb values and divide by the standard deviation
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features using fnet features of the current frame
        gmap = self.__feature_encoder(inputs) #当前帧的特征, fnet

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0: #from depth_video.py is a torch.multiprocessing.Value, initialized to 0
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap # [1,128,H//8,W//8], [1,128,H//8,W//8], [1,128,H//8,W//8]
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]

            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0) #关键帧和当前帧之间的相关运算 [None,[0]]即保留第一行之后进行unsqueeze(0)，

            # approximate flow magnitude using 1 update iteration
            #this gives us the motion of an object between two frames
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]]) 
                self.net, self.inp, self.fmap = net, inp, gmap 
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])
            else:
                self.count += 1
