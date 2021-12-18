'''
Adversarial discriminator for point-based architectures
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_modules import PointnetSAModuleVotes

class AdvClassDiscriminator(nn.Module):
    '''
    Adversarial discriminator for point-based architectures
    '''
    def __init__(self):
        super().__init__() 

        self.pconv1 = PointnetSAModuleVotes(
                            npoint=128,
                            radius=1.0,
                            nsample=16,
                            mlp=[64, 32, 32, 67],
                            use_xyz=True,
                            normalize_xyz=True,
                            in_discriminator=True,
                            sample_strat='cpl'
                       )

        self.pconv2 = PointnetSAModuleVotes(
                            npoint=64,
                            radius=1.0,
                            nsample=16,
                            mlp=[64, 32, 32, 67],
                            use_xyz=True,
                            normalize_xyz=True,
                            in_discriminator=True,
                            sample_strat='cpl'
                       )

        self.pconv3 = PointnetSAModuleVotes(
                            npoint=64,
                            radius=1.0,
                            nsample=16,
                            mlp=[64, 32, 32, 64],
                            use_xyz=True,
                            normalize_xyz=True,
                            in_discriminator=True,
                            sample_strat='cpl'
                       )

        self.gavgpool = torch.nn.AvgPool2d((64,1), stride = 1)
        self.fc1 = torch.nn.Linear(64, 1)
        self.frozen = False

    def forward(self, end_points):
        """        
        """
        inp_xyz = end_points['grad_reverse_xyz']
        inp_feat = end_points['grad_reverse_features'] 


        if self.frozen:
            with torch.no_grad(): 
                xyz, features, fps_inds = self.pconv1(inp_xyz, inp_feat)
                xyz, features, fps_inds = self.pconv2(xyz, features)
                xyz, features, fps_inds = self.pconv3(xyz, features)
                global_features = self.gavgpool(features)
                class_output = self.fc1(global_features)
                end_points['adv_classification_output'] = class_output[:,0,:]
        else:
            xyz, features, fps_inds = self.pconv1(inp_xyz, inp_feat)
            xyz, features, fps_inds = self.pconv2(xyz, features)
            xyz, features, fps_inds = self.pconv3(xyz, features)
            global_features = self.gavgpool(features)
            class_output = self.fc1(global_features)
            end_points['adv_classification_output'] = class_output[:,0,:]

        return end_points


