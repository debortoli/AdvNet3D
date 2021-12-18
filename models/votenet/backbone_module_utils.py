'''
To support the various ways we can draw upon the PointNet2/Pointconv backbone

Contact: Bob DeBortoli (debortor@oregonstate.edu)
'''
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule, PointnetSAModuleMSGVotes

def get_sa_layers(network_size, input_feature_dim, downsample, cloud_npoints):
    '''
    Get pointnet2 set abstraction layers

    downsample: bool
            Whether to downsample (i.e. pool) the cloud during processing. Not \
            downsampling can possibly preserve fine-grained features for detecting small objs
    cloud_npoints: int
            number of points in the incoming cloud, if we dont downsample, used for defining 
            the number of points used in each layer
    '''

    if downsample:
        npoint_list = [2048, 1024, 512, 256]
    else:
        npoints = 2048  
        npoint_list = [npoints, npoints, npoints, npoints]

    if network_size == 'normal':
        sa1 = PointnetSAModuleVotes(
                npoint=npoint_list[0],
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        sa2 = PointnetSAModuleVotes(
                npoint=npoint_list[1],
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        # if we use pointdan, we need this for the adapted features
        # re-entry into the network
        pointdan_enc = PointnetSAModuleVotes(
                npoint=npoint_list[1],
                radius=1.2,
                nsample=16,
                mlp=[320, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        sa3 = PointnetSAModuleVotes(
                npoint=npoint_list[2],
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        sa4 = PointnetSAModuleVotes(
                npoint=npoint_list[3],
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        fp1 = PointnetFPModule(mlp=[256+256,256,256])
        fp2 = PointnetFPModule(mlp=[256+256,256,256])
        return sa1, sa2, sa3, sa4, fp1, fp2, pointdan_enc

    elif network_size == 'small':
        sa1 = PointnetSAModuleVotes(
                npoint=npoint_list[0],
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 32, 64],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        sa2 = PointnetSAModuleVotes(
                npoint=npoint_list[1],
                radius=0.4,
                nsample=32,
                mlp=[64, 64, 256],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        sa3 = PointnetSAModuleVotes(
                npoint=npoint_list[2],
                radius=0.8,
                nsample=16,
                mlp=[256, 64, 256],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        sa4 = PointnetSAModuleVotes(
                npoint=npoint_list[3],
                radius=1.2,
                nsample=16,
                mlp=[256, 64, 256],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        fp1 = PointnetFPModule(mlp=[256+256,256,256])
        fp2 = PointnetFPModule(mlp=[256+256,256,256])
        return sa1, sa2, sa3, sa4, fp1, fp2

    elif network_size == 'two-layer':

        sa1 = PointnetSAModuleVotes(
                npoint=npoint_list[0],
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )

        sa2 = PointnetSAModuleVotes(
                npoint=npoint_list[1],
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True,
                sample_strat = 'fps'
            )
        
        fp1 = PointnetFPModule(mlp=[256+128,256,256])
        return sa1, sa2, fp1
