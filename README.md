# AdvNet3D

Code/dataset in support of: Adversarial Training on Point Clouds for Sim-to-Real 3D Object Detection

Citation: R. DeBortoli, F. Li, A. Kapoor, G. Hollinger, "Adversarial training on point clouds for sim-to-real 3D object detection," Robotics and Automation Letters, vol. 6, no. 4, pp. 6662-6669, Oct. 2021.


# VoteNet
Our code using the VoteNet network is heavily based on the VoteNet repo: https://github.com/facebookresearch/votenet. Please refer to this repo for details on licenses, compiling necessary operations, and general structure of the codebase. 

Compiling necessary operations should only involve:
```
cd pointnet2
python setup.py install
```


If you want to just use our models the following files are relevant:
* `models/votenet/adv_class_module.py` has our adversarial discriminator model
* `models/votenet/votenet.py` has the overall object detection architecture (adjusted to include our adversarial architecture functionality)


# PartA2
Our code using the PartA2 network is heavily based on the PartA2 repo: https://github.com/open-mmlab/OpenPCDet. Please refer to this repo for details on licenses and general structure of the codebase.  

If you want to just use our models the following files are relevant:
* `models/parta2/pcdet/models/backbones_3d/spconv_unet.py` has our adversarial discriminator model
* `models/parta2/pcdet/models/detectors/PartA2_net.py` has training loss computations
* `models/parta2/tools/cfgs/subt_models/PartA2_net_free_subt.yaml` has our overall dataset config. 

many other aspects outside of these files are very similar to the original repo.

# Data
Please see `data` folder for links to the data and descriptions







