# Overview
This project focused on Subterranean applications as part of the DARPA SubT Challenge. Simulated data was generated using Microsoft Airsim. Real data was gathered by Team Explorer's (CMU/OSU) custom build ground vehicles in a variety of representative environments using primarily as Velodyne VLP-16.  

# Data format
Overall, we follow the votenet data format, more info here: https://github.com/facebookresearch/votenet/blob/main/doc/tips.md. Roughly, for each data element there are 4 relevant files:
* `scan_name + '_vert.npy` contains the raw points
* `scan_name + '_ins_label.npy` contains the instance labels for every point
* `scan_name + '_sem_label.npy` contains the semantic labels for every point
* `scan_name + '_bbox.npy` contains the instance bounding boxes

For an overview of the AirSim environment, see here: https://www.microsoft.com/en-us/research/video/fly-through-in-the-airsim-simulation-team-explorer-created-for-the-darpa-subt-challenge/

# Dataset
Link to the dataset is here: https://oregonstate.box.com/s/lbv150s5avvgvcvc3t5hr4zk6hdrs0jd. There are 3 files:
* airsim-updated is only synthetic data
* real is only real data
* airsim_and_real is a dataset where the two are mixed











