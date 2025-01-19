## raisim_env_pickplace
This repo is for replaying teleoperation data from previous work.
raisimGymTorch is originally for robot learning but we are not using it here.

The gray robot simply follow the data trajectory while white one follow the trajectory by PDControll
### How to use this repo
python raisimGymTorch/env/envs/xarm/viewer.py 

### How to compile this repo
python setup.py develop --CMAKE_PREFIX_PATH /home/yc4ny/raisim_ws/raisimLib/raisim/linux --Debug
