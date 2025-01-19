from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import (
    ConfigurationSaver,
    load_param,
    tensorboard_launcher,
)
from raisimGymTorch.env.bin.xarm import NormalSampler
from raisimGymTorch.env.bin.xarm import RaisimGymEnv
from raisimGymTorch.env.RewardAnalyzer import RewardAnalyzer
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
from flir_python.utils import file_io
from flir_python.fairmotion_ops import conversions

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../../raisimLib"

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", "r"))

# create environment from the configuration file
env = VecEnv(
    RaisimGymEnv(home_path + "/rsc", dump(cfg["environment"], Dumper=RoundTripDumper))
)
env.seed(cfg["seed"])

# # shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
num_threads = cfg["environment"]["num_threads"]

# # Training
#  n_steps = math.floor(cfg["environment"]["max_time"] / cfg["environment"]["control_dt"])

obj_exp_dict = {
    "bottle": "pickplace_0911",
}

demo_path_list = file_io.get_demo_path_list("teleoperation", "pickplace_0911", "bottle")

exp_name = "pickplace_0911"
scene_name = "bottle"

for demo_path in demo_path_list[:]:
    obj_traj = file_io.load_demo_data(demo_path, "obj_traj")
    scene_name = demo_path.split("/")[-1]
    robot_qpos = np.load(os.path.join(demo_path, "robot_qpos.npy"), allow_pickle=True)
    allegro_angles_tmp = robot_qpos[:, 7:]

    robot_qpos = np.concatenate([robot_qpos[:, :6], robot_qpos[:, 7:]], axis=1)

    n_steps = min(obj_traj["bottle"].shape[0], robot_qpos.shape[0])

    start = time.time()
    env.reset()
    env.turn_on_visualization()
    env.start_video_recording(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "xarm_test_1000"
        + ".mp4"
    )
    obj_pos = obj_traj["bottle"][:, :3, 3]
    obj_rot = obj_traj["bottle"][:, :3, :3]
    obj_quat = conversions.R2Q(obj_rot)
    obj_quat = obj_quat[:, [3, 0, 1, 2]]

    action_list = np.concatenate(
        [obj_pos[:n_steps], obj_quat[:n_steps], robot_qpos[:n_steps]], axis=1
    )

    for step in range(n_steps):

        frame_start = time.time()
        obs = env.observe(False)
        # action = np.zeros((2, act_dim))
        # action = np.ascontiguousarray(action, dtype=np.float32)
        action = action_list[step : step + 1]
        action = np.ascontiguousarray(action, dtype=np.float32)
        if step == 0:
            print(action[0, 7:])
        reward, dones = env.step(action)
        frame_end = time.time()
        wait_time = cfg["environment"]["control_dt"] - (frame_end - frame_start)

        if wait_time > 0.0:
            time.sleep(wait_time)

    env.stop_video_recording()
    env.turn_off_visualization()
