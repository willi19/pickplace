from flir_python.utils import file_io
import os
import pickle
import numpy as np
import transforms3d
from dexterous_hri.utils.dataset.hand_robot_viewer_isaac import (
    RobotHandDatasetISAACViewer,
)
from dex_retargeting.constants import RobotName, HandType

obj_exp_dict = {
    "bottle": "pickplace_0911",
}


for obj_nm, exp_name in obj_exp_dict.items():
    demo_path_list = file_io.get_demo_path_list("teleoperation", exp_name, obj_nm)
    for demo_path in demo_path_list:
        demo_ind = demo_path.split("/")[-1].split("_")[-1]
        obj_traj = file_io.load_demo_data(demo_path, "obj_traj")

        robot_qpos = np.load(f"{demo_path}/robot_wrist.npy")
        xarm_wrist_position = robot_qpos[:, :3].copy()
        xarm_wrist_rotmat = np.array(
            [
                transforms3d.axangles.axangle2mat(
                    robot_qpos[frame, 3:6].copy(),
                    np.linalg.norm(robot_qpos[frame, 3:6].copy()),
                    is_normalized=False,
                )
                for frame in range(robot_qpos.shape[0])
            ]
        )
        xarm_wrist_position = xarm_wrist_position.reshape(-1, 3, 1)
        xarm_wrist_rotmat = xarm_wrist_rotmat.reshape(-1, 3, 3)
        xarm_wrist_se3 = np.concatenate(
            [xarm_wrist_rotmat, xarm_wrist_position], axis=2
        )
        xarm_wrist_se3 = np.concatenate(
            [xarm_wrist_se3, np.zeros((xarm_wrist_se3.shape[0], 1, 4))], axis=1
        )

        allegro_qpos = np.load(f"{demo_path}/robot_qpos.npy")
        allegro_qpos = allegro_qpos[:, -16:].copy()
        for object_name in obj_traj.keys():
            object_pose = obj_traj[object_name]
