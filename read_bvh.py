import os

import bvh
import numpy as np
from scipy.spatial.transform import Rotation as R
from visualize_motion import visualize_motion
from utils import read_bvh

POSITION_CHANNELS = ['Xposition', 'Yposition', 'Zposition']
ROTATION_CHANNELS = ['Xrotation', 'Yrotation', 'Zrotation']
HIP_NAME = 'Hips'
EULER_ORDER = 'xyz'


def euler_to_rot6d(euler_angles, is_degrees=True):
    r_obj = R.from_euler(seq=EULER_ORDER, angles=euler_angles, degrees=is_degrees)
    rotation_matrix = r_obj.as_matrix()
    rotation_6d = rotation_matrix[:, :2]
    return rotation_6d


def calculate_world_transform(joint_parent_map, joint_offsets, local_rotation_matrix_dict, hip_world_position,
                              world_rotation_matrix_dict, world_position_dict, joint_name):
    if joint_name in world_rotation_matrix_dict and joint_name in world_position_dict:
        return

    joint_local_rotation = local_rotation_matrix_dict[joint_name]

    if joint_name not in joint_parent_map:
        assert joint_name == HIP_NAME
        world_rotation_matrix_dict[joint_name] = joint_local_rotation
        world_position_dict[joint_name] = hip_world_position
        return

    parent_name = joint_parent_map[joint_name]
    calculate_world_transform(joint_parent_map, joint_offsets, local_rotation_matrix_dict, hip_world_position,
                              world_rotation_matrix_dict, world_position_dict, parent_name)
    parent_world_rotation = world_rotation_matrix_dict[parent_name]
    parent_world_position = world_position_dict[parent_name]
    world_rotation_matrix_dict[joint_name] = parent_world_rotation @ joint_local_rotation
    joint_local_position = parent_world_rotation @ joint_offsets[joint_name]
    world_position_dict[joint_name] = parent_world_position + joint_local_position


class BvhMocap:
    def __init__(self, bvh_file):
        bvh_object = read_bvh(bvh_file)
        self.num_frames = bvh_object.nframes
        self.frame_rate = round(1 / bvh_object.frame_time)
        self.joint_names = bvh_object.get_joints_names()
        self.num_joints = len(self.joint_names)
        self.joint_parent_map = {}
        for joint_name in self.joint_names:
            parent_joint = bvh_object.joint_parent(joint_name)
            if parent_joint:
                self.joint_parent_map[joint_name] = parent_joint.name
        self.motion_data = {}
        self.joint_indices = {joint_name: bvh_object.get_joint_index(joint_name) for joint_name in self.joint_names}
        self.joint_offsets = {joint_name: np.array(bvh_object.joint_offset(joint_name)) for joint_name in self.joint_names}
        self.hip_position = np.array(bvh_object.frames_joint_channels(HIP_NAME, POSITION_CHANNELS))
        self.joint_rotations = {}
        for joint_name in self.joint_names:
            joint_rotations = bvh_object.frames_joint_channels(joint_name, ROTATION_CHANNELS)
            self.joint_rotations[joint_name] = np.array(joint_rotations)

    def export_local_rotation_matrix_dict(self):
        local_rotation_matrix_dict = {}
        for joint_name in self.joint_names:
            local_rotation = self.joint_rotations[joint_name]
            local_rotation_matrix = R.from_euler(EULER_ORDER, local_rotation, degrees=True).as_matrix()
            local_rotation_matrix_dict[joint_name] = local_rotation_matrix
        return local_rotation_matrix_dict

    def export_world_transform(self):
        local_rotation_matrix_dict = self.export_local_rotation_matrix_dict()
        world_rotation_matrix_dict, world_position_dict = {}, {}
        for joint_name in self.joint_names:
            calculate_world_transform(self.joint_parent_map, self.joint_offsets,
                                      local_rotation_matrix_dict, self.hip_position,
                                      world_rotation_matrix_dict, world_position_dict,
                                      joint_name)
        return world_rotation_matrix_dict, world_position_dict

    def export_array(self):
        local_rotation_matrix_dict = self.export_local_rotation_matrix_dict()
        motion_array = [self.hip_position]
        for joint_name in self.joint_names:
            local_rotation_matrix = local_rotation_matrix_dict[joint_name]
            local_rotation6d = local_rotation_matrix[:, :, :2].reshape(-1, 6)
            motion_array.append(local_rotation6d)
        return np.concatenate(motion_array, axis=1)

    def visualize(self, start_frame_idx=None, end_frame_idx=None, save_path=None):
        if start_frame_idx is None:
            start_frame_idx = 0
        if end_frame_idx is None:
            end_frame_idx = self.num_frames
        _, world_positions = self.export_world_transform()
        visualize_motion(self.joint_names, self.joint_parent_map, world_positions, start_frame_idx,
                         end_frame_idx, self.frame_rate, save_path)


def main():
    bvh_file = "datasets/lafan1/dance2_subject2.bvh"
    bvh_obj = BvhMocap(bvh_file)
    data = bvh_obj.export_array()
    print(data.shape)


if __name__ == "__main__":
    main()
