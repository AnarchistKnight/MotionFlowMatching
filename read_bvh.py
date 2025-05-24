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


def calculate_world_transform(joint_parent_map, joint_offsets, joint_local_rotations, hip_world_position,
                              world_rotations, world_positions, joint_name):
    if joint_name in world_rotations and joint_name in world_positions:
        return

    joint_local_rotation = R.from_euler(EULER_ORDER, joint_local_rotations[joint_name], degrees=True).as_matrix()

    if joint_name not in joint_parent_map:
        assert joint_name == HIP_NAME
        world_rotations[joint_name] = joint_local_rotation
        world_positions[joint_name] = hip_world_position
        return

    parent_name = joint_parent_map[joint_name]
    calculate_world_transform(joint_parent_map, joint_offsets, joint_local_rotations, hip_world_position,
                              world_rotations, world_positions, parent_name)
    parent_world_rotation = world_rotations[parent_name]
    parent_world_position = world_positions[parent_name]
    world_rotations[joint_name] = parent_world_rotation @ joint_local_rotation
    world_positions[joint_name] = parent_world_position + parent_world_rotation @ joint_offsets[joint_name]


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
        self.hip_start = self.hip_position[0]
        self.hip_position[:, 0] = self.hip_position[:, 0] - self.hip_start[0]
        self.hip_position[:, 2] = self.hip_position[:, 2] - self.hip_start[2]
        self.joint_rotations = {}
        for joint_name in self.joint_names:
            joint_rotations = bvh_object.frames_joint_channels(joint_name, ROTATION_CHANNELS)
            self.joint_rotations[joint_name] = np.array(joint_rotations)

    def export_data(self):
        world_rotations = {}
        world_positions = {}
        for joint_name in self.joint_names:
            calculate_world_transform(self.joint_parent_map, self.joint_offsets, self.joint_rotations,
                                      self.hip_position, world_rotations, world_positions, joint_name)
        return world_rotations, world_positions

    def export_array(self):
        world_rotations, world_positions = self.export_data()
        result = []
        for joint_name in self.joint_names:
            world_rot = world_rotations[joint_name][:, :, :2].reshape(-1, 6)
            world_pos = world_positions[joint_name]
            result.extend([world_rot, world_pos])
        result = np.concatenate(result, axis=1)
        return result

    def visualize(self, start_frame_idx=None, end_frame_idx=None, acceleration=1):
        if start_frame_idx is None:
            start_frame_idx = 0
        if end_frame_idx is None:
            end_frame_idx = self.num_frames
        frame_rate = int(self.frame_rate * acceleration)
        _, world_positions = self.export_data()
        visualize_motion(self.joint_names, self.joint_parent_map, world_positions, self.num_frames,
                         start_frame_idx, end_frame_idx, frame_rate)


