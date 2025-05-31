import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from visualize_motion import visualize_motion
from read_bvh import calculate_world_transform
from utils import read_json, read_pickle
from transformer import FlowMatchingTransformer
from train import relocate_motion

JOINT_NAMES = {
    "100STYLE": [
        'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head', 'RightCollar', 'RightShoulder',
        'RightElbow', 'RightWrist', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightHip',
        'RightKnee', 'RightAnkle', 'RightToe', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
    ],
    "lafan1": [
        'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe', 'RightUpLeg', 'RightLeg', 'RightFoot',
        'RightToe', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
        'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
    ]
}

JOINT_PARENT_MAP = {
    "100STYLE": {
        'Chest': 'Hips', 'Chest2': 'Chest', 'Chest3': 'Chest2', 'Chest4': 'Chest3', 'Neck': 'Chest4', 'Head': 'Neck',
        'RightCollar': 'Chest4', 'RightShoulder': 'RightCollar', 'RightElbow': 'RightShoulder',
        'RightWrist': 'RightElbow', 'LeftCollar': 'Chest4', 'LeftShoulder': 'LeftCollar', 'LeftElbow': 'LeftShoulder',
        'LeftWrist': 'LeftElbow', 'RightHip': 'Hips', 'RightKnee': 'RightHip', 'RightAnkle': 'RightKnee',
        'RightToe': 'RightAnkle', 'LeftHip': 'Hips', 'LeftKnee': 'LeftHip', 'LeftAnkle': 'LeftKnee',
        'LeftToe': 'LeftAnkle'
    },
    "lafan1": {
        'LeftUpLeg': 'Hips', 'LeftLeg': 'LeftUpLeg', 'LeftFoot': 'LeftLeg', 'LeftToe': 'LeftFoot', 'RightUpLeg': 'Hips',
        'RightLeg': 'RightUpLeg', 'RightFoot': 'RightLeg', 'RightToe': 'RightFoot', 'Spine': 'Hips', 'Spine1': 'Spine',
        'Spine2': 'Spine1', 'Neck': 'Spine2', 'Head': 'Neck', 'LeftShoulder': 'Spine2', 'LeftArm': 'LeftShoulder',
        'LeftForeArm': 'LeftArm', 'LeftHand': 'LeftForeArm', 'RightShoulder': 'Spine2', 'RightArm': 'RightShoulder',
        'RightForeArm': 'RightArm', 'RightHand': 'RightForeArm'
    }
}

JOINT_OFFSETS = {
    "lafan1":
        {
            'Hips': [193.614899, 90.21743, 358.243408], 'LeftUpLeg': [0.103457, 1.85782, 10.548506],
            'LeftLeg': [43.5, -1.9e-05, 2e-06], 'LeftFoot': [42.3722, -4e-06, 3e-06],
            'LeftToe': [17.300009, 2e-06, 6e-06], 'RightUpLeg': [0.103459, 1.857818, -10.548504],
            'RightLeg': [43.500042, -1.5e-05, 1e-05], 'RightFoot': [42.372261, -8e-06, 1e-05],
            'RightToe': [17.299995, -5e-06, 0.0], 'Spine': [6.901967, -2.603743, 1e-06],
            'Spine1': [12.5881, 1.6e-05, -5e-06], 'Spine2': [12.343201, -1.1e-05, -5e-06],
            'Neck': [25.832895, 8e-06, 6e-06], 'Head': [11.766613, -4e-06, -2e-06],
            'LeftShoulder': [19.745897, -1.480371, 6.000116], 'LeftArm': [11.284124, -1.2e-05, -8e-06],
            'LeftForeArm': [33.000046, 2e-06, 2.2e-05], 'LeftHand': [25.200014, 6e-06, 1.5e-05],
            'RightShoulder': [19.746101, -1.480377, -6.000068], 'RightArm': [11.284134, -3.5e-05, -8e-06],
            'RightForeArm': [33.000092, 1.9e-05, 6e-06], 'RightHand': [25.199776, 0.000139, 0.000431]
        }
}


@torch.inference_mode()
def flow_one_step(net: nn.Module, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
    t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
    return x_t + (t_end - t_start) * net(t=t_start + (t_end - t_start) / 2,
                                         x_t=x_t + net(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


@torch.inference_mode()
def in_paint_rot(model, motion, head_frames, tail_frames, device, n_steps):
    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
    x = motion.clone()
    noise = torch.randn_like(x[:, head_frames: -tail_frames, 3:]).to(device)
    x[:, head_frames: -tail_frames, 3:] = noise
    for i in trange(n_steps):
        y = flow_one_step(net=model, x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
        x[:, head_frames: -tail_frames, 3:] = y[:, head_frames: -tail_frames, 3:]
    return x


def rotation6d_to_matrix(rotation6d):
    if rotation6d.shape[-1] != 6:
        raise ValueError(f"Input rotation6d must have last dimension of size 6, but got {rotation6d.shape[-1]}")

    rotation6d = rotation6d.reshape(-1, 3, 2)
    a1 = rotation6d[:, :, 0]  # 第一列 (x 轴)
    a2 = rotation6d[:, :, 1]  # 第二列 (y 轴)
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b3 = np.cross(b1, a2, axis=-1)
    b3 = b3 / np.linalg.norm(b3, axis=-1, keepdims=True)
    b2 = np.cross(b3, b1, axis=-1)
    rotation_matrix = np.stack([b1, b2, b3], axis=-1)
    return rotation_matrix


def visualize_root_pos_joint_rot(motion, dataset, frame_rate, save_path=None):
    hip_world_position = motion[:, :3]
    local_rotation_matrix_dict = {}
    joint_rotations = motion[:, 3:].reshape(-1, 22, 6)
    for joint_index, joint_name in enumerate(JOINT_NAMES[dataset]):
        joint_rotation_6d = joint_rotations[:, joint_index, :]
        local_rotation_matrix_dict[joint_name] = rotation6d_to_matrix(joint_rotation_6d)
    world_rotation_dict, world_position_dict = {}, {}
    for joint_name in JOINT_NAMES[dataset]:
        calculate_world_transform(joint_parent_map=JOINT_PARENT_MAP[dataset],
                                  joint_offsets=JOINT_OFFSETS[dataset],
                                  local_rotation_matrix_dict=local_rotation_matrix_dict,
                                  hip_world_position=hip_world_position,
                                  world_rotation_matrix_dict=world_rotation_dict,
                                  world_position_dict=world_position_dict,
                                  joint_name=joint_name)
    num_frame = motion.shape[0]
    visualize_motion(JOINT_NAMES[dataset], JOINT_PARENT_MAP[dataset], world_position_dict,
                     0, num_frame - 1, frame_rate, save_path)


def generate(bvh_file, start_frame, num_samples):
    device = torch.device("cuda")
    config_path = "config.json"
    config = read_json(config_path)
    num_frame = config["window_len"]
    dataset = config["dataset"]
    model = FlowMatchingTransformer.from_config(num_frame, config["model"]).to(device)
    checkpoint_path = config["train"]["checkpoint"]
    assert os.path.exists(checkpoint_path)
    video_dir = config["inpainting"]["animation_save_dir"]
    head_frames = config["inpainting"]["head_frames"]
    tail_frames = config["inpainting"]["tail_frames"]
    num_step = config["inference"]["num_flow_step"]
    os.makedirs(video_dir, exist_ok=True)
    model_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict)
    in_motion_raw = read_pickle("data.pkl")[bvh_file][start_frame: start_frame + num_frame]
    in_motion_raw = relocate_motion(in_motion_raw)
    in_save_path = os.path.join(video_dir, "original.mp4")
    visualize_root_pos_joint_rot(in_motion_raw, dataset, frame_rate=30, save_path=in_save_path)

    stat = read_pickle("stat.pkl")
    mean, std = stat["mean"], stat["std"]
    in_motion = in_motion_raw.copy()
    in_motion[:, :3] = (in_motion_raw[:, :3] - mean) / std
    in_motion = torch.tensor(in_motion, dtype=torch.float).to(device).unsqueeze(0)
    for sample_index in range(num_samples):
        out_motion = in_paint_rot(model, in_motion, head_frames, tail_frames, device, num_step)
        out_motion = out_motion.squeeze(0).detach().cpu().numpy()
        out_motion[:, :3] = mean + out_motion[:, :3] * std
        out_save_path = os.path.join(video_dir, f"inpainting_{sample_index}.mp4")
        visualize_root_pos_joint_rot(out_motion, dataset, frame_rate=30, save_path=out_save_path)


if __name__ == "__main__":
    file = "datasets/lafan1/dance2_subject4.bvh"
    generate(file, start_frame=4000, num_samples=20)
