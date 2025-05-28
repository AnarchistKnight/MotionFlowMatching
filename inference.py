import torch
import torch.nn as nn


def flow_one_step(net: nn.Module, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
    t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
    return x_t + (t_end - t_start) * net(t=t_start + (t_end - t_start) / 2,
                                         x_t=x_t + net(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


@torch.inference_mode()
def flow(model, device, num_frame, num_joint, joint_dim, n_steps=8):
    time_steps = torch.linspace(0, 1.0, n_steps + 1)
    x = torch.randn([1, num_frame, num_joint, joint_dim]).to(device)
    for i in range(n_steps):
        x = flow_one_step(net=model, x_t=x, t_start=time_steps[i].to(device), t_end=time_steps[i + 1].to(device))
    return x


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


def generate(num_samples):
    from utils import read_json, read_pickle
    from transformer import FlowMatchingTransformer
    from visualize_motion import visualize_motion
    import os
    device = torch.device("cuda")
    config_path = "config.json"
    config = read_json(config_path)
    num_frame = config["model"]["num_frames"]
    num_joint = config["model"]["num_joints"]
    joint_dim = config["model"]["joint_dim"]
    dataset = config["dataset"]
    model = FlowMatchingTransformer.from_config(config["model"]).to(device)
    checkpoint_path = config["train"]["checkpoint"]
    assert os.path.exists(checkpoint_path)
    video_dir = config["animation_save_dir"]
    os.makedirs(video_dir, exist_ok=True)
    model_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict)
    for sample_index in range(num_samples):
        motion = flow(model, device, num_frame, num_joint, joint_dim, n_steps=config["inference"]["num_flow_step"])
        motion = motion.squeeze(0).detach().cpu().numpy()
        stat = read_pickle("stat.pkl")
        mean = stat["mean"]
        std = stat["std"]
        motion = mean + motion * std
        motion_dict = {}
        for joint_index, joint_name in enumerate(JOINT_NAMES[dataset]):
            motion_dict[joint_name] = motion[:, joint_index, -3:]
        save_path = os.path.join(video_dir, f"{sample_index}.gif")
        visualize_motion(JOINT_NAMES[dataset], JOINT_PARENT_MAP[dataset], motion_dict,
                         0, num_frame - 1, 30, save_path)


if __name__ == "__main__":
    main()
