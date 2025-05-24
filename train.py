from transformer import MotionTransformer
import json
import torch


def read_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


import torch
import torch.nn as nn
import torch.nn.functional as F


def simple_flow_matching_training_step(model, optimizer, data_batch, device):
    """
    一个简化的Flow Matching训练步骤示例。
    Args:
        model: MotionTransformer 实例
        optimizer: PyTorch优化器
        data_batch: 包含 motion_data (num_frames, num_joints, features) 和 time_steps (num_frames)
        device: 训练设备 (e.g., 'cuda', 'cpu')
    """
    model.train()
    optimizer.zero_grad()

    full_motion_sequence, full_time_steps = data_batch

    # 将数据移动到指定设备
    full_motion_sequence = full_motion_sequence.to(device)
    full_time_steps = full_time_steps.to(device) # 假设 full_time_steps 是从 0 到 num_frames-1 的索引

    num_frames = full_motion_sequence.shape[0]
    total_loss = 0.0

    # 为了模拟Flow Matching，我们需要一个目标速度场
    # 一个简单的目标速度场可以是 (motion_frame_{i+1} - motion_frame_i) / delta_t
    # 假设 delta_t = 1 (帧间隔)

    # 我们只训练到倒数第二帧，因为需要下一帧来计算目标速度场
    # 在实际Flow Matching中，你会采样时间 t 并构造 x_t
    # 这里为了简单，我们还是基于帧间差异来计算目标速度
    for i in range(num_frames - 1):
        current_frame = full_motion_sequence[i].unsqueeze(0) # (1, num_joints, features)
        next_frame = full_motion_sequence[i+1].unsqueeze(0) # (1, num_joints, features)

        # 对应的帧索引
        current_frame_idx = full_time_steps[i].unsqueeze(0).unsqueeze(1) # (1, 1)

        # 预测速度场
        predicted_velocity = model(current_frame, current_frame_idx)

        # 计算目标速度场
        target_velocity = next_frame - current_frame

        # 计算损失
        loss = F.mse_loss(predicted_velocity, target_velocity)
        total_loss += loss

    # 反向传播和优化
    if num_frames > 1:
        total_loss = total_loss / (num_frames - 1) # 平均每一帧的损失
        total_loss.backward()
        optimizer.step()

    return total_loss.item()


def generate_motion_sequence(model, start_motion_frame, start_frame_idx, num_generation_steps, device):
    """
    使用训练好的Flow Matching模型生成动作序列。
    """
    model.eval() # 设置为评估模式

    # 存储生成的动作序列 (从 numpy 数组开始，方便后续处理或保存)
    generated_sequence = [start_motion_frame.squeeze(0).cpu().numpy()]
    current_motion_frame = start_motion_frame.to(device)
    current_frame_idx = start_frame_idx.to(device) # (1, 1)

    with torch.no_grad():
        for step in range(num_generation_steps):
            # 预测速度场
            predicted_velocity = model(current_motion_frame, current_frame_idx)

            # 使用欧拉积分更新下一帧动作
            # next_motion_frame = current_motion_frame + predicted_velocity * delta_t
            # 假设 delta_t = 1 (帧间隔)
            next_motion_frame = current_motion_frame + predicted_velocity

            # 更新当前帧和帧索引
            current_motion_frame = next_motion_frame
            current_frame_idx = current_frame_idx + 1 # 增加帧索引

            generated_sequence.append(current_motion_frame.squeeze(0).cpu().numpy())

    return generated_sequence


def main():
    config_path = "config.json"
    config = read_json(config_path)
    model = MotionTransformer()
    batch_size = 64
    x = torch.zeros([batch_size, 23, 9])
    t = torch.zeros([batch_size, 1])
    y = model(x, t)
    from IPython import embed
    embed()


if __name__ == "__main__":
    main()
