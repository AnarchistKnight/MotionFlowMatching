import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def visualize_motion(joint_names, joint_parent_map, world_positions, num_frames, start_frame_idx, end_frame_idx,
                     frame_rate):
    min_pos = np.array([np.inf, np.inf, np.inf])
    max_pos = np.array([-np.inf, -np.inf, -np.inf])

    # 遍历所有关节，计算所有帧的全局坐标范围，以确保动画过程中视图固定
    for joint_name in joint_names:
        joint_pos_data_for_range = world_positions[joint_name][start_frame_idx: end_frame_idx]
        min_joint_pos = np.min(joint_pos_data_for_range, axis=0)
        max_joint_pos = np.max(joint_pos_data_for_range, axis=0)
        min_pos = np.minimum(min_pos, min_joint_pos)
        max_pos = np.maximum(max_pos, max_joint_pos)

    range_coords = max_pos - min_pos
    max_range = np.max(range_coords).item()

    mid_x, mid_y, mid_z = (min_pos + max_pos) / 2

    # 设置统一的轴范围，并增加一些边距
    padding = 20  # 增加一些边距
    fig = plt.figure(figsize=(10, 8))  # 设置图窗大小
    ax = fig.add_subplot(111, projection='3d')

    def update_frame(current_frame_relative_idx):
        # 计算当前动画帧的绝对索引
        # FuncAnimation 的 frames 参数是动画的帧数，从 0 开始。
        # 这里需要将其映射到你实际数据的帧索引范围 [start_frame_idx, end_frame_idx)
        frame_abs_idx = start_frame_idx + current_frame_relative_idx
        if frame_abs_idx >= num_frames:
            # 如果当前帧超出了某个关节的数据范围，则跳过
            return

        # 重要的: 清除前一帧的绘图！这是你代码中缺少的部分。
        ax.clear()

        # 重新设置轴标签、视图范围和视角，因为 ax.clear() 会重置这些
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'帧: {frame_abs_idx + 1}/{num_frames}')  # 显示实际帧数
        ax.set_xlim(mid_x - max_range / 2 - padding, mid_x + max_range / 2 + padding)
        ax.set_ylim(mid_z - max_range / 2 - padding, mid_z + max_range / 2 + padding)
        ax.set_zlim(mid_y - max_range / 2 - padding, mid_y + max_range / 2 + padding)
        ax.view_init(elev=20, azim=45)

        # 遍历所有关节绘制点和线
        for joint_name in joint_names:
            world_pos = world_positions[joint_name][frame_abs_idx]
            # 绘制关节散点
            ax.scatter(world_pos[0], world_pos[2], world_pos[1], c='red', marker='o', s=5)

            # 绘制骨骼线 (如果存在父关节)
            if joint_name not in joint_parent_map:
                continue

            parent_name = joint_parent_map[joint_name]
            parent_world_pos = world_positions[parent_name][frame_abs_idx]
            ax.plot([world_pos[0], parent_world_pos[0]],
                    [world_pos[2], parent_world_pos[2]],
                    [world_pos[1], parent_world_pos[1]],
                    color='blue', linewidth=2)

        # 当 blit=False 时，通常返回一个空列表或 None 即可
        return []

    # FuncAnimation 的 frames 参数应该传递动画的“相对”帧数，即要播放多少帧
    # 这里是 end_frame_idx - start_frame_idx

    if start_frame_idx < 0 or end_frame_idx < start_frame_idx:
        print("动画帧数不足。请确保 end_frame_idx 大于 start_frame_idx。")
        return

    animation_frames = end_frame_idx - start_frame_idx
    anim = animation.FuncAnimation(fig, update_frame, frames=animation_frames, interval=1000 / frame_rate,
                                   blit=False, repeat=False)
    plt.show()
