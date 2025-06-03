import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualize_motion(joint_names, joint_parent_map, world_positions, start_frame_idx,
                     end_frame_idx, frame_rate, save_path=None):
    min_pos = np.array([np.inf, np.inf, np.inf])
    max_pos = np.array([-np.inf, -np.inf, -np.inf])

    for joint_name in joint_names:
        joint_pos_data_for_range = np.array(world_positions[joint_name][start_frame_idx: end_frame_idx])
        min_joint_pos = np.min(joint_pos_data_for_range, axis=0)
        max_joint_pos = np.max(joint_pos_data_for_range, axis=0)
        min_pos = np.minimum(min_pos, min_joint_pos)
        max_pos = np.maximum(max_pos, max_joint_pos)

    range_coords = max_pos - min_pos
    max_range = np.max(range_coords).item()  # 确保取的是最大的轴向范围，避免变形

    mid_x, mid_y, mid_z = (min_pos + max_pos) / 2

    # 设置统一的轴范围，并增加一些边距
    padding = 10  # 增加一些边距
    fig = plt.figure(figsize=(10, 8))  # 设置图窗大小
    ax = fig.add_subplot(111, projection='3d')

    # --- 为 blit=True 预先创建 Artist 对象 ---
    # 骨骼线：为每一根骨骼创建一个 Line3D 对象
    bone_lines = {}
    for joint_name, parent_name in joint_parent_map.items():
        # 使用一个初始的空数据创建 Line3D 对象
        line, = ax.plot([], [], [], color='blue', linewidth=2)
        bone_lines[(parent_name, joint_name)] = line

    # --- init_func 定义 ---
    # 在 blit=True 模式下，init_func 必须返回所有在动画开始前被绘制或修改的 Artist 对象
    def init_func():
        # 设置固定的视图范围和视角，这些在 init_func 中设置后通常不再改变
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(mid_x - max_range / 2 - padding, mid_x + max_range / 2 + padding)
        ax.set_ylim(mid_z - max_range / 2 - padding, mid_z + max_range / 2 + padding)  # 注意Y/Z轴交换
        ax.set_zlim(mid_y - max_range / 2 - padding, mid_y + max_range / 2 + padding)  # 注意Y/Z轴交换
        ax.view_init(elev=20, azim=45)

        # 初始化所有骨骼线数据
        for (parent_name, joint_name), line in bone_lines.items():
            parent_pos = world_positions[parent_name][start_frame_idx]
            joint_pos = world_positions[joint_name][start_frame_idx]
            line.set_data([parent_pos[0], joint_pos[0]],
                          [parent_pos[2], joint_pos[2]])  # Y和Z轴可能需要交换
            line.set_3d_properties([parent_pos[1], joint_pos[1]])  # Y和Z轴可能需要交换

        # 返回所有初始化的 Artist 对象
        return tuple(list(bone_lines.values()))

    # --- update_frame 定义 ---
    # 在 blit=True 模式下，update_frame 只需要更新 Artist 对象的数据，并返回它们
    def update_frame(current_frame_relative_idx):
        frame_abs_idx = start_frame_idx + current_frame_relative_idx

        # 更新骨骼线数据
        for child_name, parent_name in joint_parent_map.items():  # 遍历 map 中的父子关系
            line = bone_lines[(parent_name, child_name)]
            p_pos = world_positions[parent_name][frame_abs_idx]
            c_pos = world_positions[child_name][frame_abs_idx]

            line.set_data([p_pos[0], c_pos[0]],
                          [p_pos[2], c_pos[2]])  # Y和Z轴可能需要交换
            line.set_3d_properties([p_pos[1], c_pos[1]])  # Y和Z轴可能需要交换

        # --- 关键：返回所有被修改的 Artist 对象 ---
        # 返回一个元组，包含散点图、所有骨骼线和标题
        return tuple(list(bone_lines.values()))

    # --- FuncAnimation 参数 ---
    if start_frame_idx < 0 or end_frame_idx < start_frame_idx:
        print("动画帧数不足。请确保 end_frame_idx 大于 start_frame_idx。")
        return

    animation_frames = end_frame_idx - start_frame_idx
    # 设置 blit=True，并提供 init_func
    anim = animation.FuncAnimation(fig, update_frame, frames=animation_frames, init_func=init_func,
                                   interval=1000 / frame_rate, blit=True, repeat=False)
    if save_path is None:
        plt.show()
        return

    if save_path.endswith(".mp4"):
        writer_mp4 = animation.FFMpegWriter(fps=frame_rate, metadata=dict(artist='Me'), bitrate=1000)
        anim.save(save_path, writer=writer_mp4, dpi=200)  # 可以调整 dpi 提高质量
    else:
        writer_gif = animation.PillowWriter(fps=frame_rate, metadata=dict(artist='Me'))
        anim.save(save_path, writer=writer_gif, dpi=100)  # GIF 通常 dpi 较低，以减小文件大小

    plt.close()
