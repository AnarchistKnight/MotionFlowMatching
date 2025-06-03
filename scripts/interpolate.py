import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class CubicInterpolation:
    def __init__(self, head_traj, tail_traj):
        self.head_traj = head_traj
        self.tail_traj = tail_traj
        self.head_frames = head_traj.shape[0]
        self.tail_frames = tail_traj.shape[0]

    def interpolate(self, completion_frames):
        num_frames = self.head_frames + completion_frames + self.tail_frames
        all_known_indices = np.concatenate((np.arange(self.head_frames), np.arange(self.head_frames + completion_frames, num_frames)))
        target_interp_indices = np.arange(self.head_frames, self.head_frames + completion_frames)

        traj = np.concatenate([self.head_traj, self.tail_traj], axis=0)
        completed_traj = np.zeros([completion_frames, 3])
        for i in range(3):  # 遍历 x, y, z 维度
            known_x = all_known_indices
            known_y = traj[:, i]
            cs = CubicSpline(known_x, known_y)  # 创建 CubicSpline 插值函数
            interpolated_values = cs(target_interp_indices)  # 对目标帧进行插值
            completed_traj[:, i] = interpolated_values  # 将插值结果填充到完整轨迹中
        return completed_traj

    def visualize(self, completion_frames):
        completed_traj = self.interpolate(completion_frames)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.head_traj[:, 0], self.head_traj[:, 1], self.head_traj[:, 2],
                label='已知头部', color='blue', linewidth=3)

        ax.plot(self.tail_traj[:, 0], self.tail_traj[:, 1], self.tail_traj[:, 2],
                label='已知尾部', color='green', linewidth=3)

        # 绘制插值部分
        ax.plot(completed_traj[:, 0], completed_traj[:, 1], completed_traj[:, 2],
                label='CubicSpline 插值', color='red', linestyle='--', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D 轨迹补齐 (CubicSpline Interpolation)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
