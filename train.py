from transformer import FlowMatchingTransformer, flow
import torch
from utils import read_json, read_pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from tqdm import trange
from utils import save_pickle, read_pickle
import numpy as np


class MotionDataset(Dataset):
    def __init__(self, pickle_path, window_len):
        self.data = read_pickle(pickle_path)
        self.files = list(self.data.keys())
        self.length = 0
        self.window_len = window_len
        self.file_indices = []
        self.frame_indices = []
        for file_index, file in enumerate(self.files):
            if self.data[file].shape[0] < window_len:
                continue
            self.length += self.data[file].shape[0] - window_len + 1
            for frame_index in range(0, self.data[file].shape[0] - window_len + 1, 1):
                self.file_indices.append(file_index)
                self.frame_indices.append(frame_index)
        self.compute_stat()

    def __len__(self):
        return self.length

    def compute_stat(self):
        if os.path.exists("stat.pkl"):
            stat = read_pickle("stat.pkl")
            self.mean = stat["mean"]
            self.std = stat["std"]
            return
        m1 = np.zeros([self.window_len, 23, 9])
        m2 = np.zeros([self.window_len, 23, 9])
        for index in trange(self.length):
            file_index = self.file_indices[index]
            frame_index = self.frame_indices[index]
            file = self.files[file_index]
            motion = self.data[file][frame_index: frame_index + self.window_len]
            m1 += motion
            m2 += motion ** 2
        self.mean = m1 / self.length
        self.std = np.sqrt(m2 / self.length - self.mean ** 2)
        save_pickle("stat.pkl", {"mean": self.mean, "std": self.std})

    def __getitem__(self, idx):
        file_index = self.file_indices[idx]
        frame_index = self.frame_indices[idx]
        file = self.files[file_index]
        motion = self.data[file][frame_index: frame_index + self.window_len]
        motion = (motion - self.mean) / self.std
        return torch.tensor(motion, dtype=torch.float)


def train(config_path):
    config = read_json(config_path)
    window_len = config["window_len"]
    dataset = MotionDataset(config["data"], window_len)
    batch_size = config["train"]["batch_size"]
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    device = torch.device("cuda")
    model = FlowMatchingTransformer(num_frames=window_len).to(device)
    checkpoint_path = "checkpoint.pth"
    if os.path.exists(checkpoint_path):
        model_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 示例：强制加载到CPU
        model.load_state_dict(model_state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    criterion = nn.MSELoss()  # 使用MSELoss来计算速度场误差
    num_epochs = config["train"]["epochs"]

    print_every = 100
    print_count = 0
    save_every = 10000
    num_batch = len(dataloader)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0
        for batch_idx, motion in enumerate(dataloader):
            # 将数据移动到设备
            motion = motion.to(device)
            noise = torch.randn_like(motion).to(device)
            t = torch.rand(batch_size, dtype=torch.float).to(device)
            t_view = t.view(batch_size, 1, 1, 1)
            x_t = (1 - t_view) * noise + t_view * motion
            dx = motion - noise
            x_out = model(x_t, t)
            optimizer.zero_grad()
            loss = criterion(x_out, dx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print_count += 1
            if print_count % print_every == 0:
                print(f"iteration: {print_count} / {num_batch}, loss: {loss.item()}")

            if print_count % save_every == 0:
                torch.save(model.state_dict(), checkpoint_path)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

        # 你可以在这里添加验证/评估逻辑，以及保存模型检查点
        if (epoch + 1) % config["train"].get("save_interval", 10) == 0:
            # 例如保存模型
            # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            # print(f"Model saved at epoch {epoch+1}")
            pass  # 占位符

    print("\nTraining finished.")
    # 可以选择保存最终模型
    # torch.save(model.state_dict(), config["train"].get("model_save_path", "final_model.pth"))
    # print(f"Final model saved to {config['train'].get('model_save_path', 'final_model.pth')}")


if __name__ == "__main__":
    train("config.json")
