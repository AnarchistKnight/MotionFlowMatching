from scripts.transformer import FlowMatchingTransformer
import torch
from scripts.utils import read_json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from scripts.utils import save_pickle, read_pickle
import numpy as np
from tqdm import tqdm, trange


def relocate_motion(motion):
    relocated_motion = motion.copy()
    relocated_motion[:, 0] -= motion[0, 0]
    relocated_motion[:, 2] -= motion[0, 2]
    return relocated_motion


class MotionDataset(Dataset):
    def __init__(self, pickle_path, window_len, window_step):
        self.data = read_pickle(pickle_path)
        self.files = list(self.data.keys())
        self.length = 0
        self.window_len = window_len
        self.window_step = window_step
        self.file_indices = []
        self.frame_indices = []
        for file_index, file in enumerate(self.files):
            if self.data[file].shape[0] < window_len:
                continue
            length = (self.data[file].shape[0] - window_len) // window_step + 1
            self.length += length
            for frame_index in range(0, self.data[file].shape[0] - window_len + 1, window_step):
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
        m1 = np.zeros([self.window_len, 3])
        m2 = np.zeros([self.window_len, 3])
        for index in trange(self.length):
            file_index = self.file_indices[index]
            frame_index = self.frame_indices[index]
            file = self.files[file_index]
            motion = self.data[file][frame_index: frame_index + self.window_len]
            relocated_motion = self.relocate_motion(motion[:, :3])
            m1 += relocated_motion
            m2 += relocated_motion ** 2
        self.mean = m1 / self.length
        self.std = np.sqrt(np.maximum(m2 / self.length - self.mean ** 2, 0))
        self.std[self.std < 1e-5] = 1
        save_pickle("stat.pkl", {"mean": self.mean, "std": self.std})

    def __getitem__(self, idx):
        file_index = self.file_indices[idx]
        frame_index = self.frame_indices[idx]
        file = self.files[file_index]
        motion = self.data[file][frame_index: frame_index + self.window_len]
        relocated_motion = relocate_motion(motion)
        relocated_motion[:, :3] = (relocated_motion[:, :3] - self.mean) / self.std
        return torch.tensor(relocated_motion, dtype=torch.float)


def train(config):
    window_len = config["window_len"]
    window_step = config["window_step"]
    dataset = MotionDataset(pickle_path=config["data_cache_path"],
                            window_len=window_len,
                            window_step=window_step)
    batch_size = config["train"]["batch_size"]
    num_epochs = config["train"]["epochs"]
    checkpoint_path = config["train"]["checkpoint"]
    device = torch.device("cuda")

    model = FlowMatchingTransformer.from_config(num_frames=window_len,
                                                config=config["model"]).to(device)
    if os.path.exists(checkpoint_path):
        model_state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(model_state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = []
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        num_batch = len(dataloader)
        for motion in tqdm(dataloader):
            motion = motion.to(device)
            noise = torch.randn_like(motion).to(device)
            t = torch.rand(motion.shape[0], dtype=torch.float).to(device)
            t_view = t.view(motion.shape[0], 1, 1)
            x_t = (1 - t_view) * noise + t_view * motion
            dx = motion - noise
            x_out = model(x_t, t)
            optimizer.zero_grad()
            loss = nn.MSELoss()(x_out, dx)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        torch.save(model.state_dict(), checkpoint_path)
        aver_loss = sum(total_loss) / num_batch
        print(f"Epoch: {epoch}, Loss: {aver_loss:.6f}")


if __name__ == "__main__":
    config_path = "config.json"
    train(read_json(config_path))
