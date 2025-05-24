import torch
from torch import nn, Tensor
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class NeuralNet(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.layers(torch.cat((t, x_t), -1))


def flow(net: nn.Module, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
    t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
    return x_t + (t_end - t_start) * net(t=t_start + (t_end - t_start) / 2,
                                         x_t=x_t + net(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


def main():
    device = torch.device("cuda")
    # training
    model = NeuralNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    loss_fn = nn.MSELoss()

    for _ in trange(30000):
        x_1 = Tensor(make_moons(256, noise=0.05)[0]).to(device)
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(len(x_1), 1).to(device)

        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        optimizer.zero_grad()
        loss_fn(model(t=t, x_t=x_t), dx_t).backward()
        optimizer.step()

    # sampling
    x = torch.randn(300, 2).to(device)
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(20, 3), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1)

    axes[0].scatter(x.detach().cpu()[:, 0], x.detach().cpu()[:, 1], s=10)
    axes[0].set_title(f't = {time_steps[0]:.2f}')
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)

    for i in range(n_steps):
        x = flow(net=model, x_t=x, t_start=time_steps[i].to(device), t_end=time_steps[i + 1].to(device))
        axes[i + 1].scatter(x.detach().cpu()[:, 0], x.detach().cpu()[:, 1], s=10)
        axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()