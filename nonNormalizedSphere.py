import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Points(nn.Module):
    def __init__(self, N, D):
        super().__init__()
        self.N = N
        self.D = D
        self.points = nn.Parameter(torch.randn(N, D, device=device))

    def normalize_points(self):
       # return self.points / self.points.norm(dim=-1, p=2, keepdim=True)
       return self.points

    def loss(self):
        normalized_points = self.normalize_points()
        # N x N
        distance = torch.norm(
            normalized_points.reshape(self.N, 1, self.D)
            - normalized_points.reshape(1, self.N, self.D),
            p=2,
            dim=-1,
        )

        mask = torch.triu(
            torch.ones(self.N, self.N, dtype=bool, device=device), diagonal=1
        )
        #
        loss = torch.median(1 / (distance[mask] ** 2 + 1e-6)) # changed torch.mean to torch.median
        return loss


def optimize(N, D):
    points = Points(N, D).to(device)

    optim = torch.optim.SGD(points.parameters(), lr=1e-1) # changed Adam to SGD

    while True:
        optim.zero_grad()
        loss = points.loss()
        loss.backward()
        optim.step()

        yield points.normalize_points(), loss


def animate(num_iters, N, D):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    r = 1
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color="linen", alpha=0.5)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    (points,) = ax.plot(x, y, z, "*")
    txt = fig.suptitle("")

    iterable = optimize(N, D)

    def update_points(num, _x, _y, _z, points):
        txt.set_text("num={:d}".format(num))  # for debug purposes

        new_points, new_loss = next(iterable)
        new_x = new_points[:, 0].detach().cpu().numpy()
        new_y = new_points[:, 1].detach().cpu().numpy()
        new_z = new_points[:, 2].detach().cpu().numpy()

        points.set_data(new_x, new_y)
        points.set_3d_properties(new_z, "z")

        # return modified artists
        return points, txt

    ani = animation.FuncAnimation(
        fig, update_points, frames=num_iters, fargs=(x, y, z, points)
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

animate(1000, 12, 3)
