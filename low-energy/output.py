import torch
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

def print_sample(sample):

    position_channel = sample[:, 0, :, :]  # [T, H, W] for position
    walls_channel = sample[:, 1, :, :]     # [T, H, W] for walls/doors
    position_channel = position_channel.cpu().numpy()
    walls_channel = walls_channel.cpu().numpy()

    steps, height, width = position_channel.shape
    for t in range(steps):
        print_image(position_channel[t], walls_channel[t])

    plot_full_trajectory(position_channel, walls_channel)

def print_image(position, walls):

    height, width = position.shape
    print(f'height={height}, width={width}')
    for i in range(height):
        row = ""
        for j in range(width):
            if walls[i, j] > 0:
                row += "|"
            elif position[i, j] > 0:
                row += "X"
            else:
                row += "."
        print(row)

def plot_image(position, walls):

        fig = plt.figure(figsize=(12, 6))

        a = fig.add_subplot(1, 2, 1)
        im_a = a.imshow(position, cmap="viridis")
        a.set_title("Position - Timestep 0")
        fig.colorbar(im_a, ax=a)

        b = fig.add_subplot(1, 2, 2)
        im_b = b.imshow(walls, cmap="gray")
        b.set_title("Walls/Doors - Timestep 0")
        fig.colorbar(im_b, ax=b)

        fig.tight_layout()
        fig.savefig('traject.png')

def plot_trajectory(position_channel, walls_channel):

    steps = position_channel.shape[0]

    fig, axes = plt.subplots(steps, 2, figsize=(12, 6 * steps))

    if steps == 1:
        axes = axes[np.newaxis, :]

    for t in range(steps):
        ax_pos = axes[t, 0]
        im_pos = ax_pos.imshow(position_channel[t], cmap="viridis")
        ax_pos.set_title(f"Position - Timestep {t}")
        fig.colorbar(im_pos, ax=ax_pos)

        ax_wall = axes[t, 1]
        im_wall = ax_wall.imshow(walls_channel[t], cmap="gray")
        ax_wall.set_title(f"Walls/Doors - Timestep {t}")
        fig.colorbar(im_wall, ax=ax_wall)

    fig.tight_layout()
    fig.savefig('traject.png')


def plot_full_trajectory(position_channel, walls_channel):
    steps, height, width = position_channel.shape
    walls = walls_channel[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    combined = walls.copy()
    for t in range(steps):
        timestep_layer = position_channel[t]
        combined = np.maximum(combined, timestep_layer)
    ax.imshow(combined, cmap="viridis", alpha=1.0)
    ax.set_title("Agent Trajectory Over Static Walls")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("traject.png")
    plt.close(fig)

def plot_state_norms(predicted_states, target_states):
    predicted_norms = torch.norm(predicted_states, dim=-1).view(-1).detach().cpu().numpy()
    target_norms = torch.norm(target_states, dim=-1).view(-1).detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.hist(predicted_norms, bins=50, alpha=0.5, label="Predicted States")
    plt.hist(target_norms, bins=50, alpha=0.5, label="Target States")
    plt.legend()
    plt.title("Norm Distributions of Predicted and Target States")
    plt.xlabel("Norm")
    plt.ylabel("Frequency")
    plt.show()

def plot_gradient_norms(gradient_norms):
    plt.figure(figsize=(10, 6))
    for part, norms in gradient_norms.items():
        plt.plot(norms, label=part) 
    plt.title("Gradient Norms Over Training")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Gradient Norm")
    plt.legend()
    plt.show()

def collect_gradient_norms(model):
    norms = {"encoder": 0, "target_encoder": 0, "predictor": 0}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "encoder" in name and "target_encoder" not in name:  # Encoder
                norms["encoder"] += grad_norm
            elif "target_encoder" in name:  # Target Encoder
                norms["target_encoder"] += grad_norm
            elif "predictor" in name:  # Predictor
                norms["predictor"] += grad_norm
    return norms