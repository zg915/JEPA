from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import random
import os

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class LowEnergyTwoModel(nn.Module):

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256, repr_dim=256, training=False, seed=19):

        super().__init__()
        #Add seed to control randomness 
        seed_everything(seed)
        self.seed = seed

        self.encoder = Encoder(input_shape=(1, 65, 65), repr_dim=repr_dim)
        self.predictor = Predictor(repr_dim=repr_dim, action_dim=2)
        self.wall_encoder = WallEncoder(input_shape=(1, 65, 65), repr_dim=128)
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = repr_dim
        self.action_dim = 2
        self.state_dim = (2, 64, 64)
        self.output_dim = output_dim
        self.training = training
    
    def forward(self, states, actions):
        #add seed 
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        bs, action_length, action_dim = actions.shape
        trajectory = states[:,:,0:1,:,:].clone() # first channel
        wall = states[:,:,1:,:,:].clone() # second channel

        encoded_states = self.encoder(trajectory[:,:1]) # only needs to encode the initial state
        encoded_wall = self.wall_encoder(wall[:, :1]) # only needs to encode the initial state

        predicted_states = []
        predicted_states.append(encoded_states[:,0])
        for i in range(action_length):
            prediction = self.predictor(predicted_states[i], actions[:,i], encoded_wall)
            predicted_states.append(prediction)
        predicted_states = torch.stack(predicted_states, dim=1)  # (17, bs, 256)

        encoded_target_states = None
        if self.training:
            encoded_target_states = self.encoder(trajectory[:, :-1])

        return predicted_states, encoded_target_states

    
    def loss(self, predicted_states, target_states):

        torch.manual_seed(self.seed)

        predicted_states = predicted_states[:, 1:]
        mse_loss = F.mse_loss(predicted_states, target_states)
        variance = target_states.var(dim=0).mean()
        var_loss = F.relu(1e-2 - variance).mean()

        return mse_loss + var_loss


class Encoder(nn.Module):
    def __init__(self, input_shape, repr_dim=256):
        super().__init__()

        # calculate linear layer input size
        C, H, W = input_shape
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        fc_input_dim = H * W * 64

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_input_dim, repr_dim)
        self.skip_fc = nn.Linear(C * input_shape[1] * input_shape[2], repr_dim)

    
    def forward(self, x):

        B, T, C, H, W = x.size()
        y = x
        x = x.contiguous().view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(B, T, -1) # [B, T, repr_dim]

        # skip connection
        y = y.contiguous().view(B * T, -1)  # [B * T, C * H * W]
        y = self.skip_fc(y)
        y = y.view(B, T, -1)  # Reshape back to [B, T, repr_dim]
        
        return x + y

class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=32, wall_dim=128):
        super().__init__()

        self.action_embedding = nn.Sequential(
            nn.Linear(2, action_dim),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(repr_dim + action_dim + wall_dim, repr_dim * 2),
            nn.ReLU(),
            nn.Linear(repr_dim * 2, repr_dim)
        )
    
    def forward(self, state, action, wall):
        action = self.action_embedding(action)
        x = torch.cat([state, action, wall.squeeze(1)], dim=1)
        x = self.fc(x)
        return x

class WallEncoder(nn.Module):
    def __init__(self, input_shape=(1, 65, 65), repr_dim=128):
        super().__init__()

        # Calculate linear layer input size
        C, H, W = input_shape  # (C=1, H=65, W=65)
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1  # After first conv (stride=2)
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1  # After second conv (stride=2)
        fc_input_dim = H * W * 64  # Final flattened size

        # Define CNN and fully connected layers
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1),  # Input channels = 1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_input_dim, repr_dim-1)  # Output size = 128

    def forward(self, x):
        
        B, T, C, H, W = x.size()  # Expect input shape (batch_size, 1, 1, 65, 65)
        x = x.contiguous().view(B * T, C, H, W)  # Combine batch and time dimensions
        x = self.cnn(x)  # Apply CNN
        x = self.flatten(x)  # Flatten to (B*T, fc_input_dim)
        x = self.fc(x)  # Apply fully connected layer
        x = x.view(B, T, -1)  # Reshape to (batch_size, 1, repr_dim=128)
        
        # add bias term
        ones = torch.ones(B, T, 1).to(x.device)
        x = torch.cat([x, ones], dim=-1)  
        
        return x