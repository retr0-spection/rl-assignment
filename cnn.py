import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # CHW after VecTransposeImage

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Adaptive pooling for robustness
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # Compute flatten size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, observation_space.shape[1], observation_space.shape[2])
            n_flatten = self.pool(self.cnn(sample)).view(1, -1).shape[1]

        # Lightweight attention over flattened feature vector
        #
        #
        self.attn = nn.MultiheadAttention(embed_dim=n_flatten, num_heads=2, dropout=0.2, batch_first=True, device="mps")


        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is already CHW from VecTransposeImage
        x = self.cnn(obs)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x_attn, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        return self.linear(x_attn.squeeze(1))
