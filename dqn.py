import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(CNN_DQN, self).__init__()

        # CNN feature extractor

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Flatten CNN output size

        self.fc_input_dim = self._get_conv_output_dim(input_channels)

        # DNN (fully-conn) for Q-value prediction
        #
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_output_dim(self, input_channels):
        # dummy forward pass to calculate output size dynamically
        #
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        with torch.no_grad():
            x = self.cnn(dummy_input)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        if len(x.shape) == 3:  # single frame
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
