import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, num_actions=4):
        super(NeuralNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        with torch.no_grad():
            sample = torch.zeros(1, 1, *input_shape)
            conv_out = self.conv_layers(sample)
            flatten_size = conv_out.view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
