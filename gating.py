import torch
import torch.nn as nn

class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(256, 512)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(512, 256)
        self.leaky_relu3 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.layer5 = nn.Linear(256, 128)
        self.leaky_relu4 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout(dropout_rate)

        self.layer6 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        x = self.layer4(x)
        x = self.leaky_relu3(x)
        x = self.dropout4(x)

        x = self.layer5(x)
        x = self.leaky_relu4(x)
        x = self.dropout5(x)

        x = self.layer6(x)  
        return torch.softmax(x, dim=1)
    
