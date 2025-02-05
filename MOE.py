import torch
import torch.nn as nn
from gating import Gating

class MOE(nn.Module):
    def __init__(self, trained_experts):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)

    #freezing experts when they are training
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        num_experts = len(trained_experts)

        input_dim = trained_experts[0].layer1.in_features #extracts input dimension of the expert models
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        weights = self.gating(x)

        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        weights = weights.unsqueeze(1).expand_as(outputs) # ensures that the weights from the gating network correctly match the shape of the expert outputs before performing element-wise multiplication.
        return torch.sum(outputs * weights, dim=2)

