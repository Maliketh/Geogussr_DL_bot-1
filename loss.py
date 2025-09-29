import torch
import torch.nn as nn

class VectorizedGeoLoss(nn.Module):
    def __init__(self, sim_matrix, class_weights):
        super().__init__()
        self.sim_matrix = sim_matrix
        self.class_weights = class_weights

    def forward(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)
        sim_w = self.sim_matrix[targets].to(outputs.device)
        class_w = self.class_weights[targets].to(outputs.device).unsqueeze(1)
        loss = -torch.sum(class_w * sim_w * torch.log(probs + 1e-9), dim=1)
        return loss.mean()
