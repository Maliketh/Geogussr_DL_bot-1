import torch.nn as nn
import open_clip


def build_model(num_countries): #Pretty self-explanatory
    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")

    backbone = model.visual

    # Freeze weights
    for param in backbone.parameters():
        param.requires_grad = False

    # Wrap in a classifier
    class GeoGuessrCLIPResNet(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.fc = nn.Linear(backbone.output_dim, num_classes)

        def forward(self, x):
            feats = self.backbone(x)  # [B, embed_dim]
            return self.fc(feats)

    return GeoGuessrCLIPResNet(backbone, num_countries)
