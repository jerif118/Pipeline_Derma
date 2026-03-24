import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,n_binary=1,n_outputs=11, freeze=True):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        if freeze:
            for param in self.dino.parameters():
                param.requires_grad=False
        self.bh = nn.Sequential(
            nn.Linear(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_binary)
        )
        self.fh = torch.nn.Sequential(
            nn.Linear(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_outputs)
        )
    def forward(self, x):
        features = self.dino(x)
        bh = self.bh(features)
        fh = self.fh(features)
        return bh, fh

    
    def unfreeze(self):
        for param in self.dino.parameters():
            param.requires_grad=True
