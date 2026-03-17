import torch.nn as nn 
import torch
class dinosplus_classfier(nn.Module):
    def __init__(self,model,num):
        super().__init__()
        self.backbone=model
        clsdim= self.backbone.config.hidden_size
        self.fc=nn.Sequential(nn.Linear(clsdim,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,num))
    def forward(self,x):
        output=self.backbone(pixel_values=x)
        
        cls=output.last_hidden_state[:,0]
        logit=self.fc(cls)
        
        return logit