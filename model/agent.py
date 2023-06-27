import torch
import torch.nn as nn
import torch.nn.functional as F

def model_loader(path):
    checkpoint = torch.load(path)

    return checkpoint['episode'], checkpoint['state_dict_A'], checkpoint['state_dict_A']

class Agent(nn.Module):
    
    def __init__(self, table):
        super(Agent, self).__init__()
        self.table = table
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, table[0] * table[1])
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.fc(out)

        return out
        