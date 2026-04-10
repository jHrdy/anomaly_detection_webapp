import torch.nn as nn
import torch

class AE_v3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=3, padding=1),  
            nn.BatchNorm1d(2),
            nn.ReLU(),
            
            nn.Conv1d(2, 4, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Conv1d(4, 8, kernel_size=3, padding=1),  
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),  
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            )


        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),  
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            nn.ConvTranspose1d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.ConvTranspose1d(4, 2, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(2),
            nn.ReLU(),
            
            nn.ConvTranspose1d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),

            nn.Linear(93, 96),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_model(path, model_architecture=AE_v3()):
    model = model_architecture
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model