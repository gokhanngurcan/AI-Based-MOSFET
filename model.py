# model.py
import torch
import torch.nn as nn

class InverseModel(nn.Module):
    def __init__(self, input_dim=255, output_dim=10):
        super(InverseModel, self).__init__()

        # 1D CNN katmanları
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2)
        )

        # Flatten dimension hesaplama
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_output = self.cnn_layers(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).shape[1]

        # Ortak MLP feature extractor
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.025),

            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.025),

            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.025)
        )

        # Özel head index'leri
        self.special_indices = [5, 6, 7, 9]  # cs, c1, c2, vdd

        # Özel head'ler için iki katmanlı MLP tanımı
        self.special_heads = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.025),
                nn.Linear(128, 1)
            ) for i in self.special_indices
        })

        # Diğer komponentler için düz Linear head'ler
        self.regular_heads = nn.ModuleDict({
            str(i): nn.Linear(256, 1)
            for i in range(output_dim) if i not in self.special_indices
        })

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 255)
        x = self.cnn_layers(x)  # (B, 128, L)
        x = x.view(x.size(0), -1)
        x = self.shared_mlp(x)  # (B, 256)

        outputs = []
        for i in range(10):
            if str(i) in self.special_heads:
                outputs.append(self.special_heads[str(i)](x))  # (B, 1)
            else:
                outputs.append(self.regular_heads[str(i)](x))  # (B, 1)

        return torch.cat(outputs, dim=1)  # (B, 10)