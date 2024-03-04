import torch
import torch.nn as nn
from torchsummary import summary


class DNN_5(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.f1 = nn.Linear(in_features=state_dim, out_features=128)
        self.dropout1 = nn.Dropout(0.3)
        self.f2 = nn.Linear(in_features=128, out_features=128)
        self.dropout2 = nn.Dropout(0.3)
        self.f3 = nn.Linear(in_features=128, out_features=128)
        self.dropout3 = nn.Dropout(0.3)
        self.f4 = nn.Linear(in_features=128, out_features=128)
        self.dropout4 = nn.Dropout(0.3)
        self.f5 = nn.Linear(in_features=128, out_features=action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ReLU(self.f1(x))
        x = self.dropout1(x)
        x = self.ReLU(self.f2(x))
        x = self.dropout2(x)
        x = self.ReLU(self.f3(x))
        x = self.dropout3(x)
        x = self.ReLU(self.f4(x))
        x = self.dropout4(x)
        x = self.ReLU(self.f5(x))
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device to use has been set to: "{device}"')
    state_dim = 10
    action_dim = 64
    model = DNN_5(state_dim, action_dim).to(device)
    print(summary(model, (1, 10)))
