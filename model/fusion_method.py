import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Load configuration from YAML file
with open('/kaggle/working/Speech_project_Vin/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class SumFusion(nn.Module):
    def __init__(self, input_dim=config['fusion']['input_dim'], output_dim=config['fusion']['output_dim']):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=config['fusion']['input_dim'] * 2, output_dim=config['fusion']['output_dim']):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=config['fusion']['input_dim'], dim=config['fusion']['input_dim'], output_dim=config['fusion']['output_dim'], x_film=config['fusion']['film_x_film']):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):
        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)
        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=config['fusion']['input_dim'], dim=config['fusion']['input_dim'], output_dim=config['fusion']['output_dim'], x_gate=config['fusion']['gated_x_gate']):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output

# You can now select the fusion method from config['fusion']['type']
def create_fusion_model():
    fusion_type = config['fusion']['type']

    if fusion_type == 'sum':
        return SumFusion()
    elif fusion_type == 'concat':
        return ConcatFusion()
    elif fusion_type == 'film':
        return FiLM()
    elif fusion_type == 'gated':
        return GatedFusion()
    else:
        raise ValueError("Invalid fusion type in config")

if __name__ == '__main__':
    fusion_model = create_fusion_model()

    # Example forward pass with random data
    x = torch.rand(8, 512)  # Random input tensor x
    y = torch.rand(8, 512)  # Random input tensor y
    x_out, y_out, output = fusion_model(x, y)

    print(output.shape)  # Output tensor shape
