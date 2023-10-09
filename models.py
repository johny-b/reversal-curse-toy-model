# %%
import torch as t
from torch import nn

# %%
class MLP(nn.Module):
    def __init__(self, max_num: int, hidden_width: int, num_hidden: int):
        super().__init__()
        
        layers = [nn.Linear(2 * max_num, hidden_width)]
        for _ in range(num_hidden - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width, hidden_width))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_width, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        # self.model = nn.Sequential(
        #     nn.Linear(2 * max_num, hidden_width),
        #     nn.ReLU(),
        #     nn.Linear(hidden_width, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        return self.model(x)
# %%

class Transformer(nn.Module):
    def __init__(self, *, max_num, d_model, nhead, num_layers):
        super().__init__()
        
        self.max_num = max_num
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.embed = nn.Linear(max_num, d_model)
        self.unembed = nn.Linear(2 * d_model, 1)
        
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )
        
    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 2 * self.max_num
        
        first_num = x[:, :self.max_num]
        second_num = x[:, self.max_num:]
        
        x = t.stack((
            self.embed(first_num),
            self.embed(second_num),
        )).permute((1,0,2))
        
        x = self.enc(x)
        x = x.flatten(start_dim=1)
        x = self.unembed(x)
        x = nn.functional.sigmoid(x)
        return x