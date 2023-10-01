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
