# %%

from itertools import combinations

import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torch as t


def get_datasets(
    max_num: int,
    train_ratio: float,
    seed: int = None,
    device: str = "cpu",
) -> tuple[TorchDataset, TorchDataset]:
    rng = np.random.default_rng(seed)
    
    numbers = list(range(max_num))
    pairs = list(combinations(numbers, 2))
    
    #   Reverse order for randomly selected half of pairs
    #   This way information about a single pair can't be used 
    #   for predicting anything except this particular pair
    for ix, pair in enumerate(pairs):
        if rng.random() > 0.5:
            pairs[ix] = (pair[1], pair[0])
            
    rng.shuffle(pairs)
    
    split = int(len(pairs) * train_ratio)
    
    train_data = []
    test_data = []
    
    for ix, (x, y) in enumerate(pairs):
        first_second = [(x, y, 0), (y, x, 1)]
        rng.shuffle(first_second)
        
        if ix < split:
            train_data.extend(first_second)
        else:
            train_data.append(first_second[0])
            test_data.append(first_second[1])
            
    return Dataset(train_data, max_num, device=device), Dataset(test_data, max_num, device=device)

# %%

class Dataset(TorchDataset):
    def __init__(self, pairs: list[tuple[int, int, int]], max_num: int, device: str):
        self.pairs = pairs
        self.max_num = max_num
        self.device = device
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, ix) -> tuple[t.Tensor, float]:
        num_1, num_2, val = self.pairs[ix]
        
        x = t.zeros(self.max_num * 2, dtype=t.float32)
        x[num_1] = 1
        x[num_2 + self.max_num] = 1
                
        return x.to(self.device), t.tensor([val]).to(t.float32).to(self.device)

              
# %%
