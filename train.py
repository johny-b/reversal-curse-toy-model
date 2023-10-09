# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from datasets import get_datasets

writer = SummaryWriter('runs')
device = "cuda"

# %%
max_num = 100

trainset, testset = get_datasets(max_num, 0.6, device=device)
batch_size = 1024

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

train_data = next(iter(DataLoader(trainset, batch_size=len(trainset))))
test_data = next(iter(DataLoader(testset, batch_size=len(testset))))

# %%
def get_loss_and_acc(model, X, y):
    preds = model(X)
    loss = F.mse_loss(preds, y)
    acc = (preds.round() == y).to(float).mean().item()
    return loss, acc

model = models.Transformer(max_num=max_num, d_model=64, nhead=8, num_layers=1).to(device)
optimizer = t.optim.AdamW(model.parameters(), lr=0.001)
epochs = 100000

for epoch_ix, epoch in enumerate(tqdm(range(epochs))):
    for batch_ix, (X, y) in enumerate(train_loader):
        loss, acc = get_loss_and_acc(model, X, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
    
    model.eval()
    
    train_loss, train_acc = get_loss_and_acc(model, *train_data)
    test_loss, test_acc = get_loss_and_acc(model, *test_data)
    
    writer.add_scalars(
        'batch',
        {
            'train_loss': train_loss.item(),
            'train_acc': train_acc,
            'test_loss': test_loss.item(),
            'test_acc': test_acc,
        },
        epoch_ix,
    )
    model.train()
# %%
