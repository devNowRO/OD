from dataloader import *
import sys, pdb
from pathlib import Path
import torch
import torch.nn as nn

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from roadscene2vec.data.dataset import SceneGraphDataset
from roadscene2vec.learning.util.metrics import get_metrics, log_wandb, log_wandb_transfer_learning 

scene_graph_dataset  = SceneGraphDataset()
from torch.utils.data import DataLoader
from model import GCN_LSTM_PositionPredictor
from torch.utils.data import DataLoader, random_split

from torch.utils.data import DataLoader, random_split
graph_dir = "/home/irfan/roadscene2vec/examples/town2"
position_txt = "/home/irfan/roadscene2vec/examples/transferdata/pos.txt"
dataset = ScenegraphSequenceDataset(
    graph_dir=graph_dir,
    position_txt=position_txt,
    sequence_length=5
)
print(f"Dataset length: {len(dataset)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cuda"
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])



def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
    }

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)

print(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")

model = GCN_LSTM_PositionPredictor()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
i=0
best_test_loss=40.0
for epoch in range(100):
    model.train()
    for batch in train_loader:
        i+=1
 
        pred=model(
                batch["node_features_seq"].to(device),
                batch["edge_index_seq"].to(device),
                batch["prev_positions_seq"].to(device)
            )
      
        loss = loss_fn(pred, batch["target_position"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    for batch in test_loader:
        i+=1
      
        pred=model(
                batch["node_features_seq"].to(device),
                batch["edge_index_seq"].to(device),
                batch["prev_positions_seq"].to(device)
            )
       
        loss_test = loss_fn(pred, batch["target_position"].to(device))

    

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Test Loss = {loss_test.item():.4f}")

    # ---- save best model ----
    if loss_test.item() < best_test_loss:
        best_test_loss = loss_test.item()
        torch.save(model.state_dict(), "best_model_weights1.pth")
        print(f"✅ Saved new best model at epoch {epoch+1} with Test Loss {best_test_loss:.4f}")
model.load_state_dict(torch.load("model_weights1.pth"))

