import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from data_utils import PreprocessedGraphDataset
from rdkit.Chem import rdFingerprintGenerator
import matplotlib.pyplot as plt


# configuration
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
GNN_LAYERS = 6
BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 5
TEMP = 0.07


def pyg_to_rdkit(data):
    """Reconstructs an RDKit Mol object with initialized ring and valence info."""
    mol = Chem.RWMol()

    # 1. Add Atoms
    atomic_nums = data.x[:, 0].cpu().numpy()
    for atomic_num in atomic_nums:
        mol.AddAtom(Chem.Atom(int(atomic_num)))

    # 2. Add Bonds
    bond_types = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.AROMATIC,
    }

    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()

    for i in range(0, edge_index.shape[1], 2):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < v:
            b_type_idx = int(edge_attr[i, 0])
            rdkit_b_type = bond_types.get(b_type_idx, Chem.rdchem.BondType.SINGLE)
            mol.AddBond(u, v, rdkit_b_type)

    final_mol = mol.GetMol()

    try:
        final_mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(final_mol)
        Chem.SanitizeMol(
            final_mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except:
        final_mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(final_mol)

    return final_mol


# GNN encoder
class MolGNN(nn.Module):
    def __init__(self, hidden=256, layers=6):
        super().__init__()
        self.node_embeds = nn.ModuleList(
            [nn.Embedding(sz, hidden) for sz in [119, 9, 11, 12, 9, 5, 8, 2, 2]]
        )
        self.edge_embeds = nn.ModuleList(
            [nn.Embedding(sz, hidden) for sz in [22, 6, 2]]
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden)
            )
            self.convs.append(GINEConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))

    def forward(self, data):
        # embed features
        x = sum(
            [self.node_embeds[i](data.x[:, i]) for i in range(len(self.node_embeds))]
        )
        edge_attr = sum(
            [
                self.edge_embeds[i](data.edge_attr[:, i])
                for i in range(len(self.edge_embeds))
            ]
        )

        # message Passing
        for conv, bn in zip(self.convs, self.bns):
            h_in = x
            x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
            x = x + h_in  # residual

        return global_add_pool(x, data.batch)


# main model
class ContrastiveModel(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        # SciBERT for high-quality chemical text embeddings
        model_name = "allenai/scibert_scivocab_uncased"
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # projection head
        self.text_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, HIDDEN_DIM),
        )

    def forward(self, data, captions):
        # graph embedding
        g_emb = self.gnn(data)

        inputs = self.tokenizer(
            captions, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            mask = (
                inputs.attention_mask.unsqueeze(-1)
                .expand(outputs.last_hidden_state.size())
                .float()
            )
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            mean_pooled = sum_embeddings / torch.clamp(mask.sum(1), min=1e-9)

        # text embedding
        t_emb = self.text_proj(mean_pooled)

        return g_emb, t_emb


# training function
def train_contrastive(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in tqdm(loader):
        data = data.to(DEVICE)
        g_emb, t_emb = model(data, data.description)

        g_emb = F.normalize(g_emb, dim=-1)
        t_emb = F.normalize(t_emb, dim=-1)

        logits = torch.matmul(g_emb, t_emb.T) / TEMP
        labels = torch.arange(g_emb.size(0)).to(DEVICE)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# validation
def validate_contrastive(model, loader):
    """Evaluate the model on validation set without updating gradients."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Validating"):
            data = data.to(DEVICE)
            g_emb, t_emb = model(data, data.description)

            g_emb = F.normalize(g_emb, dim=-1)
            t_emb = F.normalize(t_emb, dim=-1)

            logits = torch.matmul(g_emb, t_emb.T) / TEMP
            labels = torch.arange(g_emb.size(0)).to(DEVICE)

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


# retrieval pipeline
def run_retrieval_pipeline(model, train_loader, test_loader):
    model.eval()
    train_embs, train_caps, train_fps = [], [], []

    print("Indexing Training Set...")
    for data in tqdm(train_loader):
        data = data.to(DEVICE)
        with torch.no_grad():
            # learned GNN features
            g_emb = F.normalize(model.gnn(data), dim=-1).cpu().numpy()
            train_embs.append(g_emb)
            train_caps.extend(data.description)
            # fingerprints
            for i in range(data.num_graphs):
                try:
                    mol = pyg_to_rdkit(data[i])
                    fp = gen.GetFingerprint(mol)
                except:
                    fp = gen.GetFingerprint(Chem.MolFromSmiles(""))  # empty molecule FP

                train_fps.append(fp)

    train_embs = np.vstack(train_embs).astype("float32")
    index = faiss.IndexFlatIP(HIDDEN_DIM)  # inner product for cosine similarity
    index.add(train_embs)

    print("Retrieving Test Set...")
    results = []
    for data in tqdm(test_loader):
        data = data.to(DEVICE)
        with torch.no_grad():
            q_emb = F.normalize(model.gnn(data), dim=-1).cpu().numpy()

        # top 50 candidates
        _, indices = index.search(q_emb, 50)

        for i in range(len(data.id)):
            try:
                test_mol = pyg_to_rdkit(data[i])
                test_fp = gen.GetFingerprint(test_mol)
            except:
                test_fp = gen.GetFingerprint(Chem.MolFromSmiles(""))

            best_sim, best_idx = -1.0, -1
            for neighbor_idx in indices[i]:
                sim = DataStructs.TanimotoSimilarity(test_fp, train_fps[neighbor_idx])
                if sim > best_sim:
                    best_sim, best_idx = sim, neighbor_idx

            results.append({"ID": data.id[i], "description": train_caps[best_idx]})
    return pd.DataFrame(results)


# train set
dataset_train = PreprocessedGraphDataset(TRAIN_GRAPHS)
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# model = ContrastiveModel(MolGNN(hidden=256, layers=6)).to(DEVICE)
model = ContrastiveModel(MolGNN(hidden=HIDDEN_DIM, layers=6)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# validation set
dataset_val = PreprocessedGraphDataset(VAL_GRAPHS)
loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

# test set
dataset_test = PreprocessedGraphDataset(TEST_GRAPHS)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

best_model = None
best_val_loss = float("inf")
best_epoch = -1
train_history = []
val_history = []
print("Starting Training...")

for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    train_loss = train_contrastive(model, loader_train, optimizer)
    val_loss = validate_contrastive(model, loader_val)
    train_history.append(train_loss)
    val_history.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict().copy()
        best_epoch = epoch

    print(f"Train Loss: {train_loss:.4f}")
    print(
        f"Validation Loss: {val_loss:.4f} (Best: {best_val_loss:.4f} at Epoch {best_epoch+1})"
    )

# plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_history, label="Train Loss", marker="o")
plt.plot(range(1, EPOCHS + 1), val_history, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300, bbox_inches="tight")
plt.close()

# save model
torch.save(best_model, "contrastive_model_last.pth")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

submission_df = run_retrieval_pipeline(model, loader_train, loader_test)
submission_df.to_csv("retrieval_with_val_submission.csv", index=False)
