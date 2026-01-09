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


# --- 1. CONFIGURATION ---
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
GNN_LAYERS = 6
BATCH_SIZE = 32  # Keep small for Onyxia memory
LR = 5e-4
EPOCHS = 5
TEMP = 0.07  # Temperature for InfoNCE loss


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

    # --- CRITICAL FIX: Initialize Rings and Valences ---
    try:
        # 1. Force valence calculation (handles the N, 4 warning better)
        final_mol.UpdatePropertyCache(strict=False)

        # 2. Initialize the RingInfo (This fixes the RingInfo not initialized error)
        Chem.FastFindRings(final_mol)

        # 3. Basic sanitization (skipping KEKULIZE to avoid valence errors in exotic rings)
        Chem.SanitizeMol(
            final_mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except:
        # Fallback for very "broken" molecules
        final_mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(final_mol)

    return final_mol


# --- 2. GNN ENCODER (Upgraded 256-dim) ---
class MolGNN(nn.Module):
    def __init__(self, hidden=256, layers=6):
        super().__init__()
        # 9 node and 3 edge features as per challenge docs
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
        # Embed features
        x = sum(
            [self.node_embeds[i](data.x[:, i]) for i in range(len(self.node_embeds))]
        )
        edge_attr = sum(
            [
                self.edge_embeds[i](data.edge_attr[:, i])
                for i in range(len(self.edge_embeds))
            ]
        )

        # Message Passing
        for conv, bn in zip(self.convs, self.bns):
            h_in = x
            x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
            x = x + h_in  # Residual

        return global_add_pool(x, data.batch)


# --- 3. CONTRASTIVE MODEL ---
class ContrastiveModel(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        # SciBERT for high-quality chemical text embeddings
        # self.text_encoder = AutoModel.from_pretrained(
        #     "allenai/scibert_scivocab_uncased"
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "allenai/scibert_scivocab_uncased"
        # )
        # Load Galactica
        model_name = "facebook/galactica-1.3b"
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # FIX: Explicitly set pad token and padding side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # FREEZE Galactica to save memory and avoid "Asking to pad" errors in backprop
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Upgraded Projection Head (MLP is better than linear)
        self.text_proj = nn.Sequential(
            nn.Linear(2048, 512),  # Galactica-1.3b hidden size is 2048
            nn.ReLU(),
            nn.Linear(512, HIDDEN_DIM),
        )

    def forward(self, data, captions):
        # Graph branch
        g_emb = self.gnn(data)

        # Text branch (Frozen for faster alignment)
        inputs = self.tokenizer(
            captions, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use the mean of hidden states or the last token
            # For retrieval, Mean Pooling is often more stable than Last Token
            mask = (
                inputs.attention_mask.unsqueeze(-1)
                .expand(outputs.last_hidden_state.size())
                .float()
            )
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            mean_pooled = sum_embeddings / torch.clamp(mask.sum(1), min=1e-9)

        t_emb = self.text_proj(mean_pooled)

        return g_emb, t_emb


# --- 4. TRAINING FUNCTION (INFONCE) ---
def train_contrastive(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in tqdm(loader):
        data = data.to(DEVICE)
        g_emb, t_emb = model(data, data.description)

        # InfoNCE Loss
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


# --- 5. HYBRID RETRIEVAL & TANIMOTO RERANKING ---


def run_retrieval_pipeline(model, train_loader, test_loader):
    model.eval()
    train_embs, train_caps, train_fps = [], [], []

    # A. Indexing Training Set
    print("Indexing Training Set...")
    for data in tqdm(train_loader):
        data = data.to(DEVICE)
        with torch.no_grad():
            # Learned GNN features
            g_emb = F.normalize(model.gnn(data), dim=-1).cpu().numpy()
            train_embs.append(g_emb)
            train_caps.extend(data.description)
            # Expert Fingerprints
            for i in range(data.num_graphs):
                try:
                    mol = pyg_to_rdkit(data[i])
                    # Modern generator call
                    fp = gen.GetFingerprint(mol)
                except:
                    # CRITICAL: Fallback must be an ExplicitBitVect of the same size
                    fp = gen.GetFingerprint(Chem.MolFromSmiles(""))  # Empty molecule FP

                train_fps.append(fp)

    train_embs = np.vstack(train_embs).astype("float32")
    index = faiss.IndexFlatIP(HIDDEN_DIM)  # Inner product for cosine similarity
    index.add(train_embs)

    # B. Test Set Retrieval
    print("Retrieving Test Set...")
    results = []
    for data in tqdm(test_loader):
        data = data.to(DEVICE)
        with torch.no_grad():
            q_emb = F.normalize(model.gnn(data), dim=-1).cpu().numpy()

        # Coarse search: Top 50 candidates
        _, indices = index.search(q_emb, 50)

        for i in range(len(data.id)):
            try:
                test_mol = pyg_to_rdkit(data[i])
                test_fp = gen.GetFingerprint(test_mol)
            except:
                test_fp = gen.GetFingerprint(Chem.MolFromSmiles(""))

            # Now both are guaranteed to be ExplicitBitVect
            best_sim, best_idx = -1.0, -1
            for neighbor_idx in indices[i]:
                # This will now work without ArgumentError
                sim = DataStructs.TanimotoSimilarity(test_fp, train_fps[neighbor_idx])
                if sim > best_sim:
                    best_sim, best_idx = sim, neighbor_idx

            results.append({"ID": data.id[i], "description": train_caps[best_idx]})
    return pd.DataFrame(results)


# --- EXECUTION ---
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
print("Starting Training...")

for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    loss = train_contrastive(model, loader_train, optimizer)
    val_loss = train_contrastive(model, loader_val, optimizer)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        best_epoch = epoch

    print(f"Epoch {epoch}: {loss}")
    print(f"Validation Loss: {val_loss} (Best: {best_val_loss} at Epoch {best_epoch})")

# save model
torch.save(best_model, "contrastive_model_last.pth")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

submission_df = run_retrieval_pipeline(model, loader_train, loader_test)
submission_df.to_csv("retrieval_with_val_submission.csv", index=False)
