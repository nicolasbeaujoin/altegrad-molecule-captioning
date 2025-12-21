import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_scatter import scatter_softmax

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, MessagePassing
from torch_geometric.utils import to_dense_adj, add_self_loops

from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================
# Data paths
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"

TRAIN_EMB_CSV = "data/train_embeddings.csv"
VAL_EMB_CSV = "data/validation_embeddings.csv"

# Model parameters
NODE_VOCAB_SIZES = [119, 9, 11, 12, 9, 5, 8, 2, 2]
EDGE_VOCAB_SIZES = [22, 6, 2]

# Pre-Training parameters
MASK_RATE = 0.15
MASK_TOKEN_ID = 0
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MODEL: GINE + Virtual Node + Pre-Train
# =========================================================


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, eps=0.0, train_eps=True):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )

        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_emb):
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out, x):
        out = (1 + self.eps) * x + aggr_out
        return self.mlp(out)


class MolGNN(nn.Module):
    def __init__(self, in_dim=9, hidden=128, out_dim=256, layers=3, dropout=0.1):
        super().__init__()
        self.node_emb_layers = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden) for vocab_size in NODE_VOCAB_SIZES]
        )

        self.edge_emb_layers = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden) for vocab_size in EDGE_VOCAB_SIZES]
        )

        self.virtual_node_emb = nn.Embedding(1, hidden)
        nn.init.constant_(self.virtual_node_emb.weight.data, 0)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.vn_mlps = nn.ModuleList()

        for _ in range(layers):
            self.convs.append(GINEConv(hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

            self.vn_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                )
            )

        self.pretrain_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, NODE_VOCAB_SIZES[0])
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def _embed_nodes(self, x):
        x_long = x.long()
        h_cat = []
        for i, emb_layer in enumerate(self.node_emb_layers):
            h_cat.append(emb_layer(x_long[:, i]))

        return torch.stack(h_cat, dim=0).sum(dim=0)

    def _embed_edges(self, edge_attr):
        edge_attr_long = edge_attr.long()
        edge_embs = []
        for i, emb_layer in enumerate(self.edge_emb_layers):
            edge_embs.append(emb_layer(edge_attr_long[:, i]))

        return torch.stack(edge_embs, dim=0).sum(dim=0)

    def _gnn_forward(self, h, edge_index, edge_emb, batch_idx):
        vn_emb = self.virtual_node_emb(
            torch.zeros(batch_idx.max() + 1, dtype=torch.long, device=h.device)
        )

        for conv, bn, vn_mlp in zip(self.convs, self.bns, self.vn_mlps):
            h = h + vn_emb[batch_idx]
            h_in = h

            h = conv(h, edge_index, edge_emb)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

            h = h + h_in

            aggr_nodes = global_add_pool(h, batch_idx)
            vn_emb = vn_emb + vn_mlp(aggr_nodes)

        return h

    def forward_align(self, batch):
        h = self._embed_nodes(batch.x)
        edge_emb = self._embed_edges(batch.edge_attr)

        h = self._gnn_forward(h, batch.edge_index, edge_emb, batch.batch)

        g = global_add_pool(h, batch.batch)
        z = self.projector(g)
        return F.normalize(z, dim=-1)

    def forward_pretrain(self, batch):
        h = self._embed_nodes(batch.x)
        edge_emb = self._embed_edges(batch.edge_attr)

        h = self._gnn_forward(h, batch.edge_index, edge_emb, batch.batch)

        logits = self.pretrain_head(h)

        return logits

    def forward(self, batch):
        return self.forward_align(batch)

    def forward_features(self, batch):
        """
        Returns the raw graph representation (un-normalized) for the caption decoder.
        """
        # 1. Embed and run GNN
        h = self._embed_nodes(batch.x)
        edge_emb = self._embed_edges(batch.edge_attr)
        h = self._gnn_forward(h, batch.edge_index, edge_emb, batch.batch)

        # 2. Pool to global representation (Size: hidden=128)
        # We skip the 'projector' here because we will build a new specific
        # bridge for GPT-2 in the parent class.
        g = global_add_pool(h, batch.batch)
        return g


class Graph2CaptionV2(nn.Module):
    def __init__(self, pretrained_encoder, gpt2_model_name="gpt2-medium"):
        super().__init__()

        # 1. Use the pre-trained encoder
        self.encoder = pretrained_encoder

        # Determine dimensions
        # Assuming MolGNN hidden size is 128 (default in your code)
        gnn_hidden_dim = 128

        # 2. Text Decoder
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        gpt_emb_size = self.gpt2.config.n_embd  # 1024 for medium

        # 3. The Bridge (CRITICAL STEP)
        # Maps the pre-trained GNN size (128) to GPT-2 size (1024)
        self.projection = nn.Linear(gnn_hidden_dim, gpt_emb_size)

    def forward(self, data, text_input_ids, text_attention_mask):
        # 1. Get raw graph features (using the new method)
        graph_vec = self.encoder.forward_features(data)  # [batch, 128]

        # 2. Project to GPT space
        projected_emb = self.projection(graph_vec)  # [batch, 1024]

        # 3. Reshape for GPT-2: [batch, 1, 1024]
        projected_emb = projected_emb.unsqueeze(1)

        # 4. Get Text Embeddings
        text_embeds = self.gpt2.transformer.wte(text_input_ids)

        # 5. Concatenate: [Graph_Token, Text_Tokens...]
        inputs_embeds = torch.cat((projected_emb, text_embeds), dim=1)

        # 6. Extend Attention Mask
        batch_size = text_attention_mask.shape[0]
        ones = torch.ones((batch_size, 1), device=text_attention_mask.device)
        extended_mask = torch.cat((ones, text_attention_mask), dim=1)

        return self.gpt2(
            inputs_embeds=inputs_embeds, attention_mask=extended_mask
        ).logits

    def generate_caption(self, data, max_length=100):
        # Inference logic...
        graph_vec = self.encoder.forward_features(data)
        cur_input_embeds = self.projection(graph_vec).unsqueeze(1)

        generated_ids = []
        # (Greedy loop same as before...)
        for _ in range(max_length):
            outputs = self.gpt2(inputs_embeds=cur_input_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token_id)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            next_input_embeds = self.gpt2.transformer.wte(next_token_id)
            cur_input_embeds = torch.cat((cur_input_embeds, next_input_embeds), dim=1)

        return self.tokenizer.decode(
            [t.item() for t in generated_ids], skip_special_tokens=True
        )


# =========================================================
# Pre-Training
# =========================================================


def mask_atoms_for_pretraining(batch):
    num_nodes = batch.x.size(0)

    mask = torch.rand(num_nodes) < MASK_RATE

    if mask.sum() == 0 and num_nodes > 0:
        mask[torch.randint(0, num_nodes, (1,))] = True

    y_true = batch.x[:, 0].long()[mask]

    x_masked = batch.x.clone()

    x_masked[mask] = torch.tensor(
        [MASK_TOKEN_ID] * x_masked.size(1), dtype=x_masked.dtype, device=x_masked.device
    )

    batch_masked = batch.clone()
    batch_masked.x = x_masked

    batch_masked.mask = mask
    batch_masked.y_true = y_true

    return batch_masked


def train_epoch_pretrain(mol_enc, loader, optimizer, device):
    mol_enc.train()
    total_loss, total = 0.0, 0

    criterion = nn.CrossEntropyLoss()

    for graphs in loader:
        graphs = graphs.to(device)

        batch_masked = mask_atoms_for_pretraining(graphs)

        logits = mol_enc.forward_pretrain(batch_masked)

        masked_logits = logits[batch_masked.mask]

        y_true = batch_masked.y_true

        if y_true.numel() > 0:
            loss = criterion(masked_logits, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = graphs.num_graphs
            total_loss += loss.item() * bs
            total += bs

    return total_loss / total if total > 0 else 0.0


# =========================================================
# Training
# =========================================================


def contrastive_loss(mol_vec, txt_vec, temperature=0.07):
    mol_vec = F.normalize(mol_vec, dim=-1)
    txt_vec = F.normalize(txt_vec, dim=-1)

    logits = mol_vec @ txt_vec.t() / temperature
    labels = torch.arange(len(mol_vec), device=mol_vec.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_j) / 2


def train_epoch(mol_enc, loader, optimizer, device):
    mol_enc.train()

    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs = graphs.to(device)

        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        loss = contrastive_loss(mol_vec=mol_vec, txt_vec=txt_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


# =========================================================
# Evaluation
# =========================================================


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}

    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk
        results[f"Hit@{k}"] = hitk

    return results


# =========================================================
# Main Training Loop
# =========================================================
# def main():
#     print(f"Device: {DEVICE}")

#     train_emb = load_id2emb(TRAIN_EMB_CSV)
#     val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None

#     # emb_dim = len(next(iter(train_emb.values())))

#     if not os.path.exists(TRAIN_GRAPHS):
#         print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
#         print("Please run: python prepare_graph_data.py")
#         return

#     train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
#     train_dl = DataLoader(
#         train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
#     )

#     mol_enc = MolGNN(out_dim=emb_dim).to(DEVICE)

#     print("\n--- Pre-Training ---")

#     pretrain_optimizer = torch.optim.Adam(mol_enc.parameters(), lr=PRETRAIN_LR)

#     for ep in range(PRETRAIN_EPOCHS):
#         pretrain_loss = train_epoch_pretrain(
#             mol_enc, train_dl, pretrain_optimizer, DEVICE
#         )
#         print(f"Pretrain Epoch {ep+1}/{PRETRAIN_EPOCHS} - loss={pretrain_loss:.4f}")

#     print("\n--- Training ---")

#     optimizer = torch.optim.Adam(mol_enc.parameters(), lr=LR)

#     for ep in range(EPOCHS):
#         train_loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
#         if val_emb is not None and os.path.exists(VAL_GRAPHS):
#             val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
#         else:
#             val_scores = {}
#         print(f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f} - val={val_scores}")

#     model_path = "model_checkpoint.pt"
#     torch.save(mol_enc.state_dict(), model_path)
#     print(f"\nModel saved to {model_path}")


# if __name__ == "__main__":
#     main()


def main():
    # # --- Phase 1: Pre-Training (Masking) ---
    # print("\n=== Phase 1: Pre-Training GNN ===")

    # # Initialize basic GNN (hidden=128)
    # mol_enc = MolGNN(hidden=128).to(DEVICE)

    # # Load data for pre-training (We only need graphs, not text yet)
    # # Note: Pass 'None' for embeddings if you don't use them in pre-training
    # train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, None)
    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # pretrain_optimizer = torch.optim.Adam(mol_enc.parameters(), lr=PRETRAIN_LR)

    # for ep in range(PRETRAIN_EPOCHS):
    #     loss = train_epoch_pretrain(mol_enc, train_dl, pretrain_optimizer, DEVICE)
    #     print(f"Pretrain Epoch {ep+1}: {loss:.4f}")

    # print("Pre-training complete. Saving GNN weights.")
    # torch.save(mol_enc.state_dict(), "checkpoints/mol_enc.pt")

    # --- Phase 2: Captioning Fine-Tuning ---
    print("\n=== Phase 2: Training Graph2Caption ===")

    # 1. Initialize Combined Model using the PRE-TRAINED encoder
    mol_enc = MolGNN(hidden=128).to(DEVICE)
    state_dict = torch.load(
        "checkpoints/mol_enc_pretrained_best.pt", map_location=DEVICE
    )
    mol_enc.load_state_dict(state_dict)
    mol_enc.to(DEVICE)

    model = Graph2CaptionV2(pretrained_encoder=mol_enc, gpt2_model_name="gpt2-medium")
    model.to(DEVICE)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # 2. Optimizer (Fine-tune GNN slowly, Train Bridge/GPT normal)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": 1e-4,
            },  # Lower LR for pre-trained part
            {"params": model.projection.parameters(), "lr": 1e-3},
            {"params": model.gpt2.parameters(), "lr": 5e-5},  # Very low for GPT-2
        ]
    )

    # 3. Load Data with Descriptions (Re-load to ensure we access descriptions)
    # Ensure your PreprocessedGraphDataset loads data.description!
    train_ds_cap = PreprocessedGraphDataset(TRAIN_GRAPHS, None)
    train_loader = DataLoader(
        train_ds_cap, batch_size=16, shuffle=True
    )  # Smaller batch for GPT

    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Caption Epoch {epoch+1}")

        for batch in pbar:
            batch = batch.to(DEVICE)
            captions = batch.description  # Ensure this exists in your dataset class

            # Tokenize
            inputs = model.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=100,
                return_tensors="pt",
            ).to(DEVICE)

            optimizer.zero_grad()

            # Forward
            logits = model(batch, inputs.input_ids, inputs.attention_mask)

            # Shift for Causal Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()  # Skip first token

            # Note: The shapes might need slight adjustment depending on
            # if you included the graph token in logits.
            # Usually: logits has length L+1 (Graph + Text).
            # Labels has length L.
            # So logits[:, :-1, :] predicts logits[:, 1:, :] which should align with text.

            # Simplified alignment:
            # Logits [Batch, Seq+1, Vocab]
            # We want Logits[Graph] -> Label[Word1]
            # We want Logits[Word1] -> Label[Word2]
            shift_logits = logits[:, :-1, :]  # Remove last prediction
            shift_labels = inputs.input_ids  # Target is the text itself

            loss = criterion(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Save Checkpoint
        torch.save(model.state_dict(), f"checkpoints/g2cap_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
