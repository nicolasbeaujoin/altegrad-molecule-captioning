import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, MessagePassing
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm 

from data_utils import load_id2emb, PreprocessedGraphDataset

# =========================================================
# CONFIG
# =========================================================
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"

NODE_VOCAB_SIZES = [119, 9, 11, 12, 9, 5, 8, 2, 2]
EDGE_VOCAB_SIZES = [22, 6, 2]

# Hyperparameters
MASK_RATE = 0.15
MASK_TOKEN_ID = 0
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1e-3
EPOCHS = 10
ENCODER_LR = 1e-4
PROJECTION_LR = 1e-3
GPT_LR = 5e-5
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MODEL: Encoder (GINE)
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
    def __init__(self, in_dim=9, hidden=128, layers=3, dropout=0.1):
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

    def forward_pretrain(self, batch):
        h = self._embed_nodes(batch.x)
        edge_emb = self._embed_edges(batch.edge_attr)
        h = self._gnn_forward(h, batch.edge_index, edge_emb, batch.batch)
        logits = self.pretrain_head(h)
        return logits

    def forward_features(self, batch):
        h = self._embed_nodes(batch.x)
        edge_emb = self._embed_edges(batch.edge_attr)
        h = self._gnn_forward(h, batch.edge_index, edge_emb, batch.batch)
        g = global_add_pool(h, batch.batch)
        return g


# =========================================================
# MODEL: Decoder (GPT-2 Bridge)
# =========================================================

class Graph2CaptionV2(nn.Module):
    def __init__(self, pretrained_encoder, gpt2_model_name="gpt2"):
        super().__init__()
        self.encoder = pretrained_encoder
        gnn_hidden_dim = 128
        
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        gpt_emb_size = self.gpt2.config.n_embd 

        self.projection = nn.Linear(gnn_hidden_dim, gpt_emb_size)

    def forward(self, data, text_input_ids, text_attention_mask):
        graph_vec = self.encoder.forward_features(data)  
        projected_emb = self.projection(graph_vec).unsqueeze(1) 
        
        text_embeds = self.gpt2.transformer.wte(text_input_ids)
        inputs_embeds = torch.cat((projected_emb, text_embeds), dim=1)

        batch_size = text_attention_mask.shape[0]
        ones = torch.ones((batch_size, 1), device=text_attention_mask.device)
        extended_mask = torch.cat((ones, text_attention_mask), dim=1)

        return self.gpt2(
            inputs_embeds=inputs_embeds, attention_mask=extended_mask
        ).logits

    def generate_caption(self, data, max_length=100):
        self.eval()
        with torch.no_grad():
            graph_vec = self.encoder.forward_features(data)
            cur_input_embeds = self.projection(graph_vec).unsqueeze(1)
            generated_ids = []

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
# Pre-Training Functions
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

@torch.no_grad()
def eval_epoch_pretrain(mol_enc, loader, device):
    mol_enc.eval() 
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
            
            bs = graphs.num_graphs
            total_loss += loss.item() * bs
            total += bs

    return total_loss / total if total > 0 else 0.0

# =========================================================
# Main Pipeline
# =========================================================

def main():
    os.makedirs("checkpoints", exist_ok=True)
    
# # --- Phase 1: Pre-Training (Masking) ---
#     print("\n=== Phase 1: Pre-Training GNN ===")

#     mol_enc = MolGNN(hidden=128).to(DEVICE)
    
#     train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, None)
#     train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
#     val_ds = PreprocessedGraphDataset(VAL_GRAPHS, None) 
#     val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
#     pretrain_optimizer = torch.optim.Adam(mol_enc.parameters(), lr=PRETRAIN_LR)

#     best_val_loss = float('inf')

#     for ep in range(PRETRAIN_EPOCHS):
#         train_loss = train_epoch_pretrain(mol_enc, train_dl, pretrain_optimizer, DEVICE)
        
#         val_loss = eval_epoch_pretrain(mol_enc, val_dl, DEVICE)
        
#         print(f"Pretrain Epoch {ep+1}/{PRETRAIN_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(mol_enc.state_dict(), "checkpoints/mol_enc_pretrained_best.pt")
#             print(f"  -> New best model saved (Val Loss: {val_loss:.4f})")

#     print("\nPre-training complete.")
#     print(f"Best Val Loss achieved: {best_val_loss:.4f}")
    
#     print("Loading best pre-trained weights for Phase 2...")
#     mol_enc.load_state_dict(torch.load("checkpoints/mol_enc_pretrained_best.pt", map_location=DEVICE))
    

    # --- Phase 2: Captioning Fine-Tuning ---
    print("\n=== Phase 2: Training Graph2Caption ===")

    mol_enc = MolGNN(hidden=128).to(DEVICE)
    state_dict = torch.load("checkpoints/mol_enc_pretrained_best.pt", map_location=DEVICE)
    mol_enc.load_state_dict(state_dict)
    
    model = Graph2CaptionV2(pretrained_encoder=mol_enc, gpt2_model_name="gpt2")
    model.to(DEVICE)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENCODER_LR},     
        {"params": model.projection.parameters(), "lr": PROJECTION_LR},   
        {"params": model.gpt2.parameters(), "lr": GPT_LR},        
    ])

    train_ds_cap = PreprocessedGraphDataset(TRAIN_GRAPHS, None)
    train_loader = DataLoader(train_ds_cap, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Caption Epoch {epoch+1}")

        for batch in pbar:
            batch = batch.to(DEVICE)
            captions = batch.description 

            inputs = model.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=100,
                return_tensors="pt",
            ).to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch, inputs.input_ids, inputs.attention_mask)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs.input_ids.contiguous() 

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f"checkpoints/g2cap_epoch_{epoch+1}.pt")
    
    print("Training Complete.")

if __name__ == "__main__":
    main()