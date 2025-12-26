import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_scatter import scatter_softmax

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, MessagePassing
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.utils import to_dense_batch
from graph2caption import *
from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import argparse


# =========================================================
# CONFIG
# =========================================================
# Data paths
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"

TRAIN_EMB_CSV = "data/train_embeddings.csv"
VAL_EMB_CSV = "data/validation_embeddings.csv"

# Pre-Training parameters
MASK_RATE = 0.15
MASK_TOKEN_ID = 0
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3

# Training parameters
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 2
EPOCHS_PHASE_2 = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Pre-Training functions
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


def pretrain(
    hidden_dim, train_graphs, pretrain_epochs, batch_size, pretrain_lr, device
):
    """
    Pretrain the encoder part of the Graph2CaptionV2 model.
    """
    # --- Phase 1: Pre-Training (Masking) ---
    print("\n=== Phase 1: Pre-Training GNN ===")

    # Initialize basic GNN (hidden=512)
    mol_enc = MolGNN(hidden=hidden_dim).to(device)

    # Load data for pre-training (We only need graphs, not text yet)
    # Note: Pass 'None' for embeddings if you don't use them in pre-training
    train_ds = PreprocessedGraphDataset(train_graphs, None)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    pretrain_optimizer = torch.optim.Adam(mol_enc.parameters(), lr=pretrain_lr)

    for ep in range(pretrain_epochs):
        loss = train_epoch_pretrain(mol_enc, train_dl, pretrain_optimizer, device)
        print(f"Pretrain Epoch {ep+1}: {loss:.4f}")

    print("Pre-training complete. Saving GNN weights.")
    torch.save(mol_enc.state_dict(), "checkpoints/mol_enc_best.pt")


# =========================================================
# Training function
# =========================================================


def train(hidden_dim, train_graphs, epochs_phase1, epochs_phase2, device):
    # --- Phase 2: Captioning Fine-Tuning ---
    print("\n=== Phase 2: Training Graph2Caption ===")

    # 1. Initialize Combined Model using the PRE-TRAINED encoder
    mol_enc = MolGNN(hidden=hidden_dim).to(device)
    state_dict = torch.load("checkpoints/mol_enc_best.pt", map_location=device)
    mol_enc.load_state_dict(state_dict)
    mol_enc.to(device)

    model = Graph2CaptionV2(
        pretrained_encoder=mol_enc, gpt2_model_name="gpt2-medium", num_graph_tokens=8
    )
    model.to(device)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # state_dict2 = torch.load("checkpoints/g2cap_epoch_9.pt", map_location=DEVICE)
    # model.load_state_dict(state_dict2)
    # model.to(DEVICE)

    for param in model.gpt2.parameters():
        param.requires_grad = False

    optimizer_phase1 = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": model.projector.parameters(), "lr": 1e-3},
        ]
    )

    # 3. Load Data with Descriptions (Re-load to ensure we access descriptions)
    # Ensure your PreprocessedGraphDataset loads data.description!
    train_ds_cap = PreprocessedGraphDataset(train_graphs, None)
    train_loader = DataLoader(
        train_ds_cap, batch_size=64, shuffle=True
    )  # Smaller batch for GPT

    model.train()
    # criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs_phase1):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Caption Epoch {epoch+1}")

        for batch in pbar:
            batch = batch.to(device)
            captions = batch.description  # Ensure this exists in your dataset class

            # 1. Tokenize text
            inputs = model.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            batch_size = input_ids.size(0)

            # 1. Forward pass (returns logits for [Graph_Tokens + Text_Tokens])
            logits = model(batch, inputs.input_ids, inputs.attention_mask)

            # 2. Create the target labels tensor
            # Initialize everything to -100 (which the criterion will now ignore)
            labels = torch.full(logits.shape[:2], -100, dtype=torch.long, device=device)

            # 3. Fill the text portion
            # Logic: We want the logit at index 'i' to predict the token at 'i+1'
            # The text starts at index 'num_graph_tokens' (e.g., 8)
            num_g = model.num_graph_tokens
            labels[:, num_g:-1] = inputs.input_ids[:, 1:]

            # 4. CRITICAL: Also mask the tokenizer's own padding tokens
            # If the original input was a pad token, we shouldn't calculate loss for it
            # Note: we shift the mask to match the shifted labels
            text_mask = inputs.attention_mask[:, 1:]
            labels[:, num_g:-1][text_mask == 0] = -100

            # 5. Shift Logits and Labels for Causal LM
            # Remove the very last logit (nothing to predict) and
            # the very first label (the graph token itself doesn't have a 'previous' word to predict it)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # 6. Compute Loss
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            # Debugging check
            valid_mask = shift_labels != -100
            if valid_mask.any():
                max_label = shift_labels[valid_mask].max().item()
                min_label = shift_labels[valid_mask].min().item()
                vocab_size = model.gpt2.config.vocab_size

                if max_label >= vocab_size or min_label < 0:
                    print(
                        f"CRITICAL ERROR: Label {max_label} is out of bounds for vocab size {vocab_size}"
                    )
            loss.backward()
            optimizer_phase1.step()
            optimizer_phase1.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(
            f"Phase 1, Epoch {epoch+1}/{epochs_phase1} - Average Loss: {total_loss / len(train_loader):.4f}"
        )
        # Save Checkpoint
        # torch.save(model.state_dict(), f"checkpoints/g2cap_epoch_{epoch+10}.pt")

    for param in model.gpt2.parameters():
        param.requires_grad = True

    optimizer_phase2 = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 5e-5},
            {"params": model.projector.parameters(), "lr": 1e-4},
            {"params": model.gpt2.parameters(), "lr": 1e-5},  # Extremely low LR
        ]
    )

    for epoch in range(epochs_phase2):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Caption Epoch {epoch+1}")

        for batch in pbar:
            batch = batch.to(device)
            captions = batch.description  # Ensure this exists in your dataset class

            # 1. Tokenize text
            inputs = model.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(DEVICE)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            batch_size = input_ids.size(0)

            # 1. Forward pass (returns logits for [Graph_Tokens + Text_Tokens])
            logits = model(batch, inputs.input_ids, inputs.attention_mask)

            # 2. Create the target labels tensor
            # Initialize everything to -100 (which the criterion will now ignore)
            labels = torch.full(logits.shape[:2], -100, dtype=torch.long, device=device)

            # 3. Fill the text portion
            # Logic: We want the logit at index 'i' to predict the token at 'i+1'
            # The text starts at index 'num_graph_tokens' (e.g., 8)
            num_g = model.num_graph_tokens
            labels[:, num_g:-1] = inputs.input_ids[:, 1:]

            # 4. CRITICAL: Also mask the tokenizer's own padding tokens
            # If the original input was a pad token, we shouldn't calculate loss for it
            # Note: we shift the mask to match the shifted labels
            text_mask = inputs.attention_mask[:, 1:]
            labels[:, num_g:-1][text_mask == 0] = -100

            # 5. Shift Logits and Labels for Causal LM
            # Remove the very last logit (nothing to predict) and
            # the very first label (the graph token itself doesn't have a 'previous' word to predict it)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # 6. Compute Loss
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            # Debugging check
            valid_mask = shift_labels != -100
            if valid_mask.any():
                max_label = shift_labels[valid_mask].max().item()
                min_label = shift_labels[valid_mask].min().item()
                vocab_size = model.gpt2.config.vocab_size

                if max_label >= vocab_size or min_label < 0:
                    print(
                        f"CRITICAL ERROR: Label {max_label} is out of bounds for vocab size {vocab_size}"
                    )
            loss.backward()
            optimizer_phase2.step()
            optimizer_phase2.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(
            f"Phase 2, Epoch {epoch+1}/{epochs_phase2} - Average Loss: {total_loss / len(train_loader):.4f}"
        )
        # Save Checkpoint
        torch.save(
            model.state_dict(), f"checkpoints/g2cap_epoch_{epoch+epochs_phase1}.pt"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="training")

    args = parser.parse_args()

    if args.train_type == "pretraining":
        pretrain(
            hidden_dim=512,
            train_graphs=TRAIN_GRAPHS,
            pretrain_epochs=PRETRAIN_EPOCHS,
            batch_size=BATCH_SIZE,
            pretrain_lr=PRETRAIN_LR,
            device=DEVICE,
        )
    elif args.train_type == "training":
        train(
            hidden_dim=512,
            train_graphs=TRAIN_GRAPHS,
            epochs_phase1=EPOCHS_PHASE_1,
            epochs_phase2=EPOCHS_PHASE_2,
            device=DEVICE,
        )
