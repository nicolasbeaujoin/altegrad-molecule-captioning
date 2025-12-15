import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
from torch.cuda.amp import autocast, GradScaler

# Import your model definition and dataset class
# Assuming the class from previous step is in 'model.py'
# and provided utils are in 'data_utils.py'
from model import Graph2Caption
from data_utils import PreprocessedGraphDataset

# --- 1. Hyperparameters & Setup ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Lower LR is better for fine-tuning pre-trained models
EPOCHS = 10
MAX_TEXT_LEN = 128  # Cap captions to save memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Prepare Data ---
print(f"Loading data on {DEVICE}...")
# The challenge provides 'train_graphs.pkl' with descriptions [cite: 20, 31]
train_dataset = PreprocessedGraphDataset("data/train_graphs.pkl")

# PyG DataLoader handles the graph batching (creating large block-diagonal matrices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Initialize Model & Tokenizer ---
model = Graph2Caption(gpt2_model_name="gpt2-medium")  # Use medium as advised
model.to(DEVICE)

# Vital for GPT-2: Set padding token
model.tokenizer.pad_token = model.tokenizer.eos_token

# Optimizer: AdamW is standard for Transformers
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Gradient scaler for mixed precision training
scaler = GradScaler()

# --- 4. Training Loop ---
print("Starting training...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch_idx, data in enumerate(progress_bar):
        # Move graph data to GPU
        data = data.to(DEVICE)

        # --- A. Process Text Targets ---
        # data.description is a list of strings [cite: 40]
        captions = data.description

        # Tokenize the batch of captions
        # return_tensors='pt' gives us PyTorch tensors
        # padding=True pads to the longest in the batch
        # truncation=True ensures we don't blow up memory
        tokenized = model.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].to(DEVICE)
        attention_mask = tokenized["attention_mask"].to(DEVICE)

        # --- B. Forward Pass ---
        with autocast():
            optimizer.zero_grad()

            # Pass graph + text indices to the model
            # The model embeds the graph, embeds the text, and concatenates them
            # Output shape: [batch_size, 1 + seq_len, vocab_size]
            logits = model(data, input_ids, attention_mask)

            # --- C. Calculate Loss (Shifted) ---
            # We need to predict the NEXT token.

            # 1. Targets: The actual text tokens (input_ids).
            # We don't predict the graph, we predict Token1 given Graph, Token2 given Token1...
            labels = input_ids.clone()

            # 2. Logits: The model outputs.
            # The input to the transformer was: [Graph_Emb, T1, T2, ..., Tn]
            # We want the prediction from Graph_Emb to match T1.
            # We want the prediction from T1 to match T2.
            # The prediction from Tn matches nothing (or EOS).

            # Slice logits to remove the last prediction (which has no target)
            # Shift logits: [batch, seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()

            # Slice labels to match.
            # Labels are simply input_ids.
            # Shift labels: [batch, seq_len]
            shift_labels = labels.contiguous()

            # 3. Compute Cross Entropy
            # Flatten tensors for CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # --- D. Backward Pass ---
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- E. Logging ---
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

    # Save checkpoint every epoch
    torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch+1}.pt")

print("Training finished.")
