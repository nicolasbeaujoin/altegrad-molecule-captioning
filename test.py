import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from data_utils import PreprocessedGraphDataset  # Provided in challenge files

# from graph2caption import Graph2Caption
from train_all_pretrain_gpt import Graph2CaptionV2, MolGNN

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

mol_enc = MolGNN(hidden=128).to(device)
state_dict1 = torch.load("checkpoints/mol_enc.pt", map_location=device)
mol_enc.load_state_dict(state_dict1)
mol_enc.to(device)

# Instantiate the model with the same configuration used during training
model = Graph2CaptionV2(pretrained_encoder=mol_enc, gpt2_model_name="gpt2-medium")

# Ensure padding token is set (as in training)
model.tokenizer.pad_token = model.tokenizer.eos_token

# Load trained weights (state_dict) and move to device
state_dict2 = torch.load("checkpoints/g2cap_epoch_2.pt", map_location=device)
model.load_state_dict(state_dict2)
model.to(device)
model.eval()  # Set to evaluation mode (turns off dropout, etc.)

# 2. Load Test Data
# Note: The document states test_graphs.pkl has no descriptions [cite: 23]
test_dataset = PreprocessedGraphDataset("data/test_graphs.pkl")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = []

# 3. Inference Loop
print("Generating captions for test set...")
with torch.no_grad():
    for data in tqdm(test_loader):
        data = data.to(device)

        # A. Generate Caption
        # Uses the generate_caption method defined in the previous step
        # You can adjust max_length based on your validation set statistics
        generated_text = model.generate_caption(data, max_length=100)

        # B. Store Result
        # The document requires aligning graphs using data.id [cite: 39]
        predictions.append(
            {
                "ID": data.id[0],  # data.id is a list/string in the batch
                "description": generated_text,
            }
        )

# 4. Create Submission File
# Format must be a CSV with "ID" and "description" columns
df = pd.DataFrame(predictions)
df.to_csv("submission4.csv", index=False)

print(f"Success! Generated predictions for {len(df)} molecules.")
print("Upload 'submission.csv' to Kaggle to get your BLEU/BERTScore.")
