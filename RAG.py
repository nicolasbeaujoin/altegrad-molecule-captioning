import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMÈTRES
# ==============================================================================
# Chemins des données
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS = "data/validation_graphs.pkl"
TEST_GRAPHS = "data/test_graphs.pkl"

# Configuration Matérielle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamètres Modèle
HIDDEN_DIM = 256
GNN_LAYERS = 4
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 15

# --- OPTION CRITIQUE : RELOAD ---
# True  = Charge "best_rag_model.pt" et génère le CSV (Pas d'entrainement)
# False = Lance l'entraînement complet (GINE + MolT5)
RELOAD_MODEL = False
MODEL_PATH = "best_rag_model.pt"

print(f"Running RAG Architecture (GINE + MolT5 + Neighbor Retrieval) on {DEVICE}")
print(f"Mode: {'RELOAD & INFERENCE' if RELOAD_MODEL else 'TRAINING FROM SCRATCH'}")

# ==============================================================================
# 2. OUTILS CHIMIQUES & RETRIEVAL (TANIMOTO)
# ==============================================================================
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def pyg_to_rdkit(data):
    """Reconstruction robuste RDKit depuis PyG avec gestion des valences"""
    mol = Chem.RWMol()
    atomic_nums = data.x[:, 0].cpu().numpy()
    for atomic_num in atomic_nums:
        mol.AddAtom(Chem.Atom(int(atomic_num)))

    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()

    for i in range(0, edge_index.shape[1], 2):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < v:
            b_type_idx = int(edge_attr[i, 0])
            rdkit_b_type = bond_types.get(b_type_idx, Chem.BondType.SINGLE)
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


def compute_fingerprint(graph):
    """Calcule le fingerprint pour un graphe PyG (Safe Mode)"""
    try:
        mol = pyg_to_rdkit(graph)
        return mfpgen.GetFingerprint(mol)
    except:
        # Fallback sur molécule vide pour ne pas crasher
        return mfpgen.GetFingerprint(Chem.MolFromSmiles(""))


# --- MOTEUR DE RETRIEVAL (PRE-CALCUL) ---
def build_neighbor_map(target_graphs, source_graphs, is_train_set=False):
    """
    Construit un dictionnaire {ID_Target -> Description_Source}
    Utilise Tanimoto Similarity sur les Fingerprints.
    """
    print("Building Fingerprint Index...")
    target_fps = [
        compute_fingerprint(g) for g in tqdm(target_graphs, desc="Target FPs")
    ]
    source_fps = [
        compute_fingerprint(g) for g in tqdm(source_graphs, desc="Source FPs")
    ]

    # Mapping ID -> Description pour la source
    source_desc_map = {g.id: g.description for g in source_graphs}

    neighbor_map = {}

    print("Searching Nearest Neighbors...")
    for i, target_fp in enumerate(tqdm(target_fps)):
        # BulkTanimoto est optimisé en C++ dans RDKit
        sims = DataStructs.BulkTanimotoSimilarity(target_fp, source_fps)

        best_sim = -1.0
        best_idx = -1

        # Si on compare Train vs Train, on doit exclure la molécule elle-même
        if is_train_set:
            if i < len(sims):
                sims[i] = -1.0  # On s'interdit de se choisir soi-même
            best_sim = max(sims)
            best_idx = sims.index(best_sim)
        else:
            # Test vs Train : on prend le max absolu
            best_sim = max(sims)
            best_idx = sims.index(best_sim)

        target_id = target_graphs[i].id
        neighbor_id = source_graphs[best_idx].id
        neighbor_desc = source_desc_map[neighbor_id]

        neighbor_map[target_id] = neighbor_desc

    return neighbor_map


# ==============================================================================
# 3. DATASET RAG
# ==============================================================================
class RAGDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, neighbor_map):
        self.graphs = graphs
        self.neighbor_map = neighbor_map

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]

        # 1. Fingerprints & SMILES
        try:
            mol = pyg_to_rdkit(data)
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            data.fp = torch.tensor(fp, dtype=torch.float32)
            data.smiles = Chem.MolToSmiles(mol)
        except:
            data.fp = torch.zeros(2048, dtype=torch.float32)
            data.smiles = ""

        # 2. Voisin RAG (Contexte)
        data.neighbor_desc = self.neighbor_map.get(data.id, "")

        return data


# ==============================================================================
# 4. MODÈLES (GINE + MolT5)
# ==============================================================================
class HybridMolGINE(nn.Module):
    """
    Encodeur Graphe utilisant GINE (Graph Isomorphism Network + Edge Features).
    """

    def __init__(self, hidden=256, fp_dim=2048, layers=4):
        super().__init__()
        # Embeddings
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
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))

        # Fingerprint Branch
        self.fp_mlp = nn.Sequential(
            nn.Linear(fp_dim, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.fusion_proj = nn.Linear(hidden * 2, hidden)

    def forward(self, data):
        x = sum([emb(data.x[:, i]) for i, emb in enumerate(self.node_embeds)])
        edge_attr = sum(
            [emb(data.edge_attr[:, i]) for i, emb in enumerate(self.edge_embeds)]
        )

        for conv, bn in zip(self.convs, self.bns):
            x_in = x
            x = F.relu(bn(conv(x, data.edge_index, edge_attr=edge_attr)))
            x = x + x_in  # Residual

        g_vec = global_mean_pool(x, data.batch)

        fp_input = data.fp.view(data.num_graphs, -1)
        fp_vec = self.fp_mlp(fp_input)

        return self.fusion_proj(torch.cat([g_vec, fp_vec], dim=1))


class RAGMolT5(nn.Module):
    def __init__(self, gnn_model, molt5_name="laituan245/molt5-base"):
        super().__init__()
        print(f"Building RAG Model (GINE + T5)...")
        self.gnn = gnn_model

        # Encodeur T5 (Gelé)
        self.smiles_encoder = T5EncoderModel.from_pretrained(molt5_name)
        self.tokenizer = AutoTokenizer.from_pretrained(molt5_name)
        for param in self.smiles_encoder.parameters():
            param.requires_grad = False

        text_dim = self.smiles_encoder.config.d_model
        self.text_proj = nn.Linear(text_dim, text_dim)

        # Decodeur T5 (Entraîné)
        self.t5_full = T5ForConditionalGeneration.from_pretrained(molt5_name)
        del self.t5_full.encoder
        torch.cuda.empty_cache()

        # Fusion
        decoder_dim = self.t5_full.config.d_model
        combined_dim = HIDDEN_DIM + text_dim

        self.fusion_proj = nn.Sequential(
            nn.Linear(combined_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    def prepare_rag_inputs(self, data):
        """Prompt Engineering pour le RAG"""
        prompts = []
        smiles_list = data.smiles
        neighbor_list = data.neighbor_desc

        for smi, ctx in zip(smiles_list, neighbor_list):
            if len(ctx) > 5:
                # Troncation de sécurité pour le contexte (max 300 chars pour laisser place au reste)
                ctx_short = ctx[:300]
                text = f"SMILES: {smi} . SIMILAR: {ctx_short}"
            else:
                text = f"SMILES: {smi}"
            prompts.append(text)

        tokens = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        return tokens

    def forward(self, data, caption_ids, attention_mask=None):
        # 1. Vision
        graph_vec = self.gnn(data)

        # 2. Texte RAG
        text_inputs = self.prepare_rag_inputs(data)
        with torch.no_grad():
            t_out = self.smiles_encoder(**text_inputs)
            mask = (
                text_inputs.attention_mask.unsqueeze(-1)
                .expand(t_out.last_hidden_state.size())
                .float()
            )
            text_vec = torch.sum(t_out.last_hidden_state * mask, 1) / torch.clamp(
                mask.sum(1), min=1e-9
            )

        text_vec = self.text_proj(text_vec)

        # 3. Fusion
        combined = torch.cat([graph_vec, text_vec], dim=1)
        final_emb = self.fusion_proj(combined).unsqueeze(1)

        # 4. Decode
        outputs = self.t5_full(
            inputs_embeds=None,
            encoder_outputs=(final_emb,),
            labels=caption_ids,
            decoder_attention_mask=attention_mask,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, data, max_length=128):
        graph_vec = self.gnn(data)

        text_inputs = self.prepare_rag_inputs(data)
        t_out = self.smiles_encoder(**text_inputs)
        mask = (
            text_inputs.attention_mask.unsqueeze(-1)
            .expand(t_out.last_hidden_state.size())
            .float()
        )
        text_vec = torch.sum(t_out.last_hidden_state * mask, 1) / torch.clamp(
            mask.sum(1), min=1e-9
        )
        text_vec = self.text_proj(text_vec)

        combined = torch.cat([graph_vec, text_vec], dim=1)
        final_emb = self.fusion_proj(combined).unsqueeze(1)

        # FIX VITAL POUR HUGGINGFACE
        encoder_outputs = BaseModelOutput(last_hidden_state=final_emb)

        generated_ids = self.t5_full.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ==============================================================================
# 5. BOUCLE D'ENTRAÎNEMENT (Stable Float32 + Clip)
# ==============================================================================
def train_loop(model, train_loader, val_loader, optimizer, epochs=10):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    history = {"train": [], "val": []}
    best_loss = float("inf")
    early_stop = 0

    print(f"Starting RAG Training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 0

        for data in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            torch.cuda.empty_cache()
            data = data.to(DEVICE)

            targets = model.tokenizer(
                data.description,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(DEVICE)

            labels = targets.input_ids
            labels[labels == model.tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

            # Pas d'Autocast = Stabilité Max
            loss = model(
                data, caption_ids=labels, attention_mask=targets.attention_mask
            )

            if torch.isnan(loss):
                print("Warning: NaN Loss detected, skipping batch")
                continue

            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_train = total_loss / max(steps, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Ep {epoch+1} [Val]  "):
                data = data.to(DEVICE)
                targets = model.tokenizer(
                    data.description,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(DEVICE)
                labels = targets.input_ids
                labels[labels == model.tokenizer.pad_token_id] = -100

                loss = model(
                    data, caption_ids=labels, attention_mask=targets.attention_mask
                )
                val_loss += loss.item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        scheduler.step(avg_val)

        print(
            f"Ep {epoch+1}: Train={avg_train:.4f} | Val={avg_val:.4f} | LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        if avg_val < best_loss:
            best_loss = avg_val
            early_stop = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved Best Model!")
        else:
            early_stop += 1
            if early_stop >= 3:  # ²nce de 3 époques
                print("Early Stopping")
                break
    return history


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # A. CHARGEMENT
    print("[1/5] Loading Raw Graphs...")
    with open(TRAIN_GRAPHS, "rb") as f:
        train_graphs = pickle.load(f)
    with open(VAL_GRAPHS, "rb") as f:
        val_graphs = pickle.load(f)
    with open(TEST_GRAPHS, "rb") as f:
        test_graphs = pickle.load(f)

    # B. RAG INDEXING
    if not RELOAD_MODEL:
        print("[2/5] Building RAG Index (Train/Val)...")
        train_map = build_neighbor_map(train_graphs, train_graphs, is_train_set=True)
        val_map = build_neighbor_map(val_graphs, train_graphs, is_train_set=False)
    else:
        print("[2/5] Skipping Train Index (Reload Mode)")
        train_map = {}
        val_map = {}

    print("  -> Building Test Index (Always required)...")
    test_map = build_neighbor_map(test_graphs, train_graphs, is_train_set=False)

    # C. DATASETS
    print("[3/5] Creating RAG Datasets...")
    ds_test = RAGDataset(test_graphs, test_map)
    loader_test = DataLoader(
        ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    if not RELOAD_MODEL:
        ds_train = RAGDataset(train_graphs, train_map)
        ds_val = RAGDataset(val_graphs, val_map)
        loader_train = DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        loader_val = DataLoader(
            ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

    # D. INIT MODEL
    print("[4/5] Initializing GINE + RAGMolT5...")
    gnn = HybridMolGINE(hidden=HIDDEN_DIM, fp_dim=2048, layers=GNN_LAYERS)
    model = RAGMolT5(gnn, molt5_name="laituan245/molt5-base").to(DEVICE)

    # E. TRAIN / RELOAD
    if RELOAD_MODEL:
        print(f"Reloading weights from {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model loaded!")
        else:
            print("Model not found for reload!")
            exit()
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        train_loop(model, loader_train, loader_val, optimizer, epochs=EPOCHS)

    # F. PREDICT
    print("[5/5] Generating Submission...")
    model.eval()
    results = []

    for data in tqdm(loader_test):
        data = data.to(DEVICE)
        texts = model.generate(data)
        for i, txt in enumerate(texts):
            results.append({"ID": data.id[i], "description": txt})

    df = pd.DataFrame(results)
    df.to_csv("submission_rag_gine.csv", index=False)
    print("Done! Saved to submission_rag_gine.csv")
