# Molecule-Text Retrieval

Graph neural network for molecular graph and text description retrieval.

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

Place your preprocessed graph data files in the `data/` directory:
- `train_graphs.pkl`
- `validation_graphs.pkl`
- `test_graphs.pkl`

## Usage

Run the following scripts in order:

### 1. Inspect Graph Data

Check the structure and contents of your graph files:

```bash
python inspect_graph_data.py
```

### 2. Generate Description Embeddings

Create BERT embeddings for molecular descriptions:

```bash
python generate_description_embeddings.py
```

This generates:
- `data/train_embeddings.csv`
- `data/validation_embeddings.csv`

### 3. Train GCN Model

Train the graph neural network:

```bash
python train_gcn.py
```

This creates `model_checkpoint.pt`.

### 4. Run Retrieval

Retrieve descriptions for test molecules:

```bash
python retrieval_answer.py
```

This generates `test_retrieved_descriptions.csv` with retrieved descriptions for each test molecule.

## Output

- `model_checkpoint.pt`: Trained GCN model
- `test_retrieved_descriptions.csv`: Retrieved descriptions for test set


## Logic of the new model and training proposed here

1. Model Architecture: The "Visual Prompt" Logic
The model treats the molecular graph not as a string (SMILES), but as a set of visual features that "prompt" a Large Language Model (LLM).

Encoder (MolGNN): * Input: 9 node features (atomic num, chirality, etc.) and 3 edge features (bond type, stereo, etc.).
- Structure: 6-layer GINEConv with Residual Connections and a Hidden Dimension of 256.
- Output: A dense feature map of every atom in the molecule.

Bridge (MultiTokenProjector):
- Mechanism: Uses 8 learnable latent queries and Cross-Attention to "scan" the graph features.
- Goal: Instead of shrinking the molecule into one vector, it creates 8 "Graph Tokens" that represent different structural motifs (rings, functional groups).

Decoder (GPT-2 Medium):
- Mechanism: Receives a sequence: [8 Graph Tokens] + [Text Tokens].
- Generation: It uses the graph tokens as context to predict the next word in the description.

2. Phase 0: Pre-training (Masked Atom Modeling)
Before looking at text, the GNN must understand chemistry.

- The Task: We hide (mask) 15% of the atoms in a molecule and ask the GNN to predict what they were based on their neighbors.
- Logic: This forces the GNN to learn "chemical grammar"—for example, that an Oxygen atom is likely to be near a Carbon in a carboxyl group.
- Result: A "warm" encoder that already understands molecular topology.

3. Phase 1: Alignment (Freezing GPT-2)
Connecting a new GNN to a pre-trained LLM is difficult because the initial GNN outputs are "gibberish" to GPT-2.

- Setup: Freeze all GPT-2 weights. Only train the MolGNN and the Projector.
- Logic: We keep the "brain" (GPT-2) fixed and force the "eyes" (GNN) to adapt its output until the vectors fall into a range that GPT-2 recognizes as meaningful semantic concepts.
- Goal: Prevent the random initial gradients from the GNN from destroying GPT-2's pre-trained English language knowledge.

4. Phase 2: Joint Fine-tuning (Unfreezing)
Once the modalities are aligned, we refine the entire system.

- Setup: Unfreeze everything.
- Learning Rates: This is the most critical part: GNN/Projector: moderate lr, GPT-2: very low lr.
- Logic: We allow GPT-2 to slightly adjust its vocabulary and internal logic to better suit the specific domain of "Chemical English," while the GNN continues to sharpen its feature extraction.