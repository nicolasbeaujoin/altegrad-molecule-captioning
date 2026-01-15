# ALTeGraD 25-26: Molecular Graph Captioning

By Vadim Lagresle, Nicolas Beaujoin, Justin Bec.

This repository implements several methods to perform molecule captioning, either by retrieval or by generation.

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

Place your preprocessed graph data files in the `data/` directory:
- `train_graphs.pkl`
- `validation_graphs.pkl`
- `test_graphs.pkl`

## I. Usage of the baseline method

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

### 5. Output

- `model_checkpoint.pt`: Trained GCN model
- `test_retrieved_descriptions.csv`: Retrieved descriptions for test set


## II. Method of the "Generative model" section of the report.

The file pretrain_gpt_clean.py contains everything used in this method. 

- The class MolGNN is a GINE-based graph encoder that outputs a latent representation of the graph molecule.

- The Graph2CaptionV2 is the main model that contains the MolGNN graph encoder and the pre-trained GPT2 decoder that outputs the captions.

- The functions mask_atoms_for_pretraining, train_epoch_pretrain and eval_epoch_pretrain are used for the self-supervised training phase.

- The training of the main model is then performed in the function main().




## III. Method of the section "Improved retrieval method" from the report.

The file retrieval_method.py basically contains everything used in this method.

- The function pyg_to_rdkit reconstructs an RDKit Molecule object from the molecule graphs that we have in the datasets.

- The MolGNN class contains the architecture of the graph encoder, which is explained in the report.

- The ContrastiveModel class contains the whole model (graph encoder and pre-trained text encoder) with a forward method that 
outputs a graph embedding and a text embedding for a given molecule.

- The functions train_contrastive and validate_contrastive are used to respectively perform training and validation of a ContrastiveModel.

- The run_retrieval_pipeline function is used to make the retrieval of the captions of the molecules in the test set.


## IV. Tri-Modal Retrieval-Augmented Generation Architecture

The script RAG.py implements our final architecture, bridging graph representation learning with retrieval-augmented generation (RAG) to produce chemically accurate captions. The pipeline consists of the following components:

1.⁠ ⁠Data Processing & Retrieval

- pyg_to_rdkit & compute_fingerprint: Utility functions that reconstruct RDKit molecule objects from PyTorch Geometric graphs and calculate Morgan Fingerprints (2048 bits).

- build_neighbor_map: Implements the retrieval engine. It computes Tanimoto similarity across the dataset to identify the nearest structural neighbor for every molecule. The description of this neighbor is extracted to serve as a "context prompt."

- RAGDataset: A custom dataset class that bundles the three required inputs for the model: the molecular graph, the SMILES string, and the retrieved neighbor's description.

2.⁠ ⁠Neural Architecture

- HybridMolGINE: The visual encoder. It replaces standard GCNs with GINE (Graph Isomorphism Network) layers to explicitly aggregate edge features (bond types) with node features, capturing precise chemical topology. It fuses these graph features with global fingerprint embeddings.

- RAGMolT5: The core multi-modal model. It uses a frozen MolT5 Encoder to process the textual prompt and fuses it with the GINE output. This combined representation is passed to a trainable MolT5 Decoder to generate the final caption.

3.⁠ ⁠Training Loop

- train_loop: Manages the optimization process using Cross-Entropy Loss. It includes Gradient Clipping (to stabilize the Transformer) and a Learning Rate Scheduler (ReduceLROnPlateau).
