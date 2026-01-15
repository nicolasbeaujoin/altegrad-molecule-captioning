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


## Method of the section "Improved retrieval method" from the report.

The file retrieval_method.py basically contains everything used in this method.

- The function pyg_to_rdkit reconstructs an RDKit Molecule object from the molecule graphs that we have in the datasets.

- The MolGNN class contains the architecture of the graph encoder, which is explained in the report.

- The ContrastiveModel class contains the whole model (graph encoder and pre-trained text encoder) with a forward method that 
outputs a graph embedding and a text embedding for a given molecule.

- The functions train_contrastive and validate_contrastive are used to respectively perform training and validation of a ContrastiveModel.

- The run_retrieval_pipeline function is used to make the retrieval of the captions of the molecules in the test set.