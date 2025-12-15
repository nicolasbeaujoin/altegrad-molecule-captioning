import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class MoleculeEncoder(nn.Module):
    """
    Encodes the molecule graph using the specific features from the challenge data.
    """

    def __init__(self, hidden_dim=384, num_layers=3):
        super().__init__()

        # 1. Embedding layers for the 9 specific node features [cite: 45-55]
        # Dimensions based on indices in data_utils.py (approximate max values)
        self.emb_atomic_num = nn.Embedding(119, hidden_dim)
        self.emb_chirality = nn.Embedding(10, hidden_dim)
        self.emb_degree = nn.Embedding(11, hidden_dim)
        self.emb_formal_charge = nn.Embedding(12, hidden_dim)
        self.emb_num_hs = nn.Embedding(9, hidden_dim)
        self.emb_radicals = nn.Embedding(5, hidden_dim)
        self.emb_hybridization = nn.Embedding(8, hidden_dim)
        self.emb_aromatic = nn.Embedding(2, hidden_dim)
        self.emb_ring = nn.Embedding(2, hidden_dim)

        # 2. Embedding layers for the 3 specific edge features [cite: 62-65]
        self.emb_bond_type = nn.Embedding(23, hidden_dim)
        self.emb_stereo = nn.Embedding(7, hidden_dim)
        self.emb_conjugated = nn.Embedding(2, hidden_dim)

        # 3. GNN Layers (Using GINEConv as it supports edge features well)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            # MLP for the GINEConv aggregation
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.gnn_layers.append(GINEConv(mlp, edge_dim=hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        # A. Embed Node Features
        # x shape: [num_atoms, 9]. We sum embeddings (simple fusion) or concat
        # Note: x[:, 0] is atomic_num, x[:, 1] is chirality, etc.
        node_embed = (
            self.emb_atomic_num(x[:, 0])
            + self.emb_chirality(x[:, 1])
            + self.emb_degree(x[:, 2])
            + self.emb_formal_charge(x[:, 3])
            + self.emb_num_hs(x[:, 4])
            + self.emb_radicals(x[:, 5])
            + self.emb_hybridization(x[:, 6])
            + self.emb_aromatic(x[:, 7])
            + self.emb_ring(x[:, 8])
        )

        # B. Embed Edge Features
        # edge_attr shape: [num_edges, 3]
        edge_embed = (
            self.emb_bond_type(edge_attr[:, 0])
            + self.emb_stereo(edge_attr[:, 1])
            + self.emb_conjugated(edge_attr[:, 2])
        )

        # C. Message Passing
        h = node_embed
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr=edge_embed)
            h = torch.relu(h)

        # D. Graph Pooling (Get one vector per molecule)
        # Returns shape [batch_size, hidden_dim]
        graph_embedding = global_add_pool(h, batch)
        return graph_embedding


class Graph2Caption(nn.Module):
    """
    Main Model: Projects graph embedding into GPT-2's input space.
    """

    def __init__(self, gpt2_model_name="gpt2-medium"):  # Recommended size [cite: 96]
        super().__init__()

        # 1. Graph Encoder
        self.encoder = MoleculeEncoder(hidden_dim=384)

        # 2. Text Decoder (Pre-trained)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

        # Freeze GPT-2 weights initially to prevent destroying pre-trained knowledge?
        # (Optional: depends on dataset size. With 33k samples, you might want to freeze first 10 layers)
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False

        # 3. The Bridge (Projection)
        # Maps graph dimension (384) to GPT-2 dimension (1024 for medium, 768 for small)
        gpt_emb_size = self.gpt2.config.n_embd
        self.projection = nn.Linear(384, gpt_emb_size)

    def forward(self, data, text_input_ids, text_attention_mask):
        """
        Forward pass for training.
        """
        # 1. Get Graph Embedding
        graph_vec = self.encoder(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # [batch, 384]

        # 2. Project to GPT space
        projected_emb = self.projection(graph_vec)  # [batch, gpt_emb_size]

        # 3. Prepare GPT inputs
        # Reshape projected graph to look like a token: [batch, 1, gpt_emb_size]
        projected_emb = projected_emb.unsqueeze(1)

        # Get embeddings for the text text
        text_embeds = self.gpt2.transformer.wte(
            text_input_ids
        )  # [batch, seq_len, gpt_emb_size]

        # Concatenate: [Graph_Token, Text_Tokens...]
        inputs_embeds = torch.cat((projected_emb, text_embeds), dim=1)

        # Adjust attention mask to account for the added graph token
        # Add a column of 1s to the left
        batch_size = text_attention_mask.shape[0]
        ones = torch.ones((batch_size, 1), device=text_attention_mask.device)
        extended_mask = torch.cat((ones, text_attention_mask), dim=1)

        # 4. Pass to GPT-2
        # We don't pass labels here yet because the labels need to be shifted
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=extended_mask)

        return outputs.logits

    def generate_caption(self, data, max_length=50):
        """
        Inference method for the test set.
        """
        # 1. Encode Graph
        graph_vec = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        projected_emb = self.projection(graph_vec).unsqueeze(1)

        # 2. Generate
        # We start with the graph embedding as the context
        # Note: transformers 'generate' usually expects input_ids.
        # Since we have embeddings, we might need a custom generation loop or
        # use inputs_embeds if supported by the version.

        # Simple greedy decoding loop example:
        cur_input_embeds = projected_emb
        generated_ids = []

        for _ in range(max_length):
            outputs = self.gpt2(inputs_embeds=cur_input_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated_ids.append(next_token_id)

            # Prepare next input (embedding of the predicted token)
            next_input_embeds = self.gpt2.transformer.wte(next_token_id)
            cur_input_embeds = torch.cat((cur_input_embeds, next_input_embeds), dim=1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode([t.item() for t in generated_ids])


# --- Example Usage Logic ---
# dataset = ... (Load using the provided data_utils.py)
# loader = DataLoader(dataset, batch_size=32)
# model = Graph2Caption()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# For training loop:
# labels should be the text_input_ids, but shifted by one position.
# The graph token does not have a label (we don't predict the graph).
