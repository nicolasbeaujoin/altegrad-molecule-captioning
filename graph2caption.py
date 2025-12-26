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


class MultiTokenProjector(nn.Module):
    def __init__(self, gnn_hidden=256, gpt_hidden=1024, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.gnn_hidden = gnn_hidden

        # 1. Learnable queries: These tokens "extract" info from the graph
        self.latents = nn.Parameter(torch.randn(1, num_tokens, gnn_hidden))

        # 2. Cross-Attention: Latents (Query) look at Node Features (Key/Value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=gnn_hidden, num_heads=8, batch_first=True
        )

        # 3. LayerNorm for training stability
        self.ln_gnn = nn.LayerNorm(gnn_hidden)
        self.ln_gpt = nn.LayerNorm(gpt_hidden)

        # 4. Final projection to GPT-2 dimension (e.g., 1024) [cite: 72, 96]
        self.proj = nn.Linear(gnn_hidden, gpt_hidden)

    def forward(self, node_features, batch_index):
        # Convert flattened PyG nodes to a dense batch [cite: 31, 32, 34]
        # dense_nodes: [Batch, MaxNodesPerBatch, 256]
        # mask: [Batch, MaxNodesPerBatch] (True for real nodes, False for padding)
        dense_nodes, mask = to_dense_batch(node_features, batch_index)

        # Prepare queries for the batch
        batch_size = dense_nodes.size(0)
        query = self.latents.expand(batch_size, -1, -1)  # [Batch, 8, 256]

        # Cross-Attention:
        # MultiheadAttention uses 'key_padding_mask' where True = MASK OUT (ignore)
        # PyG's mask is True = KEEP. So we use ~mask.
        attn_out, _ = self.cross_attn(
            query=query, key=dense_nodes, value=dense_nodes, key_padding_mask=~mask
        )

        # Residual and Normalization
        out = self.ln_gnn(attn_out + query)

        # Project to GPT embedding space [cite: 72]
        out = self.proj(out)
        return self.ln_gpt(out)


# Model parameters
NODE_VOCAB_SIZES = [119, 9, 11, 12, 9, 5, 8, 2, 2]
EDGE_VOCAB_SIZES = [22, 6, 2]


class MolGNN(nn.Module):
    def __init__(self, hidden=256, layers=5, dropout=0.1):
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
        # vn_emb = self.virtual_node_emb(
        #     torch.zeros(batch_idx.max() + 1, dtype=torch.long, device=h.device)
        # )
        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = F.relu(bn(conv(h, edge_index, edge_emb)))
            h = self.dropout(h) + h_in  # Residual connection
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
        return h


class Graph2CaptionV2(nn.Module):
    def __init__(
        self, pretrained_encoder, gpt2_model_name="gpt2-medium", num_graph_tokens=8
    ):
        super().__init__()
        # 1. Store the upgraded GNN encoder
        self.encoder = pretrained_encoder
        self.num_graph_tokens = num_graph_tokens

        # 2. Text Decoder Setup [cite: 96]
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.gpt2.resize_token_embeddings(len(self.tokenizer))
        gpt_hidden_dim = self.gpt2.config.n_embd  # 1024 for medium [cite: 72, 73]
        gnn_hidden_dim = 512  # Updated hidden size for upgraded GNN

        # 3. Multi-Token Projector (Attention-based Bridge) [cite: 7, 14]
        # Instead of 1 vector, this extracts 'num_graph_tokens' from the molecule
        self.projector = MultiTokenProjector(
            gnn_hidden=gnn_hidden_dim,
            gpt_hidden=gpt_hidden_dim,
            num_tokens=num_graph_tokens,
        )

    def forward(self, data, text_input_ids, text_attention_mask):
        # A. Extract detailed node features (no global pooling yet) [cite: 13, 14]
        # Shape: [TotalNodesInBatch, 256]
        node_features = self.encoder.forward_features(data)

        # B. Generate Multi-Token Graph Representation [cite: 7, 14]
        # Shape: [Batch, num_graph_tokens, 1024]
        graph_tokens = self.projector(node_features, data.batch)

        # C. Prepare Text Embeddings
        text_embeds = self.gpt2.transformer.wte(text_input_ids)  # [Batch, SeqLen, 1024]

        # D. Concatenate: [Graph_Tokens... , Text_Tokens...] [cite: 7, 14]
        # Resulting shape: [Batch, num_graph_tokens + SeqLen, 1024]
        inputs_embeds = torch.cat((graph_tokens, text_embeds), dim=1)

        # E. Extend Attention Mask [cite: 86]
        # We must add 'num_graph_tokens' ones to the mask so GPT-2 attends to the graph
        batch_size = text_attention_mask.shape[0]
        graph_mask = torch.ones(
            (batch_size, self.num_graph_tokens), device=text_attention_mask.device
        )
        extended_mask = torch.cat((graph_mask, text_attention_mask), dim=1)

        return self.gpt2(
            inputs_embeds=inputs_embeds, attention_mask=extended_mask
        ).logits

    def generate_caption(self, data, max_length=100):
        self.eval()
        with torch.no_grad():
            # 1. Encode Graph to tokens [Batch, 8, 1024]
            node_features = self.encoder.forward_features(data)
            graph_tokens = self.projector(node_features, data.batch)

            # 2. Create the Attention Mask manually
            # Since we are doing inference molecule-by-molecule (Batch Size 1),
            # we create a mask of 1s for the 8 graph tokens.
            batch_size = graph_tokens.shape[0]
            num_tokens = graph_tokens.shape[1]  # This is 8
            attention_mask = torch.ones(
                (batch_size, num_tokens), device=graph_tokens.device
            )

            # 3. Generate using the mask
            output_ids = self.gpt2.generate(
                inputs_embeds=graph_tokens,
                attention_mask=attention_mask,  # <--- PASS THE MASK HERE
                max_new_tokens=max_length,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                temperature=0.8,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
