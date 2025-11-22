"""
This code is based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import torch.nn.functional as F
import tsp
import cvrp
import math


class Embedding(nn.Module):
    """Encodes the coordinate states using 1D Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Embedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)

    def forward(self, input_data):
        output_data = self.embed(input_data)
        return output_data


class Encoder(nn.Module):
    def __init__(self, instance_embedding, reference_embedding, encoder_attn, rnn, update_fn, search_space_size,
                 hidden_size):
        super(Encoder, self).__init__()
        self.instance_embedding = instance_embedding
        self.reference_embedding = reference_embedding
        self.encoder_attn = encoder_attn
        self.gru_decoder = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, search_space_size)
        self.fc2 = nn.Linear(hidden_size, search_space_size)
        self.rnn = rnn
        self.update_fn = update_fn

    def forward(self, instance, solution, instance_hidden, config):
        batch_size, sequence_size, input_size, = instance.size()
        reference_input = instance[torch.arange(batch_size), solution[:, 0], :].unsqueeze(1).detach()

        last_hh = None
        last_hh_2 = None
        reference_hidden = self.reference_embedding(reference_input)
        for j in range(1, solution.shape[1]):

            rnn_out, last_hh = self.rnn(reference_hidden, last_hh)

            # Given a summary of the output, find an  input context
            enc_attn = self.encoder_attn(instance_hidden, rnn_out)
            context = enc_attn.permute(0, 2, 1).bmm(instance_hidden)

            ptr = solution.t()[j].long()

            if self.update_fn is not None:
                instance = self.update_fn(instance, ptr.data)
                instance_hidden = self.instance_embedding(instance)

            reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size))
            reference_hidden = self.reference_embedding(reference_input)

            rnn_input = torch.cat((reference_hidden, context), dim=2)
            rnn_out_2, last_hh_2 = self.gru_decoder(rnn_input, last_hh_2)

        mu = self.fc1(last_hh_2.squeeze(0))
        log_var = self.fc2(last_hh_2.squeeze(0))
        return self.reparameterise(mu, log_var), mu, log_var

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# --- Alternative RoPE Implementation ---
def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary positional embedding."""
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)."""

    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Positional Encoding."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # q/k/v projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # RoPE
        self.rope = RotaryPositionalEncoding(self.head_dim)

    def forward(self, query, key, value):
        """
        Args:
            query: (B, seq_len, embed_dim)
            key: (B, seq_len, embed_dim)
            value: (B, seq_len, embed_dim)

        Returns:
            attn_output: (B, seq_len, embed_dim)
            attn_weights: (B, seq_len, seq_len)
        """
        B, L, E = query.shape
        assert E == self.embed_dim

        # Linear projections
        q = self.q_proj(query)  # (B, L, E)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (B, L, num_heads, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        # Apply RoPE to q and k
        q = self.rope(q, seq_len=L)
        k = self.rope(k, seq_len=L)

        # Transpose to (B, num_heads, L, head_dim) for attention
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, L, L)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        if self.training and self.dropout > 0.0:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        attn_out = torch.matmul(attn_probs, v)  # (B, num_heads, L, head_dim)

        # Merge heads: (B, L, E)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, E)
        out = self.out_proj(attn_out)

        return out, attn_probs.mean(dim=1)


class TransEncoder(nn.Module):
    def __init__(self, search_space_size):
        super(TransEncoder, self).__init__()

        num_layers: int = 3
        num_heads: int = 8
        model_size: int = 128
        expand_factor: int = 4

        self.num_layers = num_layers

        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'mha': MultiHeadAttention(embed_dim=model_size, num_heads=num_heads, dropout=0.0),
                'norm1': nn.LayerNorm(model_size),
                'mlp': nn.Sequential(
                    nn.Linear(model_size, expand_factor * model_size),
                    nn.ReLU(),
                    nn.Linear(expand_factor * model_size, model_size),
                ),
                'norm2': nn.LayerNorm(model_size),
            })
            self.layers.append(layer)

        # VAE projection layers
        self.fc1 = nn.Linear(model_size, search_space_size)  # For mu
        self.fc2 = nn.Linear(model_size, search_space_size)  # For log_var

    def _initialize_weights(self):
        """Initialize weights with scaled initialization."""
        # Scale attention weights by 1/num_layers
        scale = 1.0 / self.num_layers
        for layer in self.layers:
            # Scale the output projection of multi-head attention
            nn.init.xavier_uniform_(layer['mha'].out_proj.weight, gain=scale)

    def forward(self, instance_hidden, solution):

        batch_size = instance_hidden.shape[0]
        x = torch.gather(
            instance_hidden, 1,
            solution.unsqueeze(-1).expand(-1, -1, instance_hidden.shape[-1])
        )

        for layer in self.layers:
            # Multi-head attention with residual
            attn_out, _ = layer['mha'](x, x, x)
            x = layer['norm1'](x + attn_out)

            # Feed-forward network with residual
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + mlp_out)

        pooled = x.mean(dim=1)  # [batch_size, model_size]

        # Project to VAE parameters
        mu = self.fc1(pooled)
        log_var = self.fc2(pooled)
        z = self.reparameterise(mu, log_var)

        return z, mu, log_var

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, 1 * hidden_size), requires_grad=True))

    def forward(self, instance_hidden, rnn_out):
        batch_size, _, hidden_size = instance_hidden.size()

        hidden = rnn_out.expand_as(instance_hidden)
        hidden = torch.cat((instance_hidden, hidden), 2)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, -1, -1)
        W = self.W.expand(batch_size, -1, -1)

        ret = torch.bmm(hidden, W)
        attns = torch.bmm(torch.relu(ret), v)
        attns = F.softmax(attns, dim=1)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, encoder_attn, rnn, hidden_size, search_space_size):
        super(Pointer, self).__init__()

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1)
                                          , requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, hidden_size)
                                          , requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.fc1 = nn.Linear(2 * hidden_size + search_space_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.encoder_attn = encoder_attn
        self.rnn = rnn

    def forward(self, instance_hidden, reference_hidden, Z, last_hh):
        rnn_out, last_hh = self.rnn(reference_hidden, last_hh)
        rnn_out = rnn_out

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(instance_hidden, rnn_out)
        context = enc_attn.permute(0, 2, 1).bmm(instance_hidden)  # (B, 1, num_feats)

        fc_input = torch.cat((context.squeeze(1), Z, reference_hidden.squeeze(1)), dim=1)  # (B, num_feats, seq_len)
        fc_output = self.fc1(fc_input)
        fc_output = self.fc2(fc_output).unsqueeze(1)
        fc_output = fc_output.expand(-1, instance_hidden.size(1), -1)
        fc_output = torch.cat((instance_hidden, fc_output), dim=2)

        v = self.v.expand(instance_hidden.size(0), -1, -1)
        W = self.W.expand(instance_hidden.size(0), -1, -1)
        probs = torch.bmm(torch.tanh(torch.bmm(fc_output, W)), v).squeeze(2)
        return probs, last_hh


class Decoder(nn.Module):
    def __init__(self, instance_embedding, reference_embedding, encoder_attn, rnn, hidden_size,
                 search_space_size, mask_fn, update_fn):
        super(Decoder, self).__init__()

        # Define the encoder & decoder models
        self.pointer = Pointer(encoder_attn, rnn, hidden_size, search_space_size)
        self.instance_embedding = instance_embedding
        self.reference_embedding = reference_embedding
        self.encoder_attn = encoder_attn
        self.mask_fn = mask_fn
        self.update_fn = update_fn
        self.rnn = rnn

    def forward(self, instance, solution, Z, instance_hidden, config, teacher_forcing, last_hh_new=None):
        batch_size, sequence_size, input_size, = instance.size()
        reference_input = instance[torch.arange(batch_size), solution[:, 0], :].unsqueeze(1).detach()
        max_steps = sequence_size if self.mask_fn is None else 10000
        tour_idx, tour_logp, tour_prob = [solution[:, [0]]], [], []

        mask = torch.ones(batch_size, sequence_size, device=config.device)
        mask[torch.arange(batch_size), solution[:, 0]] = 0
        for j in range(1, max_steps):
            if not mask.byte().any():
                break

            reference_hidden = self.reference_embedding(reference_input)
            probs, last_hh_new = self.pointer(instance_hidden, reference_hidden, Z, last_hh_new)
            probs = F.softmax(probs + mask.log(), dim=1)
            if teacher_forcing:
                # Select the actions based on the training solutions (during training)
                ptr = solution.t()[j].long()
                t = mask[torch.arange(len(mask)), ptr]
                assert t.eq(1).all()
                logp = torch.log(probs[torch.arange(batch_size), ptr])
                _, predicted_ptr = torch.max(probs, 1)
                tour_idx.append(predicted_ptr.data.unsqueeze(1))
            else:
                # Select actions greedily (during the search)
                prob, ptr = torch.max(probs, 1)
                logp = prob.log()
                tour_idx.append(ptr.data.unsqueeze(1))

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                instance = self.update_fn(instance, ptr.data)
                instance_hidden = self.instance_embedding(instance)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot
                # in these cases, and logp := 0
                is_done = instance[:, 3, :].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            if self.mask_fn is not None:
                mask = self.mask_fn(mask, instance[:, :, 2:], ptr).detach()

            reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size))
            tour_prob.append(probs)
            tour_logp.append(logp.unsqueeze(1))

        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)

        return None, tour_idx, tour_logp

class TransformerDecoder(nn.Module):
    """
    Transformer decoder for TSP that produces logits for next node selection.

    Args:
        embedding_dim: Dimension of node embeddings
        head_num: Number of attention heads
        qkv_dim: Dimension of query/key/value in attention
        latent_dim: Dimension of latent variable Z
        logit_clipping: Clipping value for logits (typically 10.0)
    """
    def __init__(self, embedding_dim, head_num, latent_dim, logit_clipping=10.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = embedding_dim//head_num
        self.latent_dim = latent_dim
        self.logit_clipping = logit_clipping
        use_bias = True

        # Query projections (conditioned on last node + Z)
        self.Wq_first = nn.Linear(embedding_dim + latent_dim, head_num * self.qkv_dim, bias=use_bias)
        self.Wq_last = nn.Linear(embedding_dim + latent_dim, head_num * self.qkv_dim, bias=use_bias)

        # Key and Value projections (for all nodes)
        self.Wk = nn.Linear(embedding_dim, head_num * self.qkv_dim, bias=use_bias)
        self.Wv = nn.Linear(embedding_dim, head_num * self.qkv_dim, bias=use_bias)

        # Multi-head combination
        self.multi_head_combine = nn.Linear(head_num * self.qkv_dim, embedding_dim)

        # Cached values
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

    def forward(self, graph_emb, current_tour, Z):
        """
        Args:
            graph_emb: Node embeddings [batch, problem_size, embedding_dim]
            current_tour: Current partial tour [batch, tour_length] with node indices
            Z: Latent variable [batch, latent_dim]
            config: Optional config object

        Returns:
            logits: [batch, problem_size] logits for next node selection
        """

        # Set up keys and values from graph embeddings (do this once)
        if self.k is None:
            self.k = self._reshape_by_heads(self.Wk(graph_emb), self.head_num)
            self.v = self._reshape_by_heads(self.Wv(graph_emb), self.head_num)
            self.single_head_key = graph_emb.transpose(1, 2)


            # Get embedding of first node in tour
        first_node_idx = current_tour[:, 0:1]  # [batch, 1]
        first_node_emb = self._get_encoding(graph_emb, first_node_idx)  # [batch, 1, embedding_dim]
        last_node_idx = current_tour[:, -1:]  # [batch, 1]
        last_node_emb = self._get_encoding(graph_emb, last_node_idx)  # [batch, 1, embedding_dim]

        # Set q_first (query for first node + Z)
        Z = Z.unsqueeze(1)

        if self.q_first is None:
            input_cat = torch.cat((first_node_emb, Z), dim=2)
            self.q_first = self._reshape_by_heads(self.Wq_first(input_cat), self.head_num)


        # Compute attention and get logits
        logits = self._compute_logits(last_node_emb, Z)

        logits = logits.squeeze(1)  # [batch, problem_size]

        return logits

    def reset_cached_values(self):
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None


    def _compute_logits(self, last_node_emb, Z):
        """Compute logits using multi-head attention."""
        head_num = self.head_num

        # Concatenate last node embedding with Z
        input_cat = torch.cat((last_node_emb, Z), dim=2)
        # shape: (batch, 1, embedding_dim + latent_dim)

        # Compute q_last
        q_last = self._reshape_by_heads(self.Wq_last(input_cat), head_num)
        # shape: (batch, head_num, 1, qkv_dim)

        # Combine queries
        q = self.q_first + q_last
        # shape: (batch, head_num, 1, qkv_dim)

        # Multi-head attention
        out_concat = self._multi_head_attention(q, self.k, self.v)
        # shape: (batch, K, head_num * qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, K, embedding_dim)

        # Single-head attention for probability calculation
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, 1, problem_size)


        # Scale and clip
        sqrt_embedding_dim = self.embedding_dim ** 0.5
        score_scaled = score / sqrt_embedding_dim
        logits = self.logit_clipping * torch.tanh(score_scaled)


        return logits

    def _get_encoding(self, graph_emb, node_idx):
        """Get node embeddings by index."""
        # graph_emb: [batch, problem_size, embedding_dim]
        # node_idx: [batch, n]
        batch_size = node_idx.size(0)
        embedding_dim = graph_emb.size(2)
        gathering_index = node_idx[:, :, None].expand(batch_size, 1, embedding_dim)
        picked_nodes = graph_emb.gather(dim=1, index=gathering_index)

        return picked_nodes

    def _reshape_by_heads(self, qkv, head_num):
        """Reshape tensor for multi-head attention."""
        batch_s = qkv.size(0)
        n = qkv.size(1)

        q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
        q_transposed = q_reshaped.transpose(1, 2)

        return q_transposed

    def _multi_head_attention(self, q, k, v):
        """Compute multi-head attention."""
        batch_s = q.size(0)
        head_num = q.size(1)
        n = q.size(2)
        key_dim = q.size(3)

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, problem_size)

        score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

        weights = F.softmax(score_scaled, dim=3)
        # shape: (batch, head_num, n, problem_size)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
        # shape: (batch, n, head_num * key_dim)

        return out_concat


class TransformerBasedDecoder(nn.Module):
    """
    Wrapper that integrates the GraphPermutationDecoder into your existing framework.
    Replaces the original Decoder class.
    """

    def __init__(self, instance_embedding, search_space_size, mask_fn, update_fn):
        super().__init__()

        self.instance_embedding = instance_embedding  # Keep for compatibility
        self.mask_fn = mask_fn
        self.update_fn = update_fn

        num_layers: int = 3
        num_heads: int = 8
        model_size: int = 128
        expand_factor: int = 1

        self.num_layers = num_layers

        #Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'mha': nn.MultiheadAttention(embed_dim=model_size, num_heads=num_heads, dropout=0.2, batch_first=True),
                'mlp': nn.Sequential(
                    nn.Linear(model_size, expand_factor * model_size),
                    nn.GELU(),
                    nn.Linear(expand_factor * model_size, model_size),
                )
            })
            self.layers.append(layer)

        self._init_weights()

        self.transformer_decoder = TransformerDecoder(
            embedding_dim=model_size,
            head_num=num_heads,
            latent_dim=search_space_size
        )

        cost_input_dim = model_size + search_space_size
        self.cost_predictor = nn.Sequential(
            nn.Linear(cost_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _print_grad(self, name):
        """Helper to print gradient stats during backward pass"""
        def hook(grad):
            if grad is None:
                print(f"{name}: Gradient is None")
                return
            norm = grad.norm().item()
            mean = grad.abs().mean().item()
            has_nan = torch.isnan(grad).any().item()
            print(f"{name} | Norm: {norm:.4f} | Mean: {mean:.4f} | Has NaN: {has_nan}")
        return hook

    def _init_weights(self):
        """Initialize weights with smaller values to prevent gradient issues"""
        for layer in self.layers:
            # Scale down MLP weights
            for module in layer['mlp']:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, instance, solution, Z, instance_hidden, config, teacher_forcing, pred_cost = True):
        """
        Maintains compatibility with existing interface but uses transformer internally.
        """
        self.transformer_decoder.reset_cached_values()

        batch_size, sequence_size, input_size = instance.size()

        graph_emb = instance_hidden

        # if graph_emb.requires_grad:
        #     graph_emb.retain_grad()  # Required for intermediate tensors
        #     graph_emb.register_hook(self._print_grad("GRAD BEFORE Layers (Input)"))

        for layer in self.layers:
            # Multi-head attention with residual
            attn_out, _ = layer['mha'](graph_emb, graph_emb, graph_emb)
            graph_emb = graph_emb + 0.1*attn_out

            # Feed-forward network with residual
            mlp_out = layer['mlp'](graph_emb)
            graph_emb = graph_emb + 0.1*mlp_out
        #
        # if graph_emb.requires_grad:
        #     graph_emb.retain_grad() # Required because graph_emb was overwritten
        #     graph_emb.register_hook(self._print_grad("GRAD AFTER Layers (Output)"))


        if pred_cost:
            p_cost= self.cost_predictor(torch.cat((graph_emb.mean(dim=1), Z), 1))
        else :
            p_cost = None



        tour_idx, tour_logp, tour_prob = [solution[:, [0]]], [], []

        # Create mask for visited cities
        mask = torch.ones(batch_size, sequence_size, device=config.device)
        mask[torch.arange(batch_size), solution[:, 0]] = 0


        # Build tour step by step
        for j in range(1, sequence_size):
            if not mask.byte().any():
                break

            # Current partial tour
            current_tour = torch.cat(tour_idx, dim=1)  # [batch, j]

            # Get logits for next city using transformer
            logits = self.transformer_decoder(graph_emb, current_tour, Z)


            probs = F.softmax(logits.masked_fill(mask == 0, -1e9), dim=1)


            if teacher_forcing:
                # Use ground truth
                ptr = solution[:, j].long()
                logp = torch.log(probs[torch.arange(batch_size), ptr])
                _, predicted_ptr = torch.max(probs, 1)
                tour_idx.append(predicted_ptr.data.unsqueeze(1))
            else:
                # Greedy selection
                prob, ptr = torch.max(probs, 1)
                logp = prob.log()
                tour_idx.append(ptr.data.unsqueeze(1))

            # Update mask
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, instance[:, :, 2:], ptr).detach()

            tour_logp.append(logp.unsqueeze(1))

        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)

        return None, tour_idx, tour_logp, p_cost

class VAE_8(nn.Module):
    def __init__(self, config):
        super(VAE_8, self).__init__()
        if config.problem == "TSP":
            input_size = 2
            mask_fn = tsp.update_mask
            update_fn = None
        elif config.problem == "CVRP":
            input_size = 4
            mask_fn = cvrp.update_mask
            update_fn = cvrp.update_dynamic


        hidden_size = 128
        self.instance_embedding = Embedding(input_size, hidden_size)
        reference_embedding = Embedding(input_size, hidden_size)
        encoder_attn = Attention(hidden_size)
        rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, dropout=0)

        self.encoder = TransEncoder(config.search_space_size)
        self.decoder = TransformerBasedDecoder(self.instance_embedding, config.search_space_size, mask_fn, update_fn)
        # self.encoder = Encoder(self.instance_embedding, reference_embedding, encoder_attn, rnn, update_fn,
        #                        config.search_space_size, hidden_size)
        #
        # self.decoder = Decoder(self.instance_embedding, reference_embedding, encoder_attn, rnn, hidden_size,
        #                        config.search_space_size, mask_fn, update_fn)

        self.instance_hidden = None
        self.dummy_solution = None

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, instance, solution_1, solution_2, config):
        instance_hidden = self.instance_embedding(instance)
        output_e = self.encoder(instance_hidden, solution_1)
        #output_e = self.encoder(instance, solution_1, instance_hidden, config)

        Z, mu, log_var = output_e

        output_prob, tour_idx, tour_logp, pred_costs = self.decoder(instance, solution_2, Z, instance_hidden, config,
                                                        True)
        return output_prob, mu, log_var, Z, tour_idx, tour_logp, pred_costs

    def decode(self, instance, Z, config):
        if self.instance_hidden is None:
            self.instance_hidden = self.instance_embedding(instance)
        output_prob, tour_idx, tour_logp, pred_costs = self.decoder(instance, self.dummy_solution, Z, self.instance_hidden, config,
                                                        False, False)
        return output_prob, tour_idx, tour_logp

    def reset_decoder(self, batch_size, config):
        self.instance_hidden = None
        self.dummy_solution = torch.zeros(batch_size, 1).long().to(config.device)