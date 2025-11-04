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

# --- Cyclical RoPE + MultiHeadAttention with Cyclical RoPE ---
class CyclicalRoPE(nn.Module):
    """
    Cyclical RoPE: This produces wrap-around (pos 0 ~ pos N) and multi-scale frequency channels.
    """
    def __init__(self, head_dim: int):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary (pairs).")
        self.head_dim = head_dim
        # inv_freq shape: head_dim/2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)  # shape (head_dim/2,)

    def get_sin_cos(self, seq_len: int, device):
        """
        Return sin, cos tensors shaped for broadcasting with [B, seq_len, heads, head_dim/2]
        sin, cos shape -> (1, seq_len, 1, head_dim/2)
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # (seq_len,)
        # base circular angle per position (wraps at seq_len)
        theta = 2.0 * math.pi * positions / float(seq_len+1)  # (seq_len,)
        # freqs: outer(theta, inv_freq) -> (seq_len, head_dim/2)
        freqs = torch.einsum("p,d->pd", theta, self.inv_freq.to(device))
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)
        return sin, cos

    @staticmethod
    def apply_rotary(x, sin, cos):
        """
        Returns x with rotary applied.
        """
        # split even / odd
        x_even = x[..., ::2]  # (..., head_dim/2)
        x_odd = x[..., 1::2]  # (..., head_dim/2)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        # interleave back to (..., head_dim)
        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)
        return x_rot


class MultiHeadAttentionCyclicRoPE(nn.Module):
    """
    Multi-head attention that applies cyclical RoPE
    """
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

        # RoPE per head
        # We'll use same head_dim for all heads; CyclicalRoPE expects head_dim
        self.rope = CyclicalRoPE(self.head_dim)

    def forward(self, query, key, value):
        """
        Returns:
            attn_output: (B, seq_len, embed_dim)
            attn_weights: (B, seq_len, seq_len) e
        """
        B, L, E = query.shape
        device = query.device
        assert E == self.embed_dim

        # Linear projections
        q = self.q_proj(query)  # (B, L, E)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (B, L, heads, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        # Get sin/cos for this seq length
        sin, cos = self.rope.get_sin_cos(L, device)  # (1, L, 1, head_dim/2)

        # Apply RoPE to q and k: operate on last head_dim by pairs
        # Convert q/k to shape compatible with even/odd split
        q = self.rope.apply_rotary(q, sin, cos)  # (B, L, heads, head_dim)
        k = self.rope.apply_rotary(k, sin, cos)

        # Transpose to (B, heads, L, head_dim) for attention
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, L, L)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        if self.training and self.dropout > 0.0:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        attn_out = torch.matmul(attn_probs, v)  # (B, heads, L, head_dim)
        # Merge heads: (B, L, E)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, E)
        out = self.out_proj(attn_out)


        return out, attn_probs.mean(dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create circular (sin, cos) positional encodings
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Map positions to angles along a circle [0, 2π)
        theta = 2 * math.pi * position / max_len

        pos_enc[:, 0::2] = torch.sin(theta * div_term)
        pos_enc[:, 1::2] = torch.cos(theta * div_term)

        pos_enc = pos_enc.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        seq_len = embedding.size(1)
        return embedding + self.pos_enc[:, :seq_len, :]


class TransEncoder(nn.Module):
    def __init__(self, search_space_size, max_len):
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
                'mha': MultiHeadAttentionCyclicRoPE(embed_dim=model_size, num_heads=num_heads, dropout=0.0),
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

        # self.encoder = Encoder(self.instance_embedding, reference_embedding, encoder_attn, rnn, update_fn,
        #                        config.search_space_size, hidden_size)
        self.encoder  = TransEncoder( config.search_space_size, config.problem_size)
        self.decoder = Decoder(self.instance_embedding, reference_embedding, encoder_attn, rnn, hidden_size,
                               config.search_space_size, mask_fn, update_fn)

        self.instance_hidden = None
        self.dummy_solution = None

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, instance, solution_1, solution_2, config):
        instance_hidden = self.instance_embedding(instance)
        output_e = self.encoder( instance_hidden,solution_1)

        Z, mu, log_var = output_e

        output_prob, tour_idx, tour_logp = self.decoder(instance, solution_2, Z, instance_hidden, config,
                                                        True)
        return output_prob, mu, log_var, Z, tour_idx, tour_logp

    def decode(self, instance, Z, config):
        if self.instance_hidden is None:
            self.instance_hidden = self.instance_embedding(instance)
        output_prob, tour_idx, tour_logp = self.decoder(instance, self.dummy_solution, Z, self.instance_hidden, config,
                                                        False)
        return output_prob, tour_idx, tour_logp

    def reset_decoder(self, batch_size, config):
        self.instance_hidden = None
        self.dummy_solution = torch.zeros(batch_size, 1).long().to(config.device)
