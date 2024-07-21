import torch
import torch.nn as nn
import torch.nn.functional as F

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # STE: pass gradient through without modification

class TopKSparsity(nn.Module):
    def __init__(self, k_ratio):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, x):
        k = int(self.k_ratio * x.shape[-1])
        topk_values, _ = torch.topk(torch.abs(x), k, dim=-1)
        mask = torch.ge(torch.abs(x), topk_values[..., -1:]).float()
        x = STEFunction.apply(x, mask)
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6)
        return x

class QuantizedTopKSparsity(nn.Module):
    def __init__(self, k_ratio):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, x):
        k = int(self.k_ratio * x.shape[-1])
        gamma = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_q = torch.round(torch.clamp(x / (gamma + 1e-6), -128, 127))
        topk_values, _ = torch.topk(torch.abs(x_q), k, dim=-1)
        mask = torch.ge(torch.abs(x_q), topk_values[..., -1:]).float()
        x_q = STEFunction.apply(x_q, mask)
        return x_q

class ReLU2GLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_up = nn.Linear(in_features, out_features)
        self.w_gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.w_up(x) * torch.relu(self.w_gate(x)) ** 2

class QSparseDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, k_ratio=0.5, quantized=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = ReLU2GLU(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.sparsity = TopKSparsity(k_ratio) if not quantized else QuantizedTopKSparsity(k_ratio)

    def forward(self, x, mask=None):
        # Apply sparsity to input
        x = self.sparsity(x)
        
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        ff_output = self.linear2(ff_output)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class QSparseModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, k_ratio, quantized=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.quantized = quantized
        
        self.decoder_layers = nn.ModuleList([
            QSparseDecoderLayer(d_model, nhead, dim_feedforward, k_ratio=k_ratio, quantized=quantized)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = self.embedding(x)
        
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        for layer in self.decoder_layers:
            x = layer(x, mask)
        
        x = self.fc(x)
        return x
