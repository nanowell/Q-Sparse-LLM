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

class TopKSparsitySTE(nn.Module):
    def __init__(self, k_ratio):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, x):
        k = int(self.k_ratio * x.shape[-1])
        topk_values, _ = torch.topk(torch.abs(x), k, dim=-1)
        mask = torch.ge(torch.abs(x), topk_values[..., -1:]).float()
        x = STEFunction.apply(x, mask)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
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
        mask = torch.ge(torch.abs(x_q), topk_values[..., -1:])
        x_q = STEFunction.apply(x_q, mask.float())
        return x_q

class ReLU2GLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_up = nn.Linear(in_features, out_features)
        self.w_gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.w_up(x) * F.relu(self.w_gate(x)) ** 2

class QSparseModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, k_ratio, quantized=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.quantized = quantized
        self.sparsity = TopKSparsitySTE(k_ratio) if not quantized else QuantizedTopKSparsity(k_ratio)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation=F.relu  # Use standard ReLU activation, skill issue on my part
            ) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.sparsity(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.fc(x)
        return x
