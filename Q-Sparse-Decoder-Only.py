import torch
import torch.nn as nn
import torch.nn.functional as F

class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for the backward pass.
    The forward pass multiplies the input by a mask, and the backward pass
    simply passes the gradient through without modification.
    """
    @staticmethod
    def forward(ctx, input, mask):
        # Forward pass: apply the mask to the input
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass: pass the gradient through without modification
        return grad_output, None  # STE: pass gradient through without modification

class TopKSparsity(nn.Module):
    """
    Applies top-k sparsity to the input tensor. 
    It selects the top-k absolute values along the last dimension and creates a mask 
    to zero out the rest. The output is then normalized by its L2 norm.
    
    Args:
        k_ratio (float): The ratio of elements to keep in the sparsity operation.
    """
    def __init__(self, k_ratio):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, x):
        # Determine the number of elements to keep
        k = int(self.k_ratio * x.shape[-1])
        
        # Find the top-k absolute values along the last dimension
        topk_values, _ = torch.topk(torch.abs(x), k, dim=-1)
        
        # Create a mask where the top-k elements are kept
        mask = torch.ge(torch.abs(x), topk_values[..., -1:]).float()
        
        # Apply the mask using STE and normalize the output
        x = STEFunction.apply(x, mask)
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6)
        return x

class QuantizedTopKSparsity(nn.Module):
    """
    Applies quantization followed by top-k sparsity to the input tensor.
    The input is scaled by the maximum absolute value, clamped to the range [-128, 127],
    and then rounded to simulate quantization. The top-k elements are then selected and
    the rest are zeroed out.
    
    Args:
        k_ratio (float): The ratio of elements to keep in the sparsity operation.
    """
    def __init__(self, k_ratio):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, x):
        # Determine the number of elements to keep
        k = int(self.k_ratio * x.shape[-1])
        
        # Scale the input by the maximum absolute value (gamma)
        gamma = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        
        # Quantize the input to the range [-128, 127]
        x_q = torch.round(torch.clamp(x / (gamma + 1e-6), -128, 127))
        
        # Find the top-k absolute values along the last dimension
        topk_values, _ = torch.topk(torch.abs(x_q), k, dim=-1)
        
        # Create a mask where the top-k elements are kept
        mask = torch.ge(torch.abs(x_q), topk_values[..., -1:]).float()
        
        # Apply the mask using STE
        x_q = STEFunction.apply(x_q, mask)
        return x_q

class ReLU2GLU(nn.Module):
    """
    A Gated Linear Unit (GLU) variant where the gate is a squared ReLU activation.
    The output is the element-wise product of the linear transformation and the gated activation.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_up = nn.Linear(in_features, out_features)
        self.w_gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Apply the linear transformation and the gated activation
        return self.w_up(x) * torch.relu(self.w_gate(x)) ** 2

class QSparseDecoderLayer(nn.Module):
    """
    A decoder layer with sparsity applied to the input, followed by self-attention 
    and a feed-forward network. The feed-forward network uses the ReLU2GLU mechanism.
    
    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value (default=0.1).
        k_ratio (float): The ratio of elements to keep in the sparsity operation.
        quantized (bool): Whether to use quantized sparsity.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, k_ratio=0.5, quantized=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = ReLU2GLU(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose between standard and quantized sparsity
        self.sparsity = TopKSparsity(k_ratio) if not quantized else QuantizedTopKSparsity(k_ratio)

    def forward(self, x, mask=None):
        # Apply sparsity to input
        x = self.sparsity(x)
        
        # Self-attention mechanism
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        ff_output = self.linear2(ff_output)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class QSparseModel(nn.Module):
    """
    A model with an embedding layer, multiple decoder layers, and a final linear layer for output.
    The decoder layers are instances of QSparseDecoderLayer.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        num_layers (int): The number of decoder layers.
        dim_feedforward (int): The dimension of the feedforward network model.
        k_ratio (float): The ratio of elements to keep in the sparsity operation.
        quantized (bool): Whether to use quantized sparsity.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, k_ratio, quantized=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.quantized = quantized
        
        # Stack multiple decoder layers
        self.decoder_layers = nn.ModuleList([
            QSparseDecoderLayer(d_model, nhead, dim_feedforward, k_ratio=k_ratio, quantized=quantized)
            for _ in range(num_layers)
        ])
        
        # Final linear layer to map to vocabulary size
        self.fc = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square subsequent mask for the decoder to prevent attending to future tokens.
        
        Args:
            sz (int): The size of the mask (sequence length).
        
        Returns:
            torch.Tensor: The generated mask.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # Apply embedding layer
        x = self.embedding(x)
        
        # Generate a square subsequent mask for the decoder
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # Pass through each decoder layer
        for layer in self.decoder_layers:
            x = layer(x, mask)
        
        # Final linear layer to produce output logits
        x = self.fc(x)
        return x
