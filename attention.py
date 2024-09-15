import torch
import math
from torch import nn
from torch.nn import Functional as F


class SelfAttention(nn.Sequential):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask = False) -> torch.Tensor:
        # x : (batch_size, seq_len, d_embed)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input.shape
        interm_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)
        
        # q_shape = (batch_size, sequence_length, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (batch_size, sequence_length, d_embed) -> (batch_size, sequence_length, n_heads, d_head) -> (batch_size,  H, seq_len, dim/ H)

        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2) 

        # (batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask where upper traingle is madeup of 1
            mask = torch.ones_like(weight, d_type = torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weigth /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1) 
        
        # (batch_size, H, seq_len, seq_len) @ (batch_size, H, seq_len, dim / H) -> (batch_size, H, seq_len, dim / H)
        output = weight @ v

        # (batch_size, seq_len, H, dim /H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)
        
        # (batch_size, seq_len, dim)
        output = self.out_proj(output)

        return output


class crossAttention(n.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias = True, out_proj_bias= True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads
        
    
    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
         # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
         # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)


        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weights = q @ k.transpose(-1, -2)
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weights/=math.sqrt(self.d_head)
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weights = F.softmax(weights, dim = -1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weights @ v
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output





        