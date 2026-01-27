import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as functional

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, depth, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        self.depth = depth
        self.d_model = d_model
        self.num_heads = num_heads
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0) # get batch size
        # 1) Linear projections to get the multi-head query, key and value tensors
        # x_query, x_key, x_value dimension: nbatch * seq_len * d_embed
        # LHS query, key, value dimensions: nbatch * h * seq_len * d_k
        query = self.WQ(x_query).view(nbatch, -1, self.num_heads, self.depth).transpose(1,2)
        key   = self.WK(x_key).view(nbatch, -1, self.num_heads, self.depth).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.num_heads, self.depth).transpose(1,2)
        # 2) Attention
        # scores has dimensions: nbatch * h * seq_len * seq_len
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.depth) # TODO scaled_dot_product_attention
        # 3) Mask out padding tokens and future tokens
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        # p_atten dimensions: nbatch * h * seq_len * seq_len
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        # x dimensions: nbatch * h * seq_len * d_k
        x = torch.matmul(p_atten, value)
        # x now has dimensions:nbtach * seq_len * d_embed
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_model)
        return self.linear(x) # final linear layer

class ResidualConnection(nn.Module):
  '''residual connection: x + dropout(sublayer(layernorm(x))) '''
  def __init__(self, dim, dropout):
      super().__init__()
      self.drop = nn.Dropout(dropout)
      self.norm = nn.LayerNorm(dim)

  def forward(self, x, sublayer):
      return x + self.drop(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    '''EncoderBlock: self-attention -> position-wise fully connected feed-forward layer'''
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.d_model = config.d_model
        self.num_heads = config.n_heads
        self.d_ff = config.d_ff
        self.dropout = config.dropout
        self.depth = self.d_model // self.num_heads

        self.atten = MultiHeadedAttention(self.num_heads, self.d_model, self.depth, self.dropout)
        self.feed_forward = nn.Sequential( 
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff , self.d_model)
        )
        self.residual1 = ResidualConnection(self.d_model, self.dropout)
        self.residual2 = ResidualConnection(self.d_model, self.dropout)

    def forward(self, x, mask=None):
        # self-attention
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        # position-wise fully connected feed-forward layer
        return self.residual2(x, self.feed_forward)


class Encoder(nn.Module):
    '''Encoder = token embedding + positional embedding -> a stack of N EncoderBlock -> layer norm'''
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_input = config.n_frames + 1 
        self.dropout = config.dropout
        self.max_pos = 50
        self.keypoints = config.n_keypoints
        self.device = config.device
        self.tok_embed = KeypointsEmbedding(self.d_model, self.d_input, self.keypoints, self.device)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder)])
        self.dropout = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model )

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        #x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        #x = self.pos_embed(x)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear1 = nn.Linear(config.d_model, config.mlp_size)
        self.linear2 = nn.Linear(config.mlp_size, num_classes)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        x = x[:,0,:]
        x = self.linear1(x)
        x = self.linear2(x)
        return  x



class KeypointsEmbedding(nn.Module):
    def __init__(self, d_model, d_input, keypoints, device):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.device = device
        self.n_frames = 30 + 1
        self.class_tokens = nn.init.kaiming_normal_(torch.empty(1,1,d_model))
        self.position_embedding = nn.Embedding(self.n_frames, d_model) 
        self.linear = nn.Linear(keypoints, d_model)
        #self.sqrt_dim_embed = math.sqrt(d_model)

    
    def forward(self, inputs):
        
        inputs = self.linear(inputs)
        tokens = self.class_tokens.repeat(inputs.size(0),1,1).to(self.device)
        x = torch.cat((inputs, tokens), dim=1)

        # Position Embbeding
        positions = torch.range(start=0, end=self.n_frames-1).type(torch.LongTensor).to(self.device)
        pe = self.position_embedding(positions)
        x = x + pe
        #x = x.type(torch.LongTensor)
        #x = x.to('cuda')
        #x = self.position_embedding(x)
        #x = x * self.sqrt_dim_embed
        return x


