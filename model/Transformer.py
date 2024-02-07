import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        params:
            d_model: int (dimension of model)
            nhead: int (number of heads)
            dim_feedforward: int (dimension of feedforward)
            dropout: float (dropout rate)
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = query + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src
    

class TransformerEncoder(nn.Module):
    def __init__(self,  d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=6):
        """
        params:
            d_model: int (dimension of model)
            nhead: int (number of heads)
            dim_feedforward: int (dimension of feedforward)
            dropout: float (dropout rate)
            num_layers: int (number of layers)
        """
        super(TransformerEncoder, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None, src_key_padding_mask=None):
        output = query

        for mod in self.layers:
            output = mod(query, key, value, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output)

        return output
    

if __name__ == '__main__':
    from torchsummary import summary

    model = TransformerEncoder()
    print(model)
