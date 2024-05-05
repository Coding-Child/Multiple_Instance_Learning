import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math

from model.ResNet import ResNet50
from model.ResNet import CNN


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=False, norm_first=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first, norm_first=norm_first)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):        
        x = src
        if self.norm_first:
            temp, attn_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + temp
            x = x + self._ff_block(self.norm2(x))
        else:
            temp, attn_weights = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self.norm1(x + temp)
            x = self.norm2(x + self._ff_block(x))

        return x, attn_weights
        
    
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, attn_weights = self.self_attn(x, x, x,
                                         attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask,
                                         need_weights=True, average_attn_weights=True, is_causal=is_causal)
        
        return self.dropout1(x), attn_weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor=enable_nested_tensor)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weights_list = list()

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_list.append(attn_weights)

        if self.norm:
            output = self.norm(output)

        return output, attn_weights_list

class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_fc):
        super(ClassificationHead, self).__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_fc)])
        self.classifier.append(nn.Linear(d_model, 1))

    def forward(self, x):
        x = self.norm(x)
        for layer in self.classifier:
            out = layer(x)
        out = F.sigmoid(out).squeeze(-1)

        return out
    
class Gate(nn.Module):
    def __init__(self, proj_dim):
        super(Gate, self).__init__()
        
        self.gate = nn.Sequential(nn.Linear(proj_dim, 1),
                                  nn.Sigmoid()
                                  )
        
    def forward(self, x):
        feature_means = x.mean(dim=-1, keepdim=True)
        out = torch.where(feature_means != 0, self.gate(x), torch.zeros_like(feature_means))

        return out

class ResNetMIL(nn.Module):
    def __init__(self,
                 model_name: str = 'resnet50',
                 pretrained: bool = True,
                 progress: bool = False,
                 key: str = 'MoCoV2',
                 num_instance: int = 56,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_fc: int = 2,
                 dropout: float = 0.1):
        """
        params:
            pretrained: bool (whether to use pretrained model)
            progress: bool (whether to display progress bar)
            key: str (model key)
            num_patches: int (number of patches)
            num_heads: int (number of heads)
            num_layers: int (number of layers)
            dropout: float (dropout rate)
        """
        super(ResNetMIL, self).__init__()

        if model_name == 'resnet50':
            self.model = ResNet50(pretrained=pretrained, progress=progress, key=key, d_model=d_model)
        else:
            self.model = CNN(d_model, num_classes=2)

        self.pe = self.positional_encoding = nn.Embedding(num_instance + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.gate = Gate(proj_dim=d_model)

        self.projection = nn.Linear(d_model, 512)
        self.classifier = ClassificationHead(d_model, num_fc)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        b, n, _, _, _ = x.size()
        src = self.model(x)
        out_1 = self.projection(src)

        scores = self.gate(src)
        scores = scores.squeeze(-1)
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
 
        src = torch.gather(src, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, src.size(-1)))

        cls_tokens = self.cls_token.expand(b, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)

        pos = torch.arange(src.size(1)).unsqueeze(0).to(x.device)
        pos = self.pe(pos)
        src = src + pos
        
        src = self.attn(src)

        cls_token_output = src[:, 0]
        out_2 = self.classifier(cls_token_output)

        return out_1.view(b * n, -1), out_2
    

if __name__ == '__main__':
    model = ResNetMIL(pretrained=True, progress=False, key="MoCoV2").cuda()
    data = torch.rand((8, 32, 3, 224, 224)).cuda()
    out_1, out_2 = model(data)
