import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ResNetAdapter import ResNet50


def create_padding_mask(features: torch.Tensor) -> torch.Tensor:
    means = features.mean(dim=2)

    mask = means == 0

    return mask

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

class ResNetMIL(nn.Module):
    def __init__(self,
                 pretrained: bool = True,
                 progress: bool = False,
                 key: str = 'MoCoV2',
                 d_model: int = 512,
                 num_heads: int = 4,
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

        self.resnet = ResNet50(pretrained=pretrained, progress=progress, key=key, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = ClassificationHead(d_model, num_fc)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        b = x.size(0)
        src, out_instance = self.resnet(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)

        attn_mask = create_padding_mask(src)
        src = self.attn(src, src_key_padding_mask=attn_mask)

        cls_token_output = src[:, 0]
        out = self.classifier(cls_token_output)

        return out_instance, out
    

if __name__ == '__main__':
    model = ResNetMIL(pretrained=True, progress=False, key="MoCoV2").cuda()
    data = torch.rand((8, 32, 3, 224, 224)).cuda()
    out_1, out_2 = model(data)
