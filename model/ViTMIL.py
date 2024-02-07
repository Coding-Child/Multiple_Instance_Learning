import torch
import torch.nn as nn

from model.Transformer import TransformerEncoder
from model.ViTAdapter import ViT


def create_padding_mask(features):
    """
    params:
        features: torch.Tensor (B, N, C) (feature tensor)
    """
    mask = torch.all(features == 0, dim=-1)
    mask = mask.transpose(1, 0)

    return mask


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        """
        params:
            p: float (p value)
            eps: float (epsilon value)
        """
        super(GeM, self).__init__()

        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=1).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class ViTMIL(nn.Module):
    def __init__(self, pretrained: bool = True,
                 progress: bool = False,
                 key: str = 'DINO_p16',
                 patch_size: int = 16,
                 num_patches: int = 32,
                 num_heads: int = 4,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        """
        params:
            pretrained: bool (whether to use pretrained model)
            progress: bool (whether to display progress bar)
            key: str (model key)
            patch_size: int (patch size)
            num_patches: int (number of patches)
            num_heads: int (number of heads)
            num_layers: int (number of layers)
            dropout: float (dropout rate)
        """
        super(ViTMIL, self).__init__()

        self.num_patches = num_patches
        self.vit = ViT(pretrained=pretrained, progress=progress, key=key, patch_size=patch_size)

        self.attn = TransformerEncoder(d_model=384, nhead=num_heads, dropout=dropout, num_layers=num_layers, dim_feedforward=384*4)

        self.pool = GeM()
        self.classifier = nn.Linear(384, 1)

    def forward(self, x):
        src, out_instance = self.vit(x)
        attn_mask = create_padding_mask(src)
        src = self.attn(src, src, src, src_key_padding_mask=attn_mask)
        
        out = self.pool(src)
        out = self.classifier(out)
        out = out.squeeze()

        return out_instance, out


if __name__ == '__main__':
    model = ViTMIL().cuda()
    data = torch.rand((8, 32, 3, 224, 224)).cuda()
    out_1, out_2 = model(data)
