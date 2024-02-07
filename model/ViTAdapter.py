import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer


# Adapter 모듈 정의
class AdapterModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        params:
            input_dim: int (input dimension)
            hidden_dim: int (hidden dimension)
        """
        super(AdapterModule, self).__init__()

        self.up_project = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.down_project = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        original_x = x
        x = self.up_project(x)
        x = self.activation(x)
        x = self.down_project(x)

        return original_x + x


class AdapterBlock(nn.Module):
    def __init__(self, block, adapter):
        """
        params:
            block: nn.Module (block)
            adapter: AdapterModule (adapter)
        """
        super(AdapterBlock, self).__init__()

        self.block = block
        self.adapter = adapter

    def forward(self, x):
        x = self.block(x)
        x = self.adapter(x)
        return x


def add_adapters_to_vit(model, hidden_dim=64):
    """
    params:
        model: VisionTransformer (ViT model)
        hidden_dim: int (hidden dimension)
    """
    for i, block in enumerate(model.blocks):
        ffn_dim = block.mlp.fc2.out_features
        adapter = AdapterModule(ffn_dim, hidden_dim)
        adapted_block = AdapterBlock(block, adapter)
        model.blocks[i] = adapted_block


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=0
    )
    if pretrained:
        URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
        model_zoo_registry = {
            "DINO_p16": "dino_vit_small_patch16_ep200.torch",
            "DINO_p8": "dino_vit_small_patch8_ep200.torch",
        }
        pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )

    add_adapters_to_vit(model, hidden_dim=64)

    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


class ViT(nn.Module):
    def __init__(self, pretrained, progress, key, patch_size=16):
        """
        params:
            pretrained: bool (whether to use pretrained model)
            progress: bool (whether to display progress bar)
            key: str (model key)
            patch_size: int (patch size)
        """
        super(ViT, self).__init__()

        self.vit = vit_small(pretrained, progress=progress, key=key, patch_size=patch_size)
        self.classifier = nn.Sequential(nn.Linear(in_features=384, out_features=3),
                                        nn.Softmax(dim=-1)
                                        )

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        x_features = self.vit(x)
        x = self.classifier(x_features)

        x_features = x_features.view(b, n, -1)

        return x_features, x


if __name__ == "__main__":
    model = ViT(pretrained=True, progress=False, key="DINO_p16", patch_size=16).cuda()
    data = torch.rand((1, 16, 3, 224, 224)).cuda()

    features, out = model(data)
    print(features.shape)
    print(out.shape)
