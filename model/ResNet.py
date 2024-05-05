import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetTrunk(ResNet):
    def __init__(self, block, layers, *args, **kwargs):
        super().__init__(block, layers, *args, **kwargs)
        del self.fc  # FC layer 제거

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress),
                              strict=False
                              )
        
    return model


class ResNet50(nn.Module):
    def __init__(self, pretrained, progress, key, d_model, **kwargs):
        """
        params:
            pretrained: bool (whether to use pretrained model)
            progress: bool (whether to display progress bar)
            key: str (model key)
            **kwargs: dict (other arguments)
        """
        super(ResNet50, self).__init__()

        self.d_model = d_model

        self.trunk = resnet50(pretrained, progress, key, **kwargs)
        self.projection = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(2048, d_model)
                                        )

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        mean_values = x.view(b * n, -1).mean(dim=1)
        zero_mask = mean_values == 0

        if zero_mask.any():
            x = x[~zero_mask]

            non_zero_features = self.trunk(x)
            non_zero_features = self.projection(non_zero_features)

            features = torch.zeros(b * n, self.d_model, device=x.device)
            features[~zero_mask] = non_zero_features
        else:
            features = self.trunk(x)
            features = self.projection(features)

        features = features.view(b, n, -1)

        return features
    

class CNN(nn.Module):
    def __init__(self, d_model, num_classes):
        super(CNN, self).__init__()

        self.d_model = d_model
        self.cnn = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2, stride=2),

                                 nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2, stride=2),

                                 nn.Conv2d(128, 256, kernel_size=3, stride=1),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2, stride=2),

                                 nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                 nn.ReLU(inplace=True),

                                 nn.AdaptiveMaxPool2d((1, 1)),
                                 nn.Flatten(),
                                 )

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        mean_values = x.view(b * n, -1).mean(dim=1)
        zero_mask = mean_values == 0

        if zero_mask.any():
            x = x[~zero_mask]

            non_zero_features = self.cnn(x)

            features = torch.zeros(b * n, self.d_model, device=x.device)
            features[~zero_mask] = non_zero_features
        else:
            features = self.cnn(x)

        features = features.view(b, n, -1)

        return features

if __name__ == '__main__':
    from torchsummary import summary

    model = ResNet50(pretrained=True, progress=True, key="MoCoV2").cuda()
    summary(model, (16, 3, 224, 224))
