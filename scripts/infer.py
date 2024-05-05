import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import pandas as pd

from model.ResNetMIL import ResNetMIL
from Dataset.MILDataset import PathologyDataset, collate_fn
from scripts.train import seed_everything
from scripts.eval import test_model
from util.loss_fn import ContrastiveLoss


def inference():
    seed_everything(42)

    transform_test = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.float32, scale=True)
                                 ])

    test_data = pd.read_csv('cmc_data_info.csv')
    test_dataset = PathologyDataset(df=test_data, num_samples=56, transform=transform_test)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=collate_fn
                             )

    criterion1 = ContrastiveLoss()
    criterion2 = nn.BCELoss()

    model = ResNetMIL(model_name='resnet50',
                      num_instance=56,
                      d_model=1024,
                      num_heads=16,
                      num_layers=6,
                      dropout=0.3,
                      num_fc=2,
                      pretrained=False
                      )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    test_model(model=model,
               data_loader=test_loader,
               criterion_1=criterion1,
               criterion_2=criterion2,
               fold=0,
               loss_weight=0.01
               )
