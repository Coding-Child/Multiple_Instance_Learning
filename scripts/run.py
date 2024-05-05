import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold
from torchvision.transforms import v2

from model.ResNetMIL import ResNetMIL
from Dataset.MILDataset import PathologyDataset, collate_fn
from scripts.train import train, seed_everything
from scripts.eval import test_model
from util.optimizer import select_optimizer, select_scheduler
from util.loss_fn import ContrastiveLoss

import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
matplotlib.use('Agg')


def main(args):
    seed_everything(args.seed)

    model_name = args.model_name
    lr = args.learning_rate
    loss_weight = args.loss_weight
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    num_fc = args.num_fc
    num_patience = args.num_patience
    pretrained = args.pretrained
    dropout = args.dropout
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_sample = args.num_instance
    csv_root = args.csv_root_dir
    sch = args.scheduler
    optim = args.optimizer
    gamma = args.gamma
    warmup_steps = args.warmup_steps

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    transform_train = v2.Compose([v2.RandomGrayscale(p=0.2),
                                  v2.RandomErasing(p=0.1),
                                  v2.RandomVerticalFlip(p=0.5),
                                  v2.RandomHorizontalFlip(p=0.5),
                                  v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True)
                                  ])
    transform_test = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.float32, scale=True)
                                 ])
    
    data = pd.read_csv(csv_root)
    test_data = pd.read_csv('camelyon_test_info.csv')
    test_dataset = PathologyDataset(df=test_data, num_samples=num_sample, transform=transform_test)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True, 
                             drop_last=False, 
                             collate_fn=collate_fn)

    criterion1 = ContrastiveLoss()
    criterion2 = nn.BCELoss()

    for i, (trn_idx, val_idx) in enumerate(kf.split(data)):
        print('-' * 150)
        print(f'Fold {i + 1} Start'.center(150))
        print('-' * 150)

        model = ResNetMIL(model_name=model_name,
                          num_instance=num_sample,
                          d_model=d_model,
                          num_heads=num_heads,
                          num_layers=num_layers,
                          dropout=dropout,
                          num_fc=num_fc,
                          pretrained=pretrained
                          )
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])
        model.cuda()

        train_data = data.iloc[trn_idx]
        val_data = data.iloc[val_idx]

        train_dataset = PathologyDataset(df=train_data, num_samples=num_sample, transform=transform_train)
        val_dataset = PathologyDataset(df=val_data, num_samples=num_sample, transform=transform_test)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=False,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn)

        optimizer = select_optimizer(model=model, lr=lr, opt=optim)
        scheduler = select_scheduler(optimizer=optimizer, train_loader=train_loader, gamma=gamma, d_model=d_model, warmup_steps=warmup_steps, scheduler=sch)

        instance_loss_epoch, bag_loss_epoch, val_instance_loss, val_bag_loss = train(model=model,
                                                                                     train_loader=train_loader,
                                                                                     val_loader=val_loader,
                                                                                     criterion1=criterion1,
                                                                                     criterion2=criterion2,
                                                                                     optimizer=optimizer,
                                                                                     scheduler=scheduler,
                                                                                     num_epochs=num_epochs,
                                                                                     num_patience=num_patience,
                                                                                     loss_weight=loss_weight,
                                                                                     fold=i
                                                                                     )

        os.makedirs('LossGraph', exist_ok=True)
        epochs = range(1, len(instance_loss_epoch) + 1)
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Instance Pred Loss", color=color)
        ax1.plot(epochs, instance_loss_epoch, label='Instance Pred Loss', color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.set_title("Training Loss")

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel("Bag Pred Loss", color=color)
        ax2.plot(epochs, bag_loss_epoch, label='Bag Pred Loss', color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.xlim(1, len(instance_loss_epoch))
        plt.xticks(range(1, len(instance_loss_epoch) + 1), rotation=45)

        plt.savefig(f'LossGraph/Loss_fold_{i + 1}.jpg', facecolor='#eeeeee')
        plt.close(fig)


        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Instance Pred Loss", color=color)
        ax1.plot(epochs, val_instance_loss, label='Instance Pred Loss', color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.set_title("Validation Loss")

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel("Bag Pred Loss", color=color)
        ax2.plot(epochs, val_bag_loss, label='Bag Pred Loss', color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.xlim(1, len(val_instance_loss))
        plt.xticks(range(1, len(val_instance_loss) + 1), rotation=45)

        plt.savefig(f'LossGraph/Val_Loss_fold_{i + 1}.jpg', facecolor='#eeeeee')
        plt.close(fig)

        test_model(model=model,
                   data_loader=test_loader,
                   criterion_1=criterion1,
                   criterion_2=criterion2,
                   fold=i,
                   loss_weight=loss_weight
                   )
