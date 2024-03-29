import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as opt

from sklearn.model_selection import KFold
from torchvision import transforms as T

from model.ResNetMIL import ResNetMIL
from Dataset.MILDataset import PathologyDataset, collate_fn, data_split
from trainer.ModelTrainer import train, seed_everything
from trainer.ModelEvaluator import test_model
from util.scheduler import NoamLR

import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')


def main(args):
    seed_everything(42)

    lr = args.lr
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    num_fc = args.num_fc
    dropout = args.dropout
    num_epochs = args.num_epochs
    num_patience = args.num_patience
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    train_num_sample = args.num_train_instance
    test_num_sample = args.num_test_instance
    csv_root = args.csv_root_dir
    loss_weight = args.loss_weight
    scheduler = args.scheduler
    gamma = args.gamma
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    transform= T.Compose([T.ToTensor()
                          ])

    train_data, test_data = data_split(csv_root, val=False)
    train_val_dataset = PathologyDataset(df=train_data, num_samples=train_num_sample, transform=transform)
    test_dataset = PathologyDataset(df=test_data, num_samples=test_num_sample, transform=transform)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True,
                             collate_fn=collate_fn)

    criterion_bag = nn.BCELoss()
    criterion_instance = nn.CrossEntropyLoss(label_smoothing=0.1)

    for i, (trn_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print('-' * 35)
        print(f'Fold {i + 1} Start')
        print('-' * 35)

        model = ResNetMIL(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout, num_fc=num_fc)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])
        model.cuda()

        optimizer = opt.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

        train_dataset = Subset(train_val_dataset, trn_idx)
        dev_dataset = Subset(train_val_dataset, val_idx)

        train_loader = DataLoader(train_dataset, 
                                  batch_size=train_batch_size, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=True, 
                                  drop_last=True, 
                                  collate_fn=collate_fn)
        dev_loader = DataLoader(dev_dataset, 
                                batch_size=train_batch_size, 
                                shuffle=False, 
                                num_workers=4,
                                pin_memory=True, 
                                drop_last=True, 
                                collate_fn=collate_fn)
        
        if scheduler:
            scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=gamma)
        else:
            scheduler = None

        instance_loss_step, bag_loss_step = train(model=model,
                                                  train_loader=train_loader,
                                                  val_loader=dev_loader,
                                                  criterion1=criterion_instance,
                                                  criterion2=criterion_bag,
                                                  optimizer=optimizer,
                                                  scheduler=scheduler,
                                                  num_epochs=num_epochs,
                                                  num_patience=num_patience,
                                                  loss_weight=loss_weight,
                                                  fold=i
                                                  )

        os.makedirs('LossGraph', exist_ok=True)
        plt.plot(instance_loss_step, label='instance pred loss')
        plt.plot(bag_loss_step, label='bag pred loss')
        plt.title("Training loss per step")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'LossGraph/Loss_fold_{i + 1}.jpg', facecolor='#eeeeee')
        plt.close()


        test_model(model=model,
                   data_loader=test_loader,
                   criterion_1=criterion_instance,
                   criterion_2=criterion_bag,
                   fold=i
                   )