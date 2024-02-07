import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt

from torchvision import transforms as T

from model.ViTMIL import ViTMIL
from model.ResNetMIL import ResNetMIL
from Dataset.MILDataset import PathologyDataset, collate_fn, data_split
from trainer.ModelTrainer import train, seed_everything
from trainer.ModelEvaluator import test_model
from util.augment import RandomRotation, RandomErasing
from util.scheduler import NoamLR

import matplotlib.pyplot as plt


def main(args):
    seed_everything(42)

    lr = args.lr
    model_type = args.model_type
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    dropout = args.dropout
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    train_num_sample = args.num_train_instance
    test_num_sample = args.num_test_instance
    csv_root = args.csv_root_dir
    loss_weight = args.loss_weight
    scheduler_type = args.scheduler
    gamma = args.gamma
    warmup = args.warmup_steps
    
    if model_type == 'ViT':
        model = ViTMIL(num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    else:
        model = ResNetMIL(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout)

    transform= T.Compose([T.ToTensor()
                          ])

    train_data, val_data, test_data = data_split(csv_root)
    train_dataset = PathologyDataset(df=train_data, num_samples=train_num_sample, transform=transform)
    dev_dataset = PathologyDataset(df=val_data, num_samples=test_num_sample, transform=transform)
    test_dataset = PathologyDataset(df=test_data, num_samples=test_num_sample, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)
    
    dev_loader = DataLoader(dev_dataset,
                            batch_size=test_batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True,
                             collate_fn=collate_fn)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])

    model.cuda()

    if isinstance(model, nn.DataParallel):
        optimizer = opt.Adam(model.module.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = opt.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    if scheduler_type == 'StepLR':
        scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=gamma)
    elif scheduler_type == 'Noam':
        scheduler = NoamLR(optimizer=optimizer, model_size=d_model, warmup_steps=warmup)
    else:
        scheduler = None

    criterion_bag = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_instance = nn.CrossEntropyLoss(label_smoothing=0.1)

    instance_loss_step, bag_loss_step = train(model=model,
                                              train_loader=train_loader,
                                              val_loader=dev_loader,
                                              criterion1=criterion_instance,
                                              criterion2=criterion_bag,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              num_epochs=num_epochs,
                                              loss_weight=loss_weight
                                              )

    plt.plot(instance_loss_step, label='instance pred loss')
    plt.plot(bag_loss_step, label='bag pred loss')
    plt.title("Training loss per step")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Loss.jpg', facecolor='#eeeeee')
    plt.close()


    test_model(model=model,
               data_loader=test_loader,
               criterion_1=criterion_instance,
               criterion_2=criterion_bag
               )