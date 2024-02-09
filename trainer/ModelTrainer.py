from tqdm import tqdm
import os
import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import random
from trainer.ModelEvaluator import evaluate_model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def train(model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, num_epochs, loss_weight):
    """
    params:
        model: torch.nn.Module (model to train)
        train_loader: torch.utils.data.DataLoader (train data loader)
        val_loader: torch.utils.data.DataLoader (validation data loader)
        criterion1: torch.nn.Module (loss function for instance level)
        criterion2: torch.nn.Module (loss function for bag level)
        optimizer: torch.optim.Optimizer (optimizer)
        scheduler: torch.optim.lr_scheduler (scheduler)
        num_epochs: int (number of epochs)
    """
    min_loss = float('inf')
    max_auroc = 0
    max_f1_score = 0
    train_instance_loss_arr = []
    train_bag_loss_arr = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for batch_idx, (data, target, pseudo_target) in enumerate(pbar):
            data, target, pseudo_target = data.cuda(), target.cuda(), pseudo_target.cuda()

            instance_pred, bag_pred = model(data)
            loss1 = criterion1(instance_pred, pseudo_target.view(-1))
            loss2 = criterion2(bag_pred, target)
            loss = (loss1 * loss_weight) + (loss2 * (1 - loss_weight))

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            train_instance_loss_arr.append(loss1.detach().cpu().numpy())
            train_bag_loss_arr.append(loss2.detach().cpu().numpy())

            pbar.update(1)
            pbar.set_postfix_str(f"Training Loss: {train_loss / (batch_idx + 1):.4f} | Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

            torch.cuda.empty_cache()
            del data, target, pseudo_target, loss1, loss2, loss

        _, _, val_auroc, val_f1_score = evaluate_model(model, val_loader, criterion1, criterion2)
        
        if val_auroc > max_auroc or val_f1_score > max_f1_score:
            if val_auroc > max_auroc:
                max_auroc = val_auroc
            else:
                max_f1_score = val_f1_score

            torch.save(model.state_dict(), 'model_check_point/best_model.pth')
            print('Best Model Saved!! AUROC: {:.4f} | F1 Score: {:.4f}'.format(max_auroc, max_f1_score))

    torch.save(model.state_dict(), 'model_check_point/final_model.pth')

    return train_instance_loss_arr, train_bag_loss_arr
