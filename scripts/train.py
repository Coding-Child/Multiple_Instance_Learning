from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import pandas as pd
from util.metric import calculate_metrics
from scripts.eval import evaluate_model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def train(model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, num_epochs, num_patience, loss_weight, fold):
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
    patience = 0
    min_loss = np.inf
    train_instance_loss_arr = list()
    train_bag_loss_arr = list()
    val_instance_loss_arr = list()
    val_bag_loss_arr = list()

    best_save_path = f'log/model_check_point/fold_{fold + 1}/best_model_fold_{fold + 1}.pth'
    final_save_path = f'log/model_check_point/fold_{fold + 1}/final_model_fold_{fold + 1}.pth'

    os.makedirs(f'log/predictions/fold_{fold + 1}', exist_ok=True)
    os.makedirs(f'log/model_check_point/fold_{fold + 1}', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()

        y_true = list()
        y_pred = list()

        total_loss = 0
        total_loss_1 = 0
        total_loss_2 = 0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='b', ascii=True, ncols=150) as pbar:
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()
                
                instance_pred, bag_pred = model(inputs)

                loss1 = criterion1(instance_pred)
                loss2 = criterion2(bag_pred, targets)
                loss = (loss1 * loss_weight) + (loss2 * (1 - loss_weight))

                total_loss += loss.item()
                total_loss_1 += loss1.item()
                total_loss_2 += loss2.item()

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()

                y_true.extend(targets.detach().cpu().numpy())
                y_pred.extend(bag_pred.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix_str(f"instance: {loss1:.4f}, bag: {loss2:.4f}, avg: {total_loss / (i + 1):.4f}")

                torch.cuda.empty_cache()
                del inputs, targets, loss1, loss2, loss

        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df.to_csv(f'log/predictions/fold_{fold + 1}/epoch_{epoch + 1}_train.csv', index=False)

        val_loss, val_instance_loss, val_bag_loss,val_auroc, val_f1_score, val_acc, val_precision, val_recall = evaluate_model(model, 
                                                                                                                               val_loader, 
                                                                                                                               criterion1, 
                                                                                                                               criterion2, 
                                                                                                                               fold=fold, 
                                                                                                                               epoch=epoch, 
                                                                                                                               loss_weight=loss_weight, 
                                                                                                                               phase='Validation')
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_instance_loss)

        avg_loss_1 = total_loss_1 / len(train_loader)
        avg_loss_2 = total_loss_2 / len(train_loader)
        avg_loss = total_loss / len(train_loader)

        train_instance_loss_arr.append(avg_loss_1)
        train_bag_loss_arr.append(avg_loss_2)
        val_instance_loss_arr.append(val_instance_loss)
        val_bag_loss_arr.append(val_bag_loss)

        if min_loss > val_loss:
            min_loss = val_loss
            auroc, f1_score, acc, precision, recall = calculate_metrics(y_true, y_pred)

            torch.save(model.module.state_dict(), best_save_path)
            print('Best Model Saved!! AUROC: {:.4f}, F1 Score: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(val_auroc, val_f1_score, val_acc, val_precision, val_recall))
            print(f'Train Model AUROC: {auroc:.4f}, F1 Score: {f1_score:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
            print(f'Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

            if num_patience != 0:
                patience = 0
        else:
            if num_patience != 0:
                patience += 1

        if num_patience != 0 and patience >= num_patience:
            print(f'Early Stopping at Epoch {epoch+1}')
            break

    torch.save(model.module.state_dict(), final_save_path)

    return train_instance_loss_arr, train_bag_loss_arr, val_instance_loss_arr, val_bag_loss_arr
