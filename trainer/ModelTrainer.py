from tqdm import tqdm
import os
import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import random
import pandas as pd
from sklearn.metrics import confusion_matrix
from trainer.ModelEvaluator import evaluate_model


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
    max_auroc = 0
    max_f1_score = 0
    train_instance_loss_arr = list()
    train_bag_loss_arr = list()

    suspicious_save_path = f'model_check_point/fold_{fold + 1}/suspicious_best_model_fold_{fold + 1}.pth'
    best_save_path = f'model_check_point/fold_{fold + 1}/best_model_fold_{fold + 1}.pth'
    final_save_path = f'model_check_point/fold_{fold + 1}/final_model_fold_{fold + 1}.pth'

    for epoch in range(num_epochs):
        model.train()

        y_true = list()
        y_pred = list()

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, (data, target, pseudo_target) in enumerate(pbar):
                data, target, pseudo_target = data.cuda(), target.cuda(), pseudo_target.cuda()

                instance_pred, bag_pred, _ = model(data)
                loss1 = criterion1(instance_pred, pseudo_target.view(-1))
                loss2 = criterion2(bag_pred, target)
                loss = (loss1 * loss_weight) + (loss2 * (1 - loss_weight))

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                train_instance_loss_arr.append(loss1.detach().cpu().numpy())
                train_bag_loss_arr.append(loss2.detach().cpu().numpy())

                y_true.extend(target.detach().cpu().numpy())
                y_pred.extend(bag_pred.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix_str(f"Instance Loss: {loss1:.4f}, Bag Loss: {loss2:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

                torch.cuda.empty_cache()
                del data, target, pseudo_target, loss1, loss2, loss
            
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        os.makedirs(f'confusion_matrix/fold_{fold + 1}', exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])
        cm_df.to_csv(f'confusion_matrix/fold_{fold + 1}/confusion_matrix_epoch_{epoch + 1}.csv', index=True)

        os.makedirs(f'model_check_point/fold_{fold + 1}', exist_ok=True)
        _, _, val_auroc, val_f1_score = evaluate_model(model, val_loader, criterion1, criterion2, fold=fold, epoch=epoch, phase='Validation')

        if val_auroc >= 0.99:
            # AUROC가 99% 이상인 경우의 특별 처리
            # 예: 다른 경로에 모델 저장, 경고 로그 출력 등
            torch.save(model.state_dict(), suspicious_save_path)
            print('Suspiciously High Performing Model Saved!! AUROC: {:.4f} | F1 Score: {:.4f}'.format(val_auroc, val_f1_score))
        elif val_auroc < 0.99 and (val_auroc > max_auroc or val_f1_score > max_f1_score):
            if val_auroc > max_auroc:
                max_auroc = val_auroc
            if val_f1_score > max_f1_score:
                max_f1_score = val_f1_score

            # 정상적인 범위 내에서의 모델 저장 처리
            torch.save(model.state_dict(), best_save_path)
            print('Best Model Saved!! AUROC: {:.4f} | F1 Score: {:.4f}'.format(max_auroc, max_f1_score))
        else:
            patience += 0

        if patience > num_patience:
            print(f'Early Stopping at Epoch {epoch+1}')
            break

    torch.save(model.state_dict(), final_save_path)

    return train_instance_loss_arr, train_bag_loss_arr
