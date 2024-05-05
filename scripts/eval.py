import os
from util.metric import calculate_metrics
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


def evaluate_model(model, data_loader, criterion1, criterion2, fold=None, epoch=None, loss_weight=None,
                   phase='Validation'):
    total_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0

    y_true = list()
    y_pred = list()

    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, desc=f'{phase}', unit='b', ascii=True, ncols=150) as pbar:
            for i, (_, inputs, targets) in enumerate(data_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()

                instance_pred, bag_pred = model(inputs)

                loss1 = criterion1(instance_pred)
                loss2 = criterion2(bag_pred, targets)
                loss = (loss1 * loss_weight) + (loss2 * (1 - loss_weight))

                total_loss += loss.item()
                total_loss_1 += loss1.item()
                total_loss_2 += loss2.item()

                y_true.extend(targets.detach().cpu().numpy())
                y_pred.extend(bag_pred.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix_str(f"instance: {loss1:.4f}, bag: {loss2:.4f}, avg: {total_loss / (i + 1):.4f}")

                torch.cuda.empty_cache()
                del inputs, targets, loss1, loss2, loss, instance_pred, bag_pred

    avg_loss_1 = total_loss_1 / len(data_loader)
    avg_loss_2 = total_loss_2 / len(data_loader)
    avg_loss = total_loss / len(data_loader)

    auroc, f1_score, acc, precision, recall = calculate_metrics(y_true, y_pred)

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    if phase == 'Validation':
        df.to_csv(f'log/predictions/fold_{fold + 1}/epoch_{epoch + 1}_{phase}.csv', index=False)
    else:
        df.to_csv(f'log/predictions/fold_{fold + 1}/{phase}.csv', index=False)

    return avg_loss, avg_loss_1, avg_loss_2, auroc, f1_score, acc, precision, recall


def test_model(model, data_loader, criterion_1, criterion_2, fold, loss_weight):
    final_path = f'log/model_check_point/fold_{fold + 1}/final_model_fold_{fold + 1}.pth'

    # Load the best model weights
    try:
        model.module.load_state_dict(torch.load(final_path))
    except:
        model.load_state_dict(torch.load(final_path))
    _, final_loss_1, final_loss_2, final_auroc, final_f1_score, final_acc, final_precision, final_recall = evaluate_model(
        model,
        data_loader,
        criterion_1,
        criterion_2,
        fold=fold,
        loss_weight=loss_weight,
        phase='Final Test')

    # Print the evaluation metrics
    print(f"Final Instance Loss: {final_loss_1:.4f}")
    print(f"Final Bag Loss: {final_loss_2:.4f}")
    print(f"Final AUROC: {final_auroc:.4f}")
    print(f"Final F1 Score: {final_f1_score:.4f}")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Final Precision: {final_precision:.4f}")
    print(f"Final Recall: {final_recall:.4f}")

    best_path = f'log/model_check_point/fold_{fold + 1}/best_model_fold_{fold + 1}.pth'
    # Load the best model weights
    try:
        model.module.load_state_dict(torch.load(best_path))
    except:
        model.load_state_dict(torch.load(best_path))
    _, best_loss_1, best_loss_2, best_auroc, best_f1_score, best_acc, best_precision, best_recall = evaluate_model(
        model,
        data_loader,
        criterion_1,
        criterion_2,
        fold=fold,
        loss_weight=loss_weight,
        phase='Best Test')

    print(f"Best Instance Loss: {best_loss_1:.4f}")
    print(f"Best Bag Loss: {best_loss_2:.4f}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Best F1 Score: {best_f1_score:.4f}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")
