import os
from util.metric import calculate_metrics
import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate_model(model, data_loader, criterion_1, criterion_2):
    """
    params:
        model: nn.Module (model)
        criterion_1: nn.Module (instance loss)
        criterion_2: nn.Module (bag loss)
        data_loader: DataLoader (data loader)
    """
    total_loss_1 = 0
    total_loss_2 = 0
    total_samples = 0
    y_true = []
    y_pred = []

    pbar = tqdm(data_loader, desc=f'Validation', unit='batch')
    model.eval()
    with torch.no_grad():
        for inputs, targets, pseudo_targets in data_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            pseudo_targets = pseudo_targets.cuda()

            instance_pred, bag_pred = model(inputs)
            loss_1 = criterion_1(instance_pred, pseudo_targets.view(-1))
            loss_2 = criterion_2(bag_pred, targets)

            total_loss_1 += loss_1.item() * inputs.size(0)
            total_loss_2 += loss_2.item() * inputs.size(0)
            total_samples += inputs.size(0)

            y_true.extend(targets.detach().cpu().numpy())
            y_pred.extend(bag_pred[:, 1].detach().cpu().numpy())

            pbar.update(1)
            pbar.set_postfix_str(f'Instance Loss: {loss_1.item():.4f}, Bag Loss: {loss_2.item():.4f}')
            
            torch.cuda.empty_cache()
            del inputs, targets, pseudo_targets

    avg_loss_1 = total_loss_1 / total_samples
    avg_loss_2 = total_loss_2 / total_samples

    auroc, f1_score = calculate_metrics(y_true, y_pred)

    return avg_loss_1, avg_loss_2, auroc, f1_score


def test_model(model, data_loader, criterion_1, criterion_2):
        best_model, final_model = tuple(sorted(os.listdir('model_check_point')))
        final_path = os.path.join('model_check_point', final_model)
        best_path = os.path.join('model_check_point', best_model)

        # Load the best model weights
        model.load_state_dict(torch.load(best_path))
        best_loss_1, best_loss_2, best_auroc, best_f1_score = evaluate_model(model, data_loader, criterion_1, criterion_2)

        # Load the best model weights
        model.load_state_dict(torch.load(final_path))
        final_loss_1, final_loss_2, final_auroc, final_f1_score = evaluate_model(model, data_loader, criterion_1, criterion_2)

        # Print the evaluation metrics
        print("Final Instance Loss:", final_loss_1)
        print("Final Bag Loss:", final_loss_2)
        print("Final AUROC:", final_auroc)
        print("Final F1 Score:", final_f1_score)

        print('-' * 35)

        print("Best Instance Loss:", best_loss_1)
        print("Best Bag Loss:", best_loss_2)
        print("Best AUROC:", best_auroc)
        print("Best F1 Score:", best_f1_score)
