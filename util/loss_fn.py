import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query):
        query_norm = F.normalize(query, dim=-1)
        similarity_matrix = torch.matmul(query_norm, query_norm.T)

        mask = torch.eye(query.size(0), dtype=torch.bool, device=query.device)
        similarity_matrix.masked_fill_(mask, -1e9)

        logits = similarity_matrix / self.temperature
        labels = torch.arange(query.size(0), device=query.device)

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class SmoothTop1SVMLoss(nn.Module):
    def __init__(self, alpha=1.0, tau=1.0):
        super(SmoothTop1SVMLoss, self).__init__()
        
        self.alpha = alpha
        self.tau = tau

    def forward(self, s, y):
        
        correct_scores = s.gather(1, y.unsqueeze(-1)).squeeze()
        L1_temp = torch.exp((self.alpha * (s != correct_scores[:, None]) + s - correct_scores[:, None]) / self.tau)
        loss = torch.log(L1_temp.sum(1)).mean()
        
        return loss
    

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()

        self.temperature = temperature
        self.nll_loss = nn.NLLLoss()
    
    def forward(self, features):
        sim = torch.matmul(features, features.T) / self.temperature

        log_prob = F.log_softmax(sim, dim=1)
        targets = torch.arange(0, features.size(0)).long().to(features.device)
        loss = self.nll_loss(log_prob, targets)

        return loss


if __name__ == '__main__':
    # Create a batch of 4 embeddings with 16 dimensions
    query = torch.randn(32, 128)

    # Initialize the InfoNCE loss with a temperature of 0.1
    loss_fn = InfoNCE(temperature=0.1)

    # Calculate the loss
    loss = loss_fn(query)

    print(loss.item())