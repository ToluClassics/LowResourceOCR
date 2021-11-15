from torch import nn
import torch
from torch.autograd import Variable

device = "cuda:0" if torch.cuda.is_available() == True else "cpu"


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def train(model, criterion, optimiser, scheduler, dataloader, tokenizer):
    model.train()
    total_loss = 0
    for batch, (imgs, labels_y,) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels_y = labels_y.to(device)

        optimiser.zero_grad()
        output = model(imgs.float(), labels_y.long()[:, :-1])

        norm = (labels_y != 0).sum()
        loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size),
                         labels_y[:, 1:].contiguous().view(-1).long()) / norm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        optimiser.step()
        total_loss += loss.item() * norm

    return total_loss / len(dataloader)


def evaluate(model, criterion, dataloader, tokenizer):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch, (imgs, labels_y,) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels_y = labels_y.to(device)

            output = model(imgs.float(), labels_y.long()[:, :-1])

            norm = (labels_y != 0).sum()
            loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size),
                             labels_y[:, 1:].contiguous().view(-1).long()) / norm

            epoch_loss += loss.item() * norm

    return epoch_loss / len(dataloader)