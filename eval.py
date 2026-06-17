import torch
import torch.nn.functional as F


def macro_f1_score(y_true, y_pred):
    labels = torch.unique(torch.cat([y_true, y_pred])).tolist()
    if not labels:
        return 0.0

    f1_scores = []
    for label in labels:
        true_positive = ((y_pred == label) & (y_true == label)).sum().item()
        false_positive = ((y_pred == label) & (y_true != label)).sum().item()
        false_negative = ((y_pred != label) & (y_true == label)).sum().item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0

        f1_denominator = precision + recall
        f1_scores.append(2 * precision * recall / f1_denominator if f1_denominator else 0.0)

    return sum(f1_scores) / len(f1_scores)


def eval_metrics(loader, model, args):

    model.eval()

    correct = 0.
    loss = 0.
    preds = []
    targets = []

    for data in loader:
        data = data.to(args.device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
        preds.append(pred.cpu())
        targets.append(data.y.cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(targets)

    return {
        'acc': correct / len(loader.dataset),
        'loss': loss / len(loader.dataset),
        'f1_macro': macro_f1_score(y_true, y_pred),
    }


def eval(loader, model, args):
    metrics = eval_metrics(loader, model, args)
    return metrics['acc'], metrics['loss']
