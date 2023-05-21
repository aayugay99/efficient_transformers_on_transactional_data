import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score, Accuracy
import time
import wandb


def train_epoch(model, optimizer, dataloader, warmup=10, device="cuda"):
    model.train()
    model.to(device)

    metrics = {
        key: {
            "f1_score": MulticlassF1Score(
                num_classes=num_classes, 
                average="weighted", 
                ignore_index=0
            ), 
            "accuracy": Accuracy(
                task="multiclass", 
                num_classes=num_classes, 
                ignore_index=0
            )
        } for key, num_classes in model.num_classes_dict.items()
    }

    loss_epoch = 0
    count = 0 
    for batch_dict in dataloader:
        logits_dict = model(batch_dict, device=device)

        loss = 0
        for key, logits in logits_dict.items():
            y = batch_dict[key][:, warmup + 1:].to(device)
            logits_pred = logits[:, warmup: -1].permute(0, 2, 1)    # B x C x T

            loss += nn.functional.cross_entropy(logits_pred, y, ignore_index=0)
            
            y_pred = logits_pred.argmax(dim=1).to("cpu")
            metrics[key]["f1_score"].update(y_pred, y.to("cpu"))
            metrics[key]["accuracy"].update(y_pred, y.to("cpu"))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_count =  torch.sum((y != 0).float()).item()
        loss_epoch += loss.item() * cur_count
        count += cur_count

    return loss_epoch / count, {feature: {m: v.compute().item() for m, v in results.items()} for feature, results in metrics.items()}


def val_epoch(model, dataloader, warmup=10, device="cuda"):
    model.eval()
    model.to(device)

    metrics = {
        key: {
            "f1_score": MulticlassF1Score(
                num_classes=num_classes, 
                average="weighted", 
                ignore_index=0
            ), 
            "accuracy": Accuracy(
                task="multiclass", 
                num_classes=num_classes, 
                ignore_index=0
            )
        } for key, num_classes in model.num_classes_dict.items()
    }

    with torch.no_grad():
        loss_epoch = 0
        count = 0 
        for batch_dict in dataloader:
            logits_dict = model(batch_dict, device=device)

            loss = 0
            for key, logits in logits_dict.items():
                y = batch_dict[key][:, warmup + 1:].to(device)
                logits_pred = logits[:, warmup: -1].permute(0, 2, 1)
        
                loss += nn.functional.cross_entropy(logits_pred, y, ignore_index=0)

                y_pred = logits_pred.argmax(dim=1).to("cpu")
                metrics[key]["f1_score"].update(y_pred, y.to("cpu"))
                metrics[key]["accuracy"].update(y_pred, y.to("cpu"))

            cur_count = torch.sum((y != 0).float()).item()
            loss_epoch += loss.item() * cur_count
            count += cur_count    

    return loss_epoch / count, {feature: {m: v.compute().item() for m, v in results.items()} for feature, results in metrics.items()}


def train_model(model, optimizer, dataloaders, n_epochs, warmup=10, device="cuda"):
    for epoch in range(n_epochs):
        train_start = time.perf_counter()
        train_loss, train_metrics = train_epoch(model, optimizer, dataloaders["train"], warmup, device)
        train_end = time.perf_counter()
        val_loss, val_metrics = val_epoch(model, dataloaders["val"], warmup, device)
        val_end = time.perf_counter()

        wandb.log({
            "Epoch": epoch+1,
            "Train time": train_end - train_start,
            "Train loss": train_loss,
            "Train metrics": train_metrics,
            "Val time": val_end - train_end,
            "Val metrics": val_metrics,
            "Val loss": val_loss
        })
        
