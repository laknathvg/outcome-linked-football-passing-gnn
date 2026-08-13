from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
from .reproducibility import make_torch_generator, set_global_seed

@dataclass(frozen=True)
class TrainingSelection:
    best_epoch: int
    best_metric: float


def _optimizer(model: nn.Module, config: dict):
    return torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]),
                            weight_decay=float(config["training"]["weight_decay"]))


def _stop(config: dict):
    return (int(config["training"]["maximum_epochs"]),
            float(config["training"]["early_stopping"]["minimum_delta"]),
            int(config["training"]["early_stopping"]["patience"]))


def select_history_epoch(model_factory: Callable[[], nn.Module], X_train, y_train, X_val, y_val, config, seed, device):
    set_global_seed(seed)
    model, criterion = model_factory().to(device), nn.CrossEntropyLoss()
    optimizer = _optimizer(model, config)
    loader = TorchDataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
                             batch_size=int(config["training"]["batch_size"]), shuffle=True,
                             generator=make_torch_generator(seed))
    max_epochs, min_delta, patience = _stop(config)
    best, best_epoch, stale = -np.inf, 1, 0
    val_x = torch.tensor(X_val, dtype=torch.float32, device=device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(val_x).argmax(1).cpu().numpy()
        metric = f1_score(y_val, pred, average="macro", zero_division=0)
        if metric > best + min_delta:
            best, best_epoch, stale = metric, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return TrainingSelection(best_epoch, float(best))


def fit_history_model(model_factory, X_train, y_train, epochs, config, seed, device):
    set_global_seed(seed)
    model, criterion = model_factory().to(device), nn.CrossEntropyLoss()
    optimizer = _optimizer(model, config)
    loader = TorchDataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
                             batch_size=int(config["training"]["batch_size"]), shuffle=True,
                             generator=make_torch_generator(seed))
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
    return model


def predict_history_model(model, X, device):
    model.eval(); x = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(1).cpu().numpy(), torch.softmax(logits, 1)[:, 1].cpu().numpy()


def select_graph_epoch(model_factory, train_graphs, val_graphs, y_val, config, seed, device):
    set_global_seed(seed)
    model, criterion = model_factory().to(device), nn.CrossEntropyLoss()
    optimizer = _optimizer(model, config)
    train_loader = GraphDataLoader(train_graphs, batch_size=int(config["training"]["batch_size"]), shuffle=True,
                                   generator=make_torch_generator(seed))
    val_loader = GraphDataLoader(val_graphs, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    max_epochs, min_delta, patience = _stop(config)
    best, best_epoch, stale = -np.inf, 1, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad(); logits = model(batch)
            loss = criterion(logits, batch.y.view(-1)); loss.backward(); optimizer.step()
        model.eval(); pred = []
        with torch.no_grad():
            for batch in val_loader:
                pred.extend(model(batch.to(device)).argmax(1).cpu().numpy().tolist())
        metric = f1_score(y_val, np.asarray(pred), average="macro", zero_division=0)
        if metric > best + min_delta:
            best, best_epoch, stale = metric, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return TrainingSelection(best_epoch, float(best))


def fit_graph_model(model_factory, train_graphs, epochs, config, seed, device):
    set_global_seed(seed)
    model, criterion = model_factory().to(device), nn.CrossEntropyLoss()
    optimizer = _optimizer(model, config)
    loader = GraphDataLoader(train_graphs, batch_size=int(config["training"]["batch_size"]), shuffle=True,
                             generator=make_torch_generator(seed))
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device); optimizer.zero_grad(); logits = model(batch)
            loss = criterion(logits, batch.y.view(-1)); loss.backward(); optimizer.step()
    return model


def predict_graph_model(model, graphs, config, device):
    loader = GraphDataLoader(graphs, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    pred, score = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            pred.extend(logits.argmax(1).cpu().numpy().tolist())
            score.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
    return np.asarray(pred), np.asarray(score)
