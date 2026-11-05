import numpy as np
import torch
from sklearn.metrics import f1_score


def fit_head(
    train_x,
    train_y,
    valid_x,
    valid_y,
    *,
    n_classes,
    seed,
    weight_decay,
    max_epochs=200,
    patience=20,
    device="cuda:0",
    fixed_epochs=None,
):
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    valid_x = np.asarray(valid_x)
    if train_x.ndim != 2 or valid_x.ndim != 2 or train_x.shape[1] != valid_x.shape[1]:
        raise ValueError("features must be 2D arrays with matching feature dimensions")
    if train_x.shape[0] == 0 or valid_x.shape[0] == 0:
        raise ValueError("training and query features must be nonempty")
    if train_y.ndim != 1 or train_y.shape[0] != train_x.shape[0]:
        raise ValueError("training labels must be a 1D array matching training features")
    if not np.isfinite(train_x).all() or not np.isfinite(valid_x).all():
        raise ValueError("features must contain only finite values")
    if not np.issubdtype(train_y.dtype, np.integer):
        raise ValueError("training labels must be integers")
    if n_classes < 2 or np.any(train_y < 0) or np.any(train_y >= n_classes):
        raise ValueError("training labels must be in the valid class range")
    counts = np.bincount(train_y.astype(np.int64), minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("every training class must be represented")
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    if not isinstance(epochs, int) or epochs < 1 or patience < 1:
        raise ValueError("epoch and patience values must be positive integers")
    train_features = torch.as_tensor(train_x, dtype=torch.float32, device=device).detach()
    query_features = torch.as_tensor(valid_x, dtype=torch.float32, device=device).detach()
    train_labels = torch.as_tensor(train_y, dtype=torch.long, device=device)
    train_features.requires_grad_(False)
    query_features.requires_grad_(False)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, n_classes),
    ).to(device)
    weights = len(train_y) / (n_classes * counts)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay)
    best_epoch = 0
    stopped_epoch = epochs
    best_score = -float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_features), train_labels)
        if not torch.isfinite(loss):
            raise ValueError("training loss must be finite")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise ValueError("training gradients must be finite")
        optimizer.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            predictions = model(query_features).argmax(dim=1).cpu().numpy()
        score = f1_score(valid_y, predictions, labels=np.arange(n_classes), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_epoch = epoch
                break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(query_features), dim=1).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("predicted probabilities must be finite")
    return {"probabilities": probabilities, "best_epoch": best_epoch, "stopped_epoch": stopped_epoch}
