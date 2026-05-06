import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(3):
        for seq, labels in loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()