import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def evaluate(model, loader, device, name):
    model.eval()
    preds = []
    labels_all = []

    with torch.no_grad():
        for seq, labels in loader:
            seq = seq.to(device)
            outputs = model(seq)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            labels_all.extend(labels.numpy())

    acc = np.mean(np.array(preds) == np.array(labels_all))

    cm = confusion_matrix(labels_all, preds)
    plt.figure()
    plt.imshow(cm)
    plt.title(name)
    plt.savefig(os.path.join("outputs", f"confusion_{name}.png"))
    plt.close()

    return acc
