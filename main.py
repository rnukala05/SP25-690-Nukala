import torch
from torch.utils.data import DataLoader
from data.dataset import ConfusionDataset
from models.cnn import CNNEncoder
from models.transformer import TransformerModel
from training.train import train_model
from evaluation.evaluate import evaluate
from utils.utils import set_seed
import matplotlib.pyplot as plt
import os

set_seed(42)

device = torch.device("cpu")

dataset = ConfusionDataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class CNNOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder()
        self.fc = torch.nn.Linear(128, 2)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1).mean(dim=1)
        return self.fc(x)

class CNNTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder()
        self.transformer = TransformerModel()

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        return self.transformer(x)

cnn_model = CNNOnly()
train_model(cnn_model, loader, device)
cnn_acc = evaluate(cnn_model, loader, device, "cnn")

transformer_model = CNNTransformer()
train_model(transformer_model, loader, device)
trans_acc = evaluate(transformer_model, loader, device, "transformer")

plt.figure()
plt.bar(["CNN", "Transformer"], [cnn_acc, trans_acc])
plt.savefig(os.path.join("outputs", "model_comparison.png"))
plt.close()

plt.figure()
plt.plot([cnn_acc]*3)
plt.savefig(os.path.join("outputs", "cnn_accuracy.png"))
plt.close()

plt.figure()
plt.plot([trans_acc]*3)
plt.savefig(os.path.join("outputs", "transformer_accuracy.png"))
plt.close()

print("CNN Accuracy:", cnn_acc)
print("Transformer Accuracy:", trans_acc)