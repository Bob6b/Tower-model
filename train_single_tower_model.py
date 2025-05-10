import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
import seaborn as sns
import pandas as pd


sns.set_theme(style="whitegrid")


class DrugInteractionDataset(Dataset):
    def __init__(self, h5_file, group='train'):
        with h5py.File(h5_file, 'r') as f:
            self.features = torch.tensor(f[group]['features'][:], dtype=torch.float32)
            self.labels = torch.tensor(f[group]['labels'][:], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_auc': []
    }

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'loss': running_loss / (total / inputs.size(0)),
                'acc': correct / total
            })

        # 训练指标
        metrics['train_loss'].append(running_loss / len(train_loader))
        metrics['train_acc'].append(correct / total)

        # 验证阶段
        model.eval()
        val_outputs = []
        val_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                val_outputs.append(outputs.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        # 处理验证结果
        val_outputs = np.concatenate(val_outputs)
        val_labels = np.concatenate(val_labels)

        # 计算验证指标
        val_preds = (val_outputs > 0.5).astype(float)
        metrics['val_loss'].append(val_loss / len(val_loader))
        metrics['val_acc'].append((val_preds == val_labels).mean())
        metrics['val_precision'].append(precision_score(val_labels, val_preds, zero_division=0))
        metrics['val_recall'].append(recall_score(val_labels, val_preds, zero_division=0))
        metrics['val_f1'].append(f1_score(val_labels, val_preds, zero_division=0))
        metrics['val_auc'].append(roc_auc_score(val_labels, val_outputs))

        # 打印详细指标
        print(f'Epoch {epoch + 1}: '
              f'Train Loss: {metrics["train_loss"][-1]:.4f} | '
              f'Val Loss: {metrics["val_loss"][-1]:.4f}\n'
              f'Train Acc: {metrics["train_acc"][-1]:.4f} | '
              f'Val Acc: {metrics["val_acc"][-1]:.4f}\n'
              f'Precision: {metrics["val_precision"][-1]:.4f} | '
              f'Recall: {metrics["val_recall"][-1]:.4f}\n'
              f'F1 Score: {metrics["val_f1"][-1]:.4f} | '
              f'AUC: {metrics["val_auc"][-1]:.4f}\n' +
              '-' * 60)

    return metrics


def plot_metrics(metrics):
    plt.figure(figsize=(18, 12))

    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(metrics['train_loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(2, 3, 2)
    plt.plot(metrics['train_acc'], label='Train')
    plt.plot(metrics['val_acc'], label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision-Recall曲线
    plt.subplot(2, 3, 3)
    plt.plot(metrics['val_precision'], label='Precision')
    plt.plot(metrics['val_recall'], label='Recall')
    plt.title('Validation Precision & Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    # F1 Score曲线
    plt.subplot(2, 3, 4)
    plt.plot(metrics['val_f1'], label='F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    # AUC曲线
    plt.subplot(2, 3, 5)
    plt.plot(metrics['val_auc'], label='AUC')
    plt.title('Validation AUC-ROC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def final_evaluation(model, loader):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    predicted = (all_outputs > 0.5).astype(int)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, predicted)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Interaction', 'Interaction'],
                yticklabels=['No Interaction', 'Interaction'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == "__main__":
    # 加载数据
    train_dataset = DrugInteractionDataset('../dataset_single_tower_model/dataset.h5', 'train')
    test_dataset = DrugInteractionDataset('../dataset_single_tower_model/dataset.h5', 'test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32)

    # 初始化模型
    model = MLPClassifier(input_dim=train_dataset.features.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    metrics = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100)

    # 绘制训练曲线
    plot_metrics(metrics)

    # 最终评估
    final_evaluation(model, val_loader)

    # 生成指标表格
    df = pd.DataFrame({
        'Epoch': range(1, len(metrics['train_loss']) + 1),
        'Train Loss': metrics['train_loss'],
        'Val Loss': metrics['val_loss'],
        'Train Acc': metrics['train_acc'],
        'Val Acc': metrics['val_acc'],
        'Val Precision': metrics['val_precision'],
        'Val Recall': metrics['val_recall'],
        'Val F1': metrics['val_f1'],
        'Val AUC': metrics['val_auc']
    })
    print("\n训练指标汇总:")
    print(df.round(4).to_markdown(index=False))